from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .core import RouterCore
from .discovery import KubernetesDiscovery, StaticDiscovery
from .settings import Settings

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
logger = logging.getLogger("llm_ring.router")


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("llm_ring")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


class KubernetesApi:
    """Minimal in-cluster API client using the pod's ServiceAccount."""

    def __init__(self, sa_dir: Path = SERVICE_ACCOUNT_DIR) -> None:
        host = os.environ["KUBERNETES_SERVICE_HOST"]
        port = os.environ["KUBERNETES_SERVICE_PORT"]
        host = f"[{host}]" if ":" in host else host
        self._token_path = sa_dir / "token"
        context = ssl.create_default_context(cafile=str(sa_dir / "ca.crt"))
        self._client = httpx.AsyncClient(base_url=f"https://{host}:{port}", verify=context, timeout=5.0)

    async def get_json(self, path: str) -> dict[str, Any]:
        # Projected tokens are rotated by the kubelet, so read the file every time.
        token = self._token_path.read_text().strip()
        response = await self._client.get(path, headers={"Authorization": f"Bearer {token}"})
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()


async def _discovery_loop(core: RouterCore, discovery: Any, interval_s: float) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            core.set_replicas(await discovery.fetch())
        except Exception as exc:
            logger.warning(json.dumps({"event": "discovery_failed", "error": f"{type(exc).__name__}: {exc}"}))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        upstream = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.upstream_timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=1024, max_keepalive_connections=512),
        )

        async def send(url: str, body: bytes, headers: Any) -> tuple[int, bytes]:
            response = await upstream.post(url, content=body, headers=headers)
            return response.status_code, response.content

        core = RouterCore(settings.policy, settings.policy_params(), send, settings.default_max_tokens)
        kube_api: KubernetesApi | None = None
        if settings.discovery == "kubernetes":
            kube_api = KubernetesApi()
            discovery: Any = KubernetesDiscovery(
                kube_api.get_json, settings.k8s_namespace, settings.k8s_service, settings.k8s_port_name
            )
        else:
            discovery = StaticDiscovery(settings.static_replicas, settings.replica_url_template)

        try:
            core.set_replicas(await discovery.fetch())
        except Exception as exc:
            logger.warning(json.dumps({"event": "discovery_failed", "error": f"{type(exc).__name__}: {exc}"}))

        task = asyncio.create_task(_discovery_loop(core, discovery, settings.discovery_interval_s))
        app.state.core = core
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await upstream.aclose()
            if kube_api is not None:
                await kube_api.aclose()

    app = FastAPI(title="llm-ring router", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        replicas = request.app.state.core.replicas
        status = 200 if replicas else 503
        return JSONResponse({"ready": bool(replicas), "replicas": len(replicas)}, status_code=status)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        result = await request.app.state.core.handle_chat(
            await request.body(),
            request.headers.get("x-conversation-id"),
            request.headers.get("x-turn"),
        )
        return Response(content=result.body, status_code=result.status, media_type="application/json", headers=result.headers)

    @app.get("/admin/state")
    async def admin_state(request: Request) -> dict[str, Any]:
        return request.app.state.core.state()

    @app.put("/admin/policy")
    async def admin_policy(request: Request) -> JSONResponse:
        try:
            payload = json.loads(await request.body())
            name = payload["policy"]
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("'params' must be an object")
            request.app.state.core.set_policy(name, params)
        except (ValueError, KeyError, TypeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(request.app.state.core.state())

    return app
