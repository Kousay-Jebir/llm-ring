"""HTTP layer (FastAPI) and replica discovery through the Kubernetes API."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import ssl
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from .config import Config
from .core import CONVERSATION_HEADER, TURN_HEADER, RouterCore
from .policies import ALL_PARAMS

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")  # mounted in every pod
logger = logging.getLogger("llm_ring.router")


def parse_endpoint_slices(document: dict[str, Any], port_name: str) -> dict[str, str]:
    """EndpointSliceList -> {pod name: base URL}, ready and non-terminating endpoints only
    (unset "ready" means ready), so traffic drains before a pod goes away."""
    replicas = {}
    for slice_ in document.get("items") or []:
        ports = slice_.get("ports") or []
        port = next((p.get("port") for p in ports if p.get("name") == port_name), None)
        if port is None and len(ports) == 1:
            port = ports[0].get("port")
        if port is None:
            continue
        for endpoint in slice_.get("endpoints") or []:
            conditions = endpoint.get("conditions") or {}
            if conditions.get("ready") is False or conditions.get("terminating") is True:
                continue
            name = (endpoint.get("targetRef") or {}).get("name") or endpoint.get("hostname")
            addresses = endpoint.get("addresses") or []
            if name and addresses:
                host = f"[{addresses[0]}]" if ":" in addresses[0] else addresses[0]  # IPv6
                replicas[name] = f"http://{host}:{port}"
    return replicas


def create_app(config: Config | None = None) -> FastAPI:
    config = config or Config.from_env()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))  # records are already JSON
    logging.getLogger("llm_ring").handlers[:] = [handler]
    logging.getLogger("llm_ring").setLevel(config.log_level.upper())
    logging.getLogger("llm_ring").propagate = False

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        upstream = httpx.AsyncClient(
            timeout=httpx.Timeout(config.upstream_timeout_s, connect=config.upstream_connect_timeout_s),
            limits=httpx.Limits(max_connections=config.upstream_max_connections,
                                max_keepalive_connections=config.upstream_max_keepalive),
        )

        async def send(url: str, body: bytes, headers: Any) -> tuple[int, bytes]:
            response = await upstream.post(url, content=body, headers=headers)
            return response.status_code, response.content

        host = os.environ["KUBERNETES_SERVICE_HOST"]
        host = f"[{host}]" if ":" in host else host
        kube = httpx.AsyncClient(base_url=f"https://{host}:{os.environ['KUBERNETES_SERVICE_PORT']}",
                                 verify=ssl.create_default_context(cafile=str(SERVICE_ACCOUNT_DIR / "ca.crt")),
                                 timeout=config.k8s_api_timeout_s)
        selector = quote(f"kubernetes.io/service-name={config.k8s_service}", safe="")
        path = f"/apis/discovery.k8s.io/v1/namespaces/{config.k8s_namespace}/endpointslices?labelSelector={selector}"

        policy_settings = {name: getattr(config, name) for name in ALL_PARAMS}
        core = RouterCore(config.policy, policy_settings, send, config.default_max_tokens)

        async def discover() -> None:
            try:
                # The token is rotated by the kubelet, so it is re-read on every call.
                token = (SERVICE_ACCOUNT_DIR / "token").read_text().strip()
                response = await kube.get(path, headers={"Authorization": f"Bearer {token}"})
                response.raise_for_status()
                core.set_replicas(parse_endpoint_slices(response.json(), config.k8s_port_name))
            except Exception as exc:  # keep the last known replicas
                logger.warning(json.dumps({"event": "discovery_failed", "error": f"{type(exc).__name__}: {exc}"}))

        async def discovery_loop() -> None:
            while True:
                await asyncio.sleep(config.discovery_interval_s)
                await discover()

        await discover()
        task = asyncio.create_task(discovery_loop())
        app.state.core = core
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            await upstream.aclose()
            await kube.aclose()

    app = FastAPI(title="llm-ring router", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}  # process only: losing replicas must not restart the router

    @app.get("/readyz")
    async def readyz(request: Request) -> JSONResponse:
        count = len(request.app.state.core.replicas)
        return JSONResponse({"ready": count > 0, "replicas": count}, status_code=200 if count else 503)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        status, body, headers = await request.app.state.core.handle_chat(
            await request.body(), request.headers.get(CONVERSATION_HEADER), request.headers.get(TURN_HEADER))
        return Response(content=body, status_code=status, media_type="application/json", headers=headers)

    @app.get("/admin/state")
    async def admin_state(request: Request) -> dict[str, Any]:
        return request.app.state.core.state()

    @app.put("/admin/policy")
    async def admin_policy(request: Request) -> JSONResponse:
        try:
            payload = json.loads(await request.body())
            params = payload.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError("'params' must be an object")
            request.app.state.core.set_policy(payload["policy"], params)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return JSONResponse(request.app.state.core.state())

    return app
