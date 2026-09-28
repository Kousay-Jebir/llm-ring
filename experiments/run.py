"""Run the main experiment.

For each policy: reset to 3 replicas, switch the router's policy, replay the same
traffic, and scale the StatefulSet to 4 replicas in the middle of the run. Each run
produces ``<run_id>.csv`` (one row per request) and ``<run_id>.json`` (metadata).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from traffic.generator import Profile
from traffic.generator import run as run_traffic

DEFAULT_POLICIES = ["consistent", "mod-n"]#, "consistent", "bounded", "llm-aware"]


def parse_policy_spec(spec: str) -> tuple[str, dict[str, Any]]:
    name, _, raw = spec.partition(":")
    params: dict[str, Any] = {}
    for item in filter(None, raw.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad parameter {item!r} in {spec!r}; expected key=value")
        lowered = value.lower()
        if lowered in ("true", "false"):
            params[key] = lowered == "true"
        else:
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
    return name.strip(), params


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class Kubectl:
    def __init__(self, namespace: str, statefulset: str) -> None:
        binary = shutil.which("kubectl")
        if binary is None:
            raise SystemExit("kubectl not found on PATH")
        self._binary, self._namespace, self._statefulset = binary, namespace, statefulset

    async def _run(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            self._binary, "-n", self._namespace, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await process.communicate()
        if process.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args)} failed: {err.decode().strip()}")
        return out.decode().strip()

    async def scale(self, replicas: int) -> None:
        await self._run("scale", f"statefulset/{self._statefulset}", f"--replicas={replicas}")

    async def ready(self) -> tuple[int, int]:
        """Return (desired replicas, ready replicas)."""
        out = await self._run("get", f"statefulset/{self._statefulset}",
                              "-o", "jsonpath={.spec.replicas} {.status.readyReplicas}")
        parts = out.split()
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0


class Router:
    def __init__(self, url: str) -> None:
        self._client = httpx.AsyncClient(base_url=url.rstrip("/"), timeout=10.0)

    async def check(self) -> None:
        try:
            (await self._client.get("/healthz")).raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(f"router not reachable ({exc}). Is `kubectl port-forward svc/router 8000:80` running?")

    async def state(self) -> dict[str, Any]:
        response = await self._client.get("/admin/state")
        response.raise_for_status()
        return response.json()

    async def set_policy(self, name: str, params: dict[str, Any]) -> None:
        response = await self._client.put("/admin/policy", json={"policy": name, "params": params})
        if response.status_code != 200:
            raise SystemExit(f"router rejected policy {name} {params}: {response.text}")

    async def aclose(self) -> None:
        await self._client.aclose()


async def wait_for_replicas(kubectl: Kubectl, router: Router, count: int, timeout_s: float = 900) -> None:
    """Wait until Kubernetes has exactly ``count`` ready replicas and the router sees them."""
    deadline = time.monotonic() + timeout_s
    while True:
        desired, ready = await kubectl.ready()
        seen = len((await router.state())["replicas"])
        if desired == ready == seen == count:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"expected {count} replicas; k8s desired={desired} ready={ready}, router sees {seen}")
        await asyncio.sleep(2)


async def reset_to(kubectl: Kubectl, router: Router, count: int) -> None:
    await kubectl.scale(count)
    await wait_for_replicas(kubectl, router, count)


async def scale_up_during_run(kubectl: Kubectl, router: Router, t0: float, at_s: float, to: int) -> dict[str, float]:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, t0 + at_s - loop.time()))
    command_s = loop.time() - t0
    await kubectl.scale(to)
    while len((await router.state())["replicas"]) < to:
        await asyncio.sleep(1)
    return {"scale_command_s": round(command_s, 3), "new_replica_ready_s": round(loop.time() - t0, 3)}


async def run_one(args: argparse.Namespace, profile: Profile, kubectl: Kubectl, router: Router,
                  spec: str, repeat: int, out_dir: Path) -> None:
    name, params = parse_policy_spec(spec)
    await reset_to(kubectl, router, args.base_replicas)
    await router.set_policy(name, params)  # also resets the policy's in-memory state

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(spec)}-r{repeat}"
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    meta: dict[str, Any] = {
        "run_id": run_id, "policy": spec, "policy_name": name, "policy_params": params,
        "repeat": repeat, "seed": args.seed, "profile": args.profile,
        "base_replicas": args.base_replicas, "scale_to": args.scale_to,
        "scale_at_s": args.scale_at, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    scale_task = None
    if args.scale_at > 0:
        scale_task = asyncio.create_task(scale_up_during_run(kubectl, router, t0, args.scale_at, args.scale_to))

    print(f"[{run_id}] running policy {spec} ...", flush=True)
    rows = await run_traffic(router_url=args.router_url, profile=profile, run_id=run_id, policy=spec,
                             out_csv=out_dir / f"{run_id}.csv", seed=args.seed, t0=t0,
                             conversation_prefix=f"s{args.seed}")
    meta["duration_s"] = round(loop.time() - t0, 3)

    if scale_task is not None:
        if scale_task.done():
            meta.update(scale_task.result())
        else:
            scale_task.cancel()
            meta["scale_skipped"] = "traffic finished before the scale-up; lower --scale-at"
            print(f"[{run_id}] WARNING: {meta['scale_skipped']}", flush=True)

    meta["requests"] = len(rows)
    meta["errors"] = sum(1 for r in rows if r.get("status") != 200)
    meta["router_state_at_end"] = await router.state()
    (out_dir / f"{run_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{run_id}] done: {meta['requests']} requests, {meta['errors']} errors", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    profile = Profile.load(args.profile)
    for spec in args.policies:
        parse_policy_spec(spec)  # fail fast on typos
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kubectl = Kubectl(args.namespace, args.statefulset)
    router = Router(args.router_url)
    try:
        await router.check()
        if args.prepare:
            # The first time llm-3 starts, it downloads the model; later starts reuse its
            # volume. Warming it once makes every run's scale-up cost the same.
            print("preparing: warming the extra replica's volume ...", flush=True)
            await reset_to(kubectl, router, args.scale_to)
            await reset_to(kubectl, router, args.base_replicas)
        for spec in args.policies:
            for repeat in range(args.repeats):
                await run_one(args, profile, kubectl, router, spec, repeat, out_dir)
                await asyncio.sleep(args.cooldown_s)
    finally:
        try:
            await kubectl.scale(args.base_replicas)
        finally:
            await router.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default="http://localhost:8000")
    parser.add_argument("--profile", default="traffic/profiles/standard.json")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-at", type=float, default=90.0, help="seconds after start; 0 disables")
    parser.add_argument("--base-replicas", type=int, default=3)
    parser.add_argument("--scale-to", type=int, default=4)
    parser.add_argument("--cooldown-s", type=float, default=10.0)
    parser.add_argument("--namespace", default="llm-ring")
    parser.add_argument("--statefulset", default="llm")
    parser.add_argument("--out-dir", default="experiments/results/raw")
    parser.add_argument("--prepare", action="store_true", help="warm the extra replica's volume first")
    args = parser.parse_args()
    if args.scale_to <= args.base_replicas and args.scale_at > 0:
        parser.error("--scale-to must be greater than --base-replicas")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
