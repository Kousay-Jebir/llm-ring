"""Run the experiment: for each policy, reset to the base replica count, switch the router's
policy, replay the same traffic, and add a replica mid-run. Each run writes <run_id>.csv
(one row per request) and <run_id>.json (metadata). Defaults come from settings.toml.

Needs the router reachable at router_url, e.g. `kubectl port-forward svc/router 8000:80`.
Policy specs can carry parameters: `llm-aware:token_load=false`, `bounded:epsilon=0.5`.

<run_id>.json fields:
  run_id, policy (spec as given), policy_name, policy_params, repeat, seed, profile,
  base_replicas, scale_to, scale_at_s, started_at      what was run
  scale_command_s       when `kubectl scale` was sent (seconds since start)
  new_replica_ready_s   when the router first saw the new replica; analysis uses it
  scale_skipped         set instead of the two above if traffic ended first
  duration_s, requests, errors, router_state_at_end   how it went
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

from traffic.generator import SETTINGS, Profile
from traffic.generator import run as run_traffic

EXP = SETTINGS["experiment"]


def parse_policy_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """'llm-aware:token_load=false,epsilon=0.5' -> ('llm-aware', {'token_load': False, 'epsilon': 0.5})"""
    name, _, raw = spec.partition(":")
    params: dict[str, Any] = {}
    for item in filter(None, raw.split(",")):
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad parameter {item!r} in {spec!r}; expected key=value")
        if value.lower() in ("true", "false"):
            params[key] = value.lower() == "true"
        else:
            for convert in (int, float, str):
                try:
                    params[key] = convert(value)
                    break
                except ValueError:
                    continue
    return name.strip(), params


async def kubectl(args: argparse.Namespace, *command: str) -> str:
    process = await asyncio.create_subprocess_exec(args.kubectl, "-n", args.namespace, *command,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(command)} failed: {err.decode().strip()}")
    return out.decode().strip()


async def router_replica_count(admin: httpx.AsyncClient) -> int:
    response = await admin.get("/admin/state")
    response.raise_for_status()
    return len(response.json()["replicas"])


async def reset_to(args: argparse.Namespace, admin: httpx.AsyncClient, count: int) -> None:
    """Scale to ``count`` and wait until Kubernetes has them all ready AND the router sees them."""
    await kubectl(args, "scale", f"statefulset/{args.statefulset}", f"--replicas={count}")
    deadline = time.monotonic() + EXP["replica_wait_timeout_s"]
    while True:
        parts = (await kubectl(args, "get", f"statefulset/{args.statefulset}", "-o",
                               "jsonpath={.spec.replicas} {.status.readyReplicas}")).split()
        desired, ready = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        seen = await router_replica_count(admin)
        if desired == ready == seen == count:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"expected {count} replicas; k8s desired={desired} ready={ready}, router sees {seen}")
        await asyncio.sleep(EXP["replica_poll_s"])


async def scale_up_during_run(args: argparse.Namespace, admin: httpx.AsyncClient, t0: float) -> dict[str, float]:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, t0 + args.scale_at - loop.time()))
    command_s = loop.time() - t0
    await kubectl(args, "scale", f"statefulset/{args.statefulset}", f"--replicas={args.scale_to}")
    while await router_replica_count(admin) < args.scale_to:
        await asyncio.sleep(EXP["scale_poll_s"])
    return {"scale_command_s": round(command_s, 3), "new_replica_ready_s": round(loop.time() - t0, 3)}


async def run_one(args: argparse.Namespace, profile: Profile, admin: httpx.AsyncClient, spec: str, repeat: int) -> None:
    name, params = parse_policy_spec(spec)
    await reset_to(args, admin, args.base_replicas)
    response = await admin.put("/admin/policy", json={"policy": name, "params": params})  # also resets its state
    if response.status_code != 200:
        raise SystemExit(f"router rejected policy {name} {params}: {response.text}")

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^a-z0-9]+', '-', spec.lower()).strip('-')}-r{repeat}"
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    meta: dict[str, Any] = {
        "run_id": run_id, "policy": spec, "policy_name": name, "policy_params": params, "repeat": repeat,
        "seed": args.seed, "profile": args.profile, "base_replicas": args.base_replicas, "scale_to": args.scale_to,
        "scale_at_s": args.scale_at, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    scale = asyncio.create_task(scale_up_during_run(args, admin, t0)) if args.scale_at > 0 else None
    print(f"[{run_id}] running policy {spec} ...", flush=True)
    rows = await run_traffic(router_url=args.router_url, profile=profile, run_id=run_id, policy=spec,
                             out_csv=Path(args.out_dir) / f"{run_id}.csv", seed=args.seed, t0=t0,
                             conversation_prefix=f"s{args.seed}")  # same IDs, so same ring positions, for every policy
    meta["duration_s"] = round(loop.time() - t0, 3)
    if scale is not None:
        if scale.done():
            meta.update(scale.result())
        else:
            scale.cancel()
            meta["scale_skipped"] = "traffic finished before the scale-up; lower --scale-at"
            print(f"[{run_id}] WARNING: {meta['scale_skipped']}", flush=True)
    meta["requests"] = len(rows)
    meta["errors"] = sum(r.get("status") != 200 for r in rows)
    meta["router_state_at_end"] = (await admin.get("/admin/state")).json()
    (Path(args.out_dir) / f"{run_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{run_id}] done: {meta['requests']} requests, {meta['errors']} errors", flush=True)


async def main_async(args: argparse.Namespace) -> None:
    profile = Profile.load(args.profile)
    for spec in args.policies:
        parse_policy_spec(spec)  # fail fast on typos
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(base_url=args.router_url.rstrip("/"), timeout=EXP["admin_timeout_s"]) as admin:
        try:
            (await admin.get("/healthz")).raise_for_status()
        except httpx.HTTPError as exc:
            raise SystemExit(f"router not reachable ({exc}). Is `kubectl port-forward svc/router 8000:80` running?")
        try:
            if args.prepare:
                # llm-3's first start downloads the model; later starts reuse its volume.
                # Warming it once makes every run's scale-up cost the same.
                print("preparing: warming the extra replica's volume ...", flush=True)
                await reset_to(args, admin, args.scale_to)
                await reset_to(args, admin, args.base_replicas)
            for spec in args.policies:
                for repeat in range(args.repeats):
                    await run_one(args, profile, admin, spec, repeat)
                    await asyncio.sleep(args.cooldown_s)
        finally:
            await kubectl(args, "scale", f"statefulset/{args.statefulset}", f"--replicas={args.base_replicas}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default=SETTINGS["router_url"])
    parser.add_argument("--profile", default=EXP["profile"])
    parser.add_argument("--policies", nargs="+", default=EXP["policies"])
    parser.add_argument("--repeats", type=int, default=EXP["repeats"])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--scale-at", type=float, default=EXP["scale_at_s"], help="seconds after start; 0 disables")
    parser.add_argument("--base-replicas", type=int, default=EXP["base_replicas"])
    parser.add_argument("--scale-to", type=int, default=EXP["scale_to"])
    parser.add_argument("--cooldown-s", type=float, default=EXP["cooldown_s"])
    parser.add_argument("--namespace", default=EXP["namespace"])
    parser.add_argument("--statefulset", default=EXP["statefulset"])
    parser.add_argument("--out-dir", default=EXP["out_dir"])
    parser.add_argument("--prepare", action="store_true", help="warm the extra replica's volume first")
    args = parser.parse_args()
    if args.scale_at > 0 and args.scale_to <= args.base_replicas:
        parser.error("--scale-to must be greater than --base-replicas")
    args.kubectl = shutil.which("kubectl") or parser.error("kubectl not found on PATH")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
