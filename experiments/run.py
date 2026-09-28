"""Compare routing policies. For each policy: reset to the base replica count, switch the
router's policy, replay the same traffic, and add a replica mid-run.

Each run writes <run_id>.csv (one row per request) and <run_id>.json (what was run, and
new_replica_ready_s: when the router first saw the new replica). Defaults: settings.toml.
Policy specs can carry parameters: llm-aware:token_load=false  bounded:epsilon=0.5
"""

import argparse
import asyncio
import json
import re
import shutil
import time
from pathlib import Path

import httpx

from traffic.generator import SETTINGS, Profile
from traffic.generator import run as run_traffic

EXP = SETTINGS["experiment"]
KUBECTL = shutil.which("kubectl")


def parse_policy_spec(spec: str) -> tuple[str, dict]:
    """'llm-aware:token_load=false,epsilon=0.5' -> ('llm-aware', {'token_load': False, 'epsilon': 0.5})"""
    name, _, raw = spec.partition(":")
    params = {}
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
                    pass
    return name.strip(), params


async def kubectl(*command: str) -> str:
    process = await asyncio.create_subprocess_exec(KUBECTL, "-n", EXP["namespace"], *command,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(command)} failed: {err.decode().strip()}")
    return out.decode().strip()


async def scale(replicas: int) -> None:
    await kubectl("scale", f"statefulset/{EXP['statefulset']}", f"--replicas={replicas}")


async def router_replicas(admin: httpx.AsyncClient) -> int:
    response = await admin.get("/admin/state")
    response.raise_for_status()
    return len(response.json()["replicas"])


async def reset_to(admin: httpx.AsyncClient, count: int) -> None:
    """Scale to ``count`` and wait until Kubernetes has them ready AND the router sees them."""
    await scale(count)
    deadline = time.monotonic() + EXP["replica_wait_timeout_s"]
    while True:
        status = await kubectl("get", f"statefulset/{EXP['statefulset']}", "-o",
                               "jsonpath={.spec.replicas} {.status.readyReplicas}")
        desired, ready = (status.split() + ["0"])[:2]
        seen = await router_replicas(admin)
        if int(desired) == int(ready) == seen == count:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"expected {count} replicas; desired={desired} ready={ready}, router sees {seen}")
        await asyncio.sleep(EXP["replica_poll_s"])


async def scale_up(admin: httpx.AsyncClient, t0: float, at_s: float) -> dict:
    """At ``at_s`` seconds, add a replica; return when it was requested and when the router saw it."""
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, t0 + at_s - loop.time()))
    command_s = loop.time() - t0
    await scale(EXP["scale_to"])
    while await router_replicas(admin) < EXP["scale_to"]:
        await asyncio.sleep(EXP["scale_poll_s"])
    return {"scale_command_s": round(command_s, 3), "new_replica_ready_s": round(loop.time() - t0, 3)}


async def run_one(args: argparse.Namespace, profile: Profile, admin: httpx.AsyncClient, spec: str, repeat: int) -> None:
    name, params = parse_policy_spec(spec)
    await reset_to(admin, EXP["base_replicas"])
    response = await admin.put("/admin/policy", json={"policy": name, "params": params})  # also resets its state
    if response.status_code != 200:
        raise SystemExit(f"router rejected policy {spec}: {response.text}")

    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{re.sub(r'[^a-z0-9]+', '-', spec.lower()).strip('-')}-r{repeat}"
    meta = {"run_id": run_id, "policy": spec, "policy_name": name, "policy_params": params, "repeat": repeat,
            "seed": args.seed, "profile": args.profile, "base_replicas": EXP["base_replicas"],
            "scale_to": EXP["scale_to"], "scale_at_s": args.scale_at, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    t0 = asyncio.get_running_loop().time()
    scaling = asyncio.create_task(scale_up(admin, t0, args.scale_at)) if args.scale_at > 0 else None
    print(f"[{run_id}] running {spec} ...", flush=True)
    rows = await run_traffic(router_url=args.router_url, profile=profile, run_id=run_id, policy=spec,
                             out_csv=Path(args.out_dir) / f"{run_id}.csv", seed=args.seed, t0=t0,
                             conversation_prefix=f"s{args.seed}")  # same IDs, so same ring positions, for every policy
    meta["duration_s"] = round(asyncio.get_running_loop().time() - t0, 3)
    if scaling and scaling.done():
        meta.update(scaling.result())
    elif scaling:
        scaling.cancel()
        meta["scale_skipped"] = "the new replica wasn't ready before traffic ended (use --prepare, or lower --scale-at)"
        print(f"[{run_id}] WARNING: {meta['scale_skipped']}", flush=True)
    meta["requests"] = len(rows)
    meta["errors"] = sum(row.get("status") != 200 for row in rows)
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
            raise SystemExit(f"router not reachable at {args.router_url} ({exc})")
        try:
            if args.prepare:  # the extra replica's first start downloads the model; do it once, before measuring
                print("preparing: warming the extra replica's volume ...", flush=True)
                await reset_to(admin, EXP["scale_to"])
                await reset_to(admin, EXP["base_replicas"])
            for spec in args.policies:
                for repeat in range(args.repeats):
                    await run_one(args, profile, admin, spec, repeat)
                    await asyncio.sleep(args.cooldown_s)
        finally:
            await scale(EXP["base_replicas"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default=SETTINGS["router_url"])
    parser.add_argument("--profile", default=EXP["profile"])
    parser.add_argument("--policies", nargs="+", default=EXP["policies"])
    parser.add_argument("--repeats", type=int, default=EXP["repeats"])
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--scale-at", type=float, default=EXP["scale_at_s"], help="seconds after start; 0 disables")
    parser.add_argument("--cooldown-s", type=float, default=EXP["cooldown_s"])
    parser.add_argument("--out-dir", default=EXP["out_dir"])
    parser.add_argument("--prepare", action="store_true", help="warm the extra replica's volume first")
    args = parser.parse_args()
    if KUBECTL is None:
        parser.error("kubectl not found on PATH")
    if args.scale_at > 0 and EXP["scale_to"] <= EXP["base_replicas"]:
        parser.error("settings.toml: scale_to must be greater than base_replicas")
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()Remove-Item experiments/results/work_per_replica.png -ErrorAction SilentlyContinue
