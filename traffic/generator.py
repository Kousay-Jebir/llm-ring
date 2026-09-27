"""Conversation traffic generator.

Simulates users holding multi-turn chats through the router. Every turn resends the
whole conversation, exactly like a chat client, and records what the router and the
replica reported (which replica, why, how many prompt tokens came from cache).

Usage (from the repository root, with the router port-forwarded to localhost:8000):
    python -m traffic.generator --profile traffic/profiles/quick.json --out results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# (url, json payload, headers) -> (status, response headers, body)
PostFn = Callable[[str, dict[str, Any], dict[str, str]], Awaitable[tuple[int, Mapping[str, str], bytes]]]

CSV_FIELDS = [
    "run_id", "policy", "conversation_id", "heavy", "turn", "t_start_s", "t_end_s", "status",
    "replica", "route_reason", "latency_ms", "est_prompt_tokens", "cache_n", "prompt_n",
    "prompt_ms", "predicted_n", "predicted_ms", "error",
]

QUESTIONS = [
    "Can you summarize that in one sentence?",
    "What is the main risk with this approach?",
    "How would this work on Kubernetes?",
    "Give me a concrete example.",
    "What would you monitor in production?",
    "How does this affect cost?",
    "What happens if a node fails?",
    "Explain it to a new engineer.",
    "What is the first thing you would change?",
    "Which metric matters most here?",
    "How would you test this?",
    "What are the trade-offs compared to the alternative?",
]

OPENERS = [
    "How should I design autoscaling for a web API?",
    "What is the difference between a Deployment and a StatefulSet?",
    "How do I choose resource requests and limits?",
    "What is a good strategy for zero-downtime deployments?",
    "How does DNS work inside a Kubernetes cluster?",
    "When should I use a message queue?",
    "How do I debug a pod stuck in CrashLoopBackOff?",
    "What makes a good alerting rule?",
]

WORDS = (
    "cluster node pod replica service latency throughput cache request response queue "
    "deployment rollout scaling capacity storage volume network ingress metric alert "
    "incident budget region zone failover backup snapshot config secret token policy "
    "controller scheduler workload traffic error retry timeout limit quota version"
).split()


@dataclass(frozen=True)
class Profile:
    conversations: int
    heavy_fraction: float
    turns_min: int
    turns_max: int
    arrival_window_s: float
    think_time_mean_s: float
    think_time_max_s: float
    heavy_document_words: int
    system_prompt_words: int
    max_tokens: int
    max_context_tokens: int
    request_timeout_s: float
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        profile = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        if profile.conversations < 1 or not 1 <= profile.turns_min <= profile.turns_max:
            raise ValueError("invalid conversations/turns in profile")
        if not 0 <= profile.heavy_fraction <= 1:
            raise ValueError("heavy_fraction must be between 0 and 1")
        return profile


@dataclass(frozen=True)
class ConversationPlan:
    conversation_id: str
    heavy: bool
    start_offset_s: float
    user_messages: tuple[str, ...]
    think_times_s: tuple[float, ...]


def _filler(rng: random.Random, words: int) -> str:
    sentences, current = [], []
    for _ in range(words):
        current.append(rng.choice(WORDS))
        if len(current) >= rng.randint(8, 14):
            sentences.append(" ".join(current).capitalize() + ".")
            current = []
    if current:
        sentences.append(" ".join(current).capitalize() + ".")
    return " ".join(sentences)


def system_prompt(run_id: str, profile: Profile) -> str:
    """One system prompt shared by all conversations of a run, like a real app.

    It starts with the run id so that caches left by a previous run can never be
    reused, which keeps runs independent and comparisons fair.
    """
    rng = random.Random(f"system-{run_id}")
    return (
        f"[run {run_id}] You are a concise assistant for a cloud platform team. "
        f"Reference notes: {_filler(rng, profile.system_prompt_words)}"
    )


def plan_conversations(profile: Profile, id_prefix: str, seed: int) -> list[ConversationPlan]:
    """Deterministic plan: the same seed always produces the same traffic.

    ``id_prefix`` names the conversations. Using the same prefix for every policy
    gives every policy the same ring positions, hence the same hot spots.
    """
    plans = []
    heavy_count = round(profile.conversations * profile.heavy_fraction)
    for index in range(profile.conversations):
        rng = random.Random(f"{seed}-{index}")
        heavy = index < heavy_count
        turns = rng.randint(profile.turns_min, profile.turns_max)
        opener = rng.choice(OPENERS)
        if heavy:
            document = _filler(rng, profile.heavy_document_words)
            opener = f"Here is our internal design document:\n\n{document}\n\nPlease review it. {opener}"
        messages = [opener] + [rng.choice(QUESTIONS) for _ in range(turns - 1)]
        thinks = [min(rng.expovariate(1 / profile.think_time_mean_s), profile.think_time_max_s)
                  for _ in range(turns - 1)]
        plans.append(ConversationPlan(
            conversation_id=f"{id_prefix}-c{index:03d}",
            heavy=heavy,
            start_offset_s=rng.uniform(0, profile.arrival_window_s),
            user_messages=tuple(messages),
            think_times_s=tuple(thinks),
        ))
    return plans


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    # Same heuristic as the router, to stay consistent.
    return max(1, sum(len(m["content"]) for m in messages) // 4 + 4 * len(messages))


def _parse_response(body: bytes) -> tuple[str | None, dict[str, Any]]:
    try:
        data = json.loads(body)
        content = data["choices"][0]["message"]["content"]
        return content, data.get("timings") or {}
    except (ValueError, KeyError, IndexError, TypeError):
        return None, {}


async def _run_conversation(plan: ConversationPlan, *, profile: Profile, run_id: str, policy: str,
                            seed: int, router_url: str, post: PostFn, t0: float,
                            record: Callable[[dict[str, Any]], None]) -> None:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, t0 + plan.start_offset_s - loop.time()))
    history = [{"role": "system", "content": system_prompt(run_id, profile)}]
    for turn, user_message in enumerate(plan.user_messages, start=1):
        history.append({"role": "user", "content": user_message})
        est = estimate_tokens(history) + profile.max_tokens
        if est > profile.max_context_tokens:
            break  # stay inside the replica's per-slot context window
        payload = {
            "messages": history,
            "max_tokens": profile.max_tokens,
            "temperature": 0,  # deterministic answers: same traffic in every run
            "seed": seed,
            "cache_prompt": True,
        }
        headers = {"X-Conversation-ID": plan.conversation_id, "X-Turn": str(turn)}
        started = loop.time()
        row: dict[str, Any] = {
            "run_id": run_id, "policy": policy, "conversation_id": plan.conversation_id,
            "heavy": int(plan.heavy), "turn": turn, "t_start_s": round(started - t0, 3),
            "est_prompt_tokens": estimate_tokens(history),
        }
        try:
            status, resp_headers, body = await post(f"{router_url}/v1/chat/completions", payload, headers)
        except Exception as exc:  # network error: record and stop this conversation
            row.update(status=0, error=f"{type(exc).__name__}: {exc}", t_end_s=round(loop.time() - t0, 3),
                       latency_ms=round((loop.time() - started) * 1000, 1))
            record(row)
            return
        finished = loop.time()
        lowered = {k.lower(): v for k, v in resp_headers.items()}
        content, timings = _parse_response(body) if status == 200 else (None, {})
        row.update(
            status=status, t_end_s=round(finished - t0, 3), latency_ms=round((finished - started) * 1000, 1),
            replica=lowered.get("x-routed-to", ""), route_reason=lowered.get("x-route-reason", ""),
            cache_n=timings.get("cache_n"), prompt_n=timings.get("prompt_n"),
            prompt_ms=timings.get("prompt_ms"), predicted_n=timings.get("predicted_n"),
            predicted_ms=timings.get("predicted_ms"),
        )
        if status != 200 or content is None:
            row["error"] = body[:300].decode("utf-8", "replace")
            record(row)
            return
        record(row)
        history.append({"role": "assistant", "content": content})
        if turn <= len(plan.think_times_s):
            await asyncio.sleep(plan.think_times_s[turn - 1])


def _httpx_post(timeout_s: float) -> tuple[PostFn, Callable[[], Awaitable[None]]]:
    import httpx  # imported lazily so the planning logic has no dependency

    client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0),
                               limits=httpx.Limits(max_connections=200))

    async def post(url: str, payload: dict[str, Any], headers: dict[str, str]):
        response = await client.post(url, json=payload, headers=headers)
        return response.status_code, response.headers, response.content

    return post, client.aclose


async def run(*, router_url: str, profile: Profile, run_id: str, policy: str, out_csv: str | Path,
              seed: int = 42, post: PostFn | None = None, t0: float | None = None,
              conversation_prefix: str | None = None) -> list[dict[str, Any]]:
    """Run all conversations concurrently; write one CSV row per request as it completes.

    ``conversation_prefix`` defaults to ``run_id``. Cache isolation between runs does
    not depend on it: the system prompt always embeds the run id.
    """
    loop = asyncio.get_running_loop()
    t0 = loop.time() if t0 is None else t0
    close: Callable[[], Awaitable[None]] | None = None
    if post is None:
        post, close = _httpx_post(profile.request_timeout_s)

    rows: list[dict[str, Any]] = []
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        def record(row: dict[str, Any]) -> None:
            rows.append(row)
            writer.writerow(row)
            handle.flush()  # partial results survive an interrupted run

        try:
            await asyncio.gather(*(
                _run_conversation(plan, profile=profile, run_id=run_id, policy=policy, seed=seed,
                                  router_url=router_url.rstrip("/"), post=post, t0=t0, record=record)
                for plan in plan_conversations(profile, conversation_prefix or run_id, seed)
            ))
        finally:
            if close is not None:
                await close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default="http://localhost:8000")
    parser.add_argument("--profile", default="traffic/profiles/quick.json")
    parser.add_argument("--policy-label", default="unknown", help="stored in the CSV for analysis")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = args.out or f"experiments/results/{run_id}.csv"
    profile = Profile.load(args.profile)
    rows = asyncio.run(run(router_url=args.router_url, profile=profile, run_id=run_id,
                           policy=args.policy_label, out_csv=out, seed=args.seed))
    ok = sum(1 for r in rows if r.get("status") == 200)
    print(json.dumps({"run_id": run_id, "requests": len(rows), "ok": ok, "csv": out, "profile": asdict(profile)}))


if __name__ == "__main__":
    main()
