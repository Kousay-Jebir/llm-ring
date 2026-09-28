"""Conversation traffic generator.

Simulated users hold multi-turn chats through the router. Every turn resends the whole
conversation, like a chat client, and one CSV row per request records which replica
answered, why, and how many prompt tokens came from cache.

    python -m traffic.generator --policy-label consistent-manual
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from router.core import (CONVERSATION_HEADER, REASON_HEADER, REPLICA_HEADER, TIMINGS, TURN_HEADER,
                         estimate_tokens)

SETTINGS = tomllib.loads((Path(__file__).resolve().parents[1] / "settings.toml").read_text(encoding="utf-8"))

# One CSV row per request. Where each column comes from:
CSV_FIELDS = [
    "run_id", "policy", "conversation_id", "heavy", "turn",  # the traffic plan
    "t_start_s", "t_end_s", "status",  # measured here (seconds since the run started)
    "replica", "route_reason",  # router response headers
    "latency_ms",  # measured here: request sent -> answer received
    "est_prompt_tokens",  # estimate_tokens() of the conversation sent (the router's rule)
    *TIMINGS,  # llama.cpp "timings": cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms
    "error",  # failures only: the exception or the start of the response body
]

QUESTIONS = [
    "Can you summarize that in one sentence?", "What is the main risk with this approach?",
    "How would this work on Kubernetes?", "Give me a concrete example.",
    "What would you monitor in production?", "How does this affect cost?",
    "What happens if a node fails?", "Explain it to a new engineer.",
    "What is the first thing you would change?", "Which metric matters most here?",
    "How would you test this?", "What are the trade-offs compared to the alternative?",
]
OPENERS = [
    "How should I design autoscaling for a web API?",
    "What is the difference between a Deployment and a StatefulSet?",
    "How do I choose resource requests and limits?",
    "What is a good strategy for zero-downtime deployments?",
    "How does DNS work inside a Kubernetes cluster?", "When should I use a message queue?",
    "How do I debug a pod stuck in CrashLoopBackOff?", "What makes a good alerting rule?",
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
    max_context_tokens: int  # stop a conversation before it outgrows a replica's slot
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
    """Shared by all conversations of a run, like a real app. It starts with the run id,
    so nothing cached by a previous run can be reused: runs stay independent."""
    rng = random.Random(f"system-{run_id}")
    return (f"[run {run_id}] You are a concise assistant for a cloud platform team. "
            f"Reference notes: {_filler(rng, profile.system_prompt_words)}")


def plan_conversations(profile: Profile, id_prefix: str, seed: int) -> list[ConversationPlan]:
    """Deterministic: the same seed always gives the same traffic. Using the same
    ``id_prefix`` for every policy gives every policy the same ring positions."""
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
        thinks = [min(rng.expovariate(1 / profile.think_time_mean_s), profile.think_time_max_s) for _ in range(turns - 1)]
        plans.append(ConversationPlan(f"{id_prefix}-c{index:03d}", heavy, rng.uniform(0, profile.arrival_window_s),
                                      tuple(messages), tuple(thinks)))
    return plans


async def _conversation(plan: ConversationPlan, profile: Profile, run_id: str, policy: str, seed: int,
                        url: str, client: httpx.AsyncClient, t0: float, record) -> None:
    loop = asyncio.get_running_loop()
    await asyncio.sleep(max(0.0, t0 + plan.start_offset_s - loop.time()))
    history = [{"role": "system", "content": system_prompt(run_id, profile)}]
    for turn, user_message in enumerate(plan.user_messages, start=1):
        history.append({"role": "user", "content": user_message})
        est_tokens = estimate_tokens(history)
        if est_tokens + profile.max_tokens > profile.max_context_tokens:
            break
        # temperature 0 + fixed seed: same answers, so same later prompts, in every run
        payload = {"messages": history, "max_tokens": profile.max_tokens, "temperature": 0,
                   "seed": seed, "cache_prompt": True}
        started = loop.time()
        row: dict[str, Any] = {"run_id": run_id, "policy": policy, "conversation_id": plan.conversation_id,
                               "heavy": int(plan.heavy), "turn": turn, "t_start_s": round(started - t0, 3),
                               "est_prompt_tokens": est_tokens}
        try:
            response = await client.post(url, json=payload,
                                         headers={CONVERSATION_HEADER: plan.conversation_id, TURN_HEADER: str(turn)})
        except Exception as exc:  # network error: record it and stop this conversation
            record(row | {"status": 0, "error": f"{type(exc).__name__}: {exc}", "t_end_s": round(loop.time() - t0, 3),
                          "latency_ms": round((loop.time() - started) * 1000, 1)})
            return
        finished = loop.time()
        content, timings = None, {}
        if response.status_code == 200:
            try:
                data = response.json()
                content, timings = data["choices"][0]["message"]["content"], data.get("timings") or {}
            except (ValueError, KeyError, IndexError, TypeError):
                pass
        row |= {"status": response.status_code, "t_end_s": round(finished - t0, 3),
                "latency_ms": round((finished - started) * 1000, 1),
                "replica": response.headers.get(REPLICA_HEADER, ""), "route_reason": response.headers.get(REASON_HEADER, ""),
                **{k: timings.get(k) for k in TIMINGS}}
        if response.status_code != 200 or content is None:
            record(row | {"error": response.content[:300].decode("utf-8", "replace")})
            return
        record(row)
        history.append({"role": "assistant", "content": content})
        if turn <= len(plan.think_times_s):
            await asyncio.sleep(plan.think_times_s[turn - 1])


async def run(*, router_url: str, profile: Profile, run_id: str, policy: str, out_csv: str | Path, seed: int,
              t0: float | None = None, conversation_prefix: str | None = None,
              client: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
    """Run all conversations concurrently; one CSV row per request, written as it completes."""
    t0 = asyncio.get_running_loop().time() if t0 is None else t0
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(profile.request_timeout_s, connect=SETTINGS["generator"]["connect_timeout_s"]),
            limits=httpx.Limits(max_connections=SETTINGS["generator"]["max_connections"]))
    rows: list[dict[str, Any]] = []
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()

        def record(row: dict[str, Any]) -> None:
            rows.append(row)
            writer.writerow(row)
            handle.flush()  # partial results survive an interrupted run

        try:
            await asyncio.gather(*(
                _conversation(plan, profile, run_id, policy, seed, f"{router_url.rstrip('/')}/v1/chat/completions",
                              client, t0, record)
                for plan in plan_conversations(profile, conversation_prefix or run_id, seed)))
        finally:
            if own_client:
                await client.aclose()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--router-url", default=SETTINGS["router_url"])
    parser.add_argument("--profile", default=SETTINGS["generator"]["profile"])
    parser.add_argument("--policy-label", default="unknown", help="stored in the CSV for analysis")
    parser.add_argument("--run-id", default=None, help="default: timestamp + random suffix (never reuse one)")
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--out", default=None, help="default: experiments/results/<run-id>.csv")
    args = parser.parse_args()
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out = args.out or f"experiments/results/{run_id}.csv"
    rows = asyncio.run(run(router_url=args.router_url, profile=Profile.load(args.profile), run_id=run_id,
                           policy=args.policy_label, out_csv=out, seed=args.seed))
    print(json.dumps({"run_id": run_id, "requests": len(rows), "ok": sum(r.get("status") == 200 for r in rows), "csv": out}))


if __name__ == "__main__":
    main()
