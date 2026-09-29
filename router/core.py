"""Router logic, independent of the web framework: identify the conversation, choose a
replica, forward the request, account for load, log."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from .policies import RouteRequest, build_policy
from .ring import stable_hash

logger = logging.getLogger("llm_ring.router")

# ---- What travels between clients, the router and the replicas -----------------------
# Request headers, set by clients (the traffic generator):
CONVERSATION_HEADER = "x-conversation-id"  # which conversation this message belongs to
TURN_HEADER = "x-turn"  # turn number; only logged
# Response headers, added by the router to every answer:
REPLICA_HEADER = "x-routed-to"  # replica that served the request
REASON_HEADER = "x-route-reason"  # why that replica: home, overflow, sticky, ...
POLICY_HEADER = "x-route-policy"  # policy in effect (the conversation id is echoed too)
# Measurements llama.cpp puts in the "timings" object of every answer:
TIMINGS = (
    "cache_n",  # prompt tokens reused from the replica's cache (hits)
    "prompt_n",  # prompt tokens processed (misses)
    "prompt_ms",  # time spent processing them
    "predicted_n",  # tokens generated
    "predicted_ms",  # time spent generating them
)

# Rough rule for English text: ~4 characters per token, plus ~4 template tokens per
# message. Only relative sizes matter for routing. The traffic generator uses this too.
CHARS_PER_TOKEN = 4
TOKENS_PER_MESSAGE = 4
CONVERSATION_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")

# (url, body, headers) -> (status, body); transport failures raise.
Sender = Callable[[str, bytes, Mapping[str, str]], Awaitable[tuple[int, bytes]]]
Response = tuple[int, bytes, dict[str, str]]


def _text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):  # OpenAI-style content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return content if isinstance(content, str) else ""


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    return max(1, sum(len(_text(m)) for m in messages) // CHARS_PER_TOKEN + TOKENS_PER_MESSAGE * len(messages))


def conversation_id(header: str | None, messages: list[dict[str, Any]]) -> str:
    """The X-Conversation-ID header, or a hash of the first system and user messages
    (the same on every turn; two users starting identically would share it)."""
    if header:
        if not CONVERSATION_ID.match(header.strip()):
            raise ValueError("X-Conversation-ID must be 1-128 chars of [A-Za-z0-9._:-]")
        return header.strip()
    system = next((_text(m) for m in messages if m.get("role") == "system"), "")
    user = next((_text(m) for m in messages if m.get("role") == "user"), "")
    return f"auto-{stable_hash(json.dumps([system, user], ensure_ascii=False)):016x}"


class LoadTracker:
    """In-flight requests and estimated tokens per replica. A replica that disappears
    keeps its counters until its in-flight requests finish; it just stops being reported."""

    def __init__(self) -> None:
        self._current: set[str] = set()
        self._requests: Counter[str] = Counter()
        self._tokens: Counter[str] = Counter()

    def sync(self, replicas: Iterable[str]) -> None:
        self._current = set(replicas)

    def acquire(self, replica: str, tokens: int) -> None:
        self._requests[replica] += 1
        self._tokens[replica] += tokens

    def release(self, replica: str, tokens: int) -> None:
        self._requests[replica] = max(0, self._requests[replica] - 1)
        self._tokens[replica] = max(0, self._tokens[replica] - tokens)

    def requests(self, replica: str) -> int:
        return self._requests[replica]

    def tokens(self, replica: str) -> int:
        return self._tokens[replica]

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {r: {"requests": self._requests[r], "tokens": self._tokens[r]} for r in sorted(self._current)}


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> Response:
    return status, json.dumps({"error": {"message": message, "code": status}}).encode(), headers or {}


class RouterCore:
    def __init__(self, policy: str, policy_settings: Mapping[str, Any], sender: Sender, default_max_tokens: int) -> None:
        self._settings = dict(policy_settings)
        self._sender = sender
        self._default_max_tokens = default_max_tokens
        self.replicas: dict[str, str] = {}  # name -> base URL
        self._load = LoadTracker()
        self._policy = build_policy(policy, [], self._settings)
        self._overrides: dict[str, Any] = {}
        self._served: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()

    def set_replicas(self, replicas: Mapping[str, str]) -> None:
        if dict(replicas) == self.replicas:
            return
        added, removed = sorted(replicas.keys() - self.replicas.keys()), sorted(self.replicas.keys() - replicas.keys())
        self.replicas = dict(replicas)
        self._policy.update_replicas(self.replicas)
        self._load.sync(self.replicas)
        logger.info(json.dumps({"event": "replicas_changed", "replicas": sorted(self.replicas),
                                "added": added, "removed": removed}))

    def set_policy(self, name: str, overrides: Mapping[str, Any] | None = None) -> None:
        """Swap the policy (ValueError on bad name/params). Resets its state and the counters."""
        self._policy = build_policy(name, self.replicas, self._settings, overrides)
        self._overrides = dict(overrides or {})
        self._served.clear()
        self._reasons.clear()
        logger.info(json.dumps({"event": "policy_changed", "policy": name, "overrides": self._overrides}))

    def state(self) -> dict[str, Any]:
        return {"policy": self._policy.name, "policy_overrides": self._overrides, "replicas": self.replicas,
                "load": self._load.snapshot(), "served": dict(self._served), "reasons": dict(self._reasons)}

    async def handle_chat(self, raw_body: bytes, conversation_header: str | None, turn: str | None) -> Response:
        try:
            payload = json.loads(raw_body)
        except ValueError:
            return _error(400, "request body must be JSON")
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            return _error(400, "'messages' must be a non-empty list of objects")
        if payload.get("stream"):
            return _error(400, "streaming is not supported by this router")
        try:
            conv = conversation_id(conversation_header, messages)
        except ValueError as exc:
            return _error(400, str(exc))
        if not self.replicas:
            return _error(503, "no ready replicas")
        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            max_tokens = self._default_max_tokens
        request = RouteRequest(conv, estimate_tokens(messages), max_tokens)

        # No await between choosing and acquiring: on a single event loop, every
        # decision sees the load of every request routed before it.
        policy = self._policy
        decision = policy.choose(request, self._load)
        charged = request.cost if decision.cost is None else decision.cost
        self._load.acquire(decision.replica, charged)
        started, error = time.monotonic(), None
        try:
            status, body = await self._sender(f"{self.replicas[decision.replica]}/v1/chat/completions",
                                              raw_body, {"content-type": "application/json"})
        except Exception as exc:  # connection refused, timeout, ...
            status, body, error = 502, b"", f"{type(exc).__name__}: {exc}"
        finally:
            self._load.release(decision.replica, charged)  # always, or load leaks forever
        latency_ms = (time.monotonic() - started) * 1000

        success = 200 <= status < 300
        policy.observe(request, decision.replica, success)
        self._served[decision.replica] += 1
        self._reasons[decision.reason] += 1

        record = {"event": "request", "conversation_id": conv, "turn": turn, "policy": policy.name,
                  "replica": decision.replica, "reason": decision.reason, "status": status,
                  "latency_ms": round(latency_ms, 1), "est_prompt_tokens": request.prompt_tokens}
        if success:
            try:
                timings = json.loads(body).get("timings") or {}
            except (ValueError, AttributeError):
                timings = {}
            record |= {k: timings.get(k) for k in TIMINGS}
        if error:
            record["error"] = error
        if decision.trace:
            record["trace"] = list(decision.trace)
        logger.info(json.dumps(record))

        headers = {REPLICA_HEADER: decision.replica, REASON_HEADER: decision.reason,
                   POLICY_HEADER: policy.name, CONVERSATION_HEADER: conv}
        if error:
            return _error(502, f"upstream {decision.replica} failed: {error}", headers)
        return status, body, headers
