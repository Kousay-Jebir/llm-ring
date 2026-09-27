"""Framework-independent router logic: choose a replica, forward, account, log."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .conversation import conversation_id, estimate_tokens
from .load import LoadTracker
from .policies import RouteRequest, build_policy

# (url, body, headers) -> (status, body). Transport errors must be raised as exceptions.
Sender = Callable[[str, bytes, Mapping[str, str]], Awaitable[tuple[int, bytes]]]

logger = logging.getLogger("llm_ring.router")


@dataclass
class RoutedResponse:
    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> RoutedResponse:
    body = json.dumps({"error": {"message": message, "code": status}}).encode()
    return RoutedResponse(status, body, headers or {})


def _parse_timings(body: bytes) -> dict[str, Any]:
    try:
        timings = json.loads(body).get("timings") or {}
    except (ValueError, AttributeError):
        return {}
    return {k: timings.get(k) for k in ("cache_n", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}


class RouterCore:
    def __init__(
        self,
        policy_name: str,
        policy_defaults: Mapping[str, Any],
        sender: Sender,
        default_max_tokens: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sender = sender
        self._clock = clock
        self._defaults = dict(policy_defaults)
        self._default_max_tokens = default_max_tokens
        self._replicas: dict[str, str] = {}
        self._load = LoadTracker()
        self._policy = build_policy(policy_name, [], self._defaults)
        self._policy_overrides: dict[str, Any] = {}
        self._served: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()

    # -- replicas and policy -------------------------------------------------------

    @property
    def replicas(self) -> dict[str, str]:
        return dict(self._replicas)

    def set_replicas(self, replicas: Mapping[str, str]) -> bool:
        """Update the replica set. Returns True if it changed."""
        new = dict(replicas)
        if new == self._replicas:
            return False
        added, removed = sorted(new.keys() - self._replicas.keys()), sorted(self._replicas.keys() - new.keys())
        self._replicas = new
        self._policy.update_replicas(new)
        self._load.sync(new)
        logger.info(json.dumps({"event": "replicas_changed", "replicas": sorted(new), "added": added, "removed": removed}))
        return True

    def set_policy(self, name: str, overrides: Mapping[str, Any] | None = None) -> None:
        """Swap the policy (raises ValueError on invalid name/params). Resets its state."""
        policy = build_policy(name, self._replicas, self._defaults, overrides)
        self._policy = policy
        self._policy_overrides = dict(overrides or {})
        self._served.clear()
        self._reasons.clear()
        logger.info(json.dumps({"event": "policy_changed", "policy": name, "overrides": self._policy_overrides}))

    def state(self) -> dict[str, Any]:
        return {
            "policy": self._policy.name,
            "policy_overrides": self._policy_overrides,
            "replicas": self._replicas,
            "load": self._load.snapshot(),
            "served": dict(self._served),
            "reasons": dict(self._reasons),
        }

    # -- request path ----------------------------------------------------------------

    async def handle_chat(self, raw_body: bytes, conversation_header: str | None, turn_header: str | None) -> RoutedResponse:
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
            conv_id = conversation_id(conversation_header, messages)
        except ValueError as exc:
            return _error(400, str(exc))
        if not self._replicas:
            return _error(503, "no ready replicas")

        max_tokens = payload.get("max_tokens")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            max_tokens = self._default_max_tokens
        request = RouteRequest(conv_id, estimate_tokens(messages), max_tokens)

        # Decision and accounting happen with no await in between, so concurrent
        # requests on the event loop always see each other's load.
        policy = self._policy
        decision = policy.choose(request, self._load)
        base_url = self._replicas[decision.replica]
        charged = request.cost if decision.cost is None else decision.cost
        self._load.acquire(decision.replica, charged)
        started = self._clock()
        error = None
        try:
            status, body = await self._sender(
                f"{base_url}/v1/chat/completions", raw_body, {"content-type": "application/json"}
            )
        except Exception as exc:  # transport failure: connection refused, timeout, ...
            status, body, error = 502, b"", f"{type(exc).__name__}: {exc}"
        finally:
            self._load.release(decision.replica, charged)
        latency_ms = (self._clock() - started) * 1000

        success = 200 <= status < 300
        policy.observe(request, decision.replica, success)
        self._served[decision.replica] += 1
        self._reasons[decision.reason] += 1

        record = {
            "event": "request",
            "conversation_id": conv_id,
            "turn": turn_header,
            "policy": policy.name,
            "replica": decision.replica,
            "reason": decision.reason,
            "status": status,
            "latency_ms": round(latency_ms, 1),
            "est_prompt_tokens": request.prompt_tokens,
            **(_parse_timings(body) if success else {}),
        }
        if error:
            record["error"] = error
        logger.info(json.dumps(record))

        headers = {
            "x-routed-to": decision.replica,
            "x-route-reason": decision.reason,
            "x-route-policy": policy.name,
            "x-conversation-id": conv_id,
        }
        if error:
            return _error(502, f"upstream {decision.replica} failed: {error}", headers)
        return RoutedResponse(status, body, headers)
