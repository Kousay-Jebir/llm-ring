"""Bounded loads adapted to LLM serving: our three additions.

1. Cache-aware token load: load is measured in estimated tokens of *actual work*.
   On the replica that holds a conversation's cache, a request only costs its new
   tokens; anywhere else it costs the whole history. Charging the full history
   everywhere (the naive version) makes long cached conversations look expensive,
   pushes them over capacity and redirects them, destroying their cache.
2. Long-conversation stickiness: a long conversation is only redirected when its
   replica is far above average (``long_epsilon``), because a redirect costs a
   full re-processing of its history.
3. Placement memory: the router remembers where each conversation was last served
   (and how long its prompt was), keeps it there instead of sending it back to its
   ring home after an overflow, and uses it to estimate the cached part.

Each addition can be switched off individually, for ablation experiments.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

from ..load import LoadTracker
from .base import Decision, RouteRequest
from .bounded import BoundedLoadsPolicy


@dataclass(frozen=True)
class _Placement:
    replica: str
    prompt_tokens: int  # size of the prompt last processed there (now cached)


class LLMAwarePolicy(BoundedLoadsPolicy):
    name = "llm-aware"

    def __init__(
        self,
        replicas: Iterable[str],
        vnodes: int = 100,
        epsilon: float = 0.25,
        long_epsilon: float = 1.0,
        long_threshold_tokens: int = 800,
        token_load: bool = True,
        long_stickiness: bool = True,
        placement_memory: bool = True,
        memory_max_entries: int = 50_000,
    ) -> None:
        if long_epsilon < epsilon:
            raise ValueError("long_epsilon must be >= epsilon")
        if long_threshold_tokens < 1:
            raise ValueError("long_threshold_tokens must be >= 1")
        if memory_max_entries < 1:
            raise ValueError("memory_max_entries must be >= 1")
        self._long_epsilon = long_epsilon
        self._long_threshold = long_threshold_tokens
        self._long_stickiness = long_stickiness
        self._memory_enabled = placement_memory
        self._memory_max = memory_max_entries
        self._placement: OrderedDict[str, _Placement] = OrderedDict()
        super().__init__(
            replicas,
            vnodes=vnodes,
            epsilon=epsilon,
            load_metric="tokens" if token_load else "requests",
        )

    def update_replicas(self, replicas: Iterable[str]) -> None:
        super().update_replicas(replicas)
        current = set(self.replicas)
        # A conversation whose replica disappeared has lost its cache anyway.
        for conversation, placement in list(self._placement.items()):
            if placement.replica not in current:
                del self._placement[conversation]

    @property
    def remembered_conversations(self) -> int:
        return len(self._placement)

    def _cost_on(self, request: RouteRequest, replica: str, placement: _Placement | None) -> int:
        """Work charged if ``request`` runs on ``replica``, given what is cached there."""
        if self._load_metric == "requests":
            return 1
        if placement is not None and placement.replica == replica:
            new_tokens = max(1, request.prompt_tokens - placement.prompt_tokens)
            return new_tokens + request.max_tokens
        return request.cost

    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        candidates = self._ring.candidates(request.conversation_id)
        home = candidates[0]

        placement = self._placement.get(request.conversation_id) if self._memory_enabled else None
        if placement is not None and placement.replica not in self.replicas:
            placement = None
        preferred = placement.replica if placement is not None else home
        # Without placement memory we cannot know whether a cache exists; we assume
        # the conversation has been living at its home replica.
        cache_likely = placement is not None if self._memory_enabled else True

        cost = self._cost_on(request, preferred, placement)
        capacity = self._capacity(load, cost, self._epsilon)
        limit = capacity
        if self._long_stickiness and cache_likely and request.prompt_tokens >= self._long_threshold:
            limit = self._capacity(load, cost, self._long_epsilon)

        current = self._load(load, preferred)
        if self._fits(current, cost, limit):
            reason = "home" if preferred == home else "sticky"
            if not self._fits(current, cost, capacity):
                reason += "-long"  # kept only thanks to long-conversation stickiness
            return Decision(preferred, reason, cost)

        for replica in candidates:
            if replica == preferred:
                continue
            other_cost = self._cost_on(request, replica, placement)
            if self._fits(self._load(load, replica), other_cost, self._capacity(load, other_cost, self._epsilon)):
                return Decision(replica, "overflow", other_cost)
        fallback = self._least_loaded(load, candidates)
        return Decision(fallback, "fallback-least-loaded", self._cost_on(request, fallback, placement))

    def observe(self, request: RouteRequest, replica: str, success: bool) -> None:
        if not (self._memory_enabled and success):
            return
        self._placement[request.conversation_id] = _Placement(replica, request.prompt_tokens)
        self._placement.move_to_end(request.conversation_id)
        while len(self._placement) > self._memory_max:
            self._placement.popitem(last=False)  # forget the least recently active
