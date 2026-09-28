"""Routing policies and the registry that builds them by name.

round-robin  rotate through replicas (baseline, no affinity)
mod-n        hash(conversation) % N (affinity until N changes)
consistent   hash ring (scaling moves ~1/N of conversations)
bounded      ring + cap of (1 + epsilon) x average load (Mirrokni, Thorup, Zadimoghaddam)
llm-aware    bounded + our LLM additions (see LLMAwarePolicy)
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .ring import HashRing, stable_hash


@dataclass(frozen=True)
class RouteRequest:
    conversation_id: str
    prompt_tokens: int  # estimated size of the whole conversation in this request
    max_tokens: int  # generation budget

    @property
    def cost(self) -> int:
        return self.prompt_tokens + self.max_tokens


@dataclass(frozen=True)
class Decision:
    replica: str
    reason: str  # "home", "overflow", "sticky", ...
    cost: int | None = None  # load to charge; None means RouteRequest.cost


class Policy:
    """Base class. ``params`` lists the settings a policy accepts."""

    name = ""
    params: tuple[str, ...] = ()

    def __init__(self, replicas: Iterable[str]) -> None:
        self.update_replicas(replicas)

    def update_replicas(self, replicas: Iterable[str]) -> None:
        # Sorted, so order-dependent policies don't depend on discovery order.
        self.replicas = tuple(sorted(set(replicas)))

    def choose(self, request: RouteRequest, load: Any) -> Decision:
        raise NotImplementedError

    def observe(self, request: RouteRequest, replica: str, success: bool) -> None:
        """Called after the replica answered. Only llm-aware uses it."""


class RoundRobinPolicy(Policy):
    name = "round-robin"

    def __init__(self, replicas: Iterable[str]) -> None:
        self._counter = 0
        super().__init__(replicas)

    def choose(self, request, load):
        replica = self.replicas[self._counter % len(self.replicas)]
        self._counter += 1
        return Decision(replica, "round-robin")


class ModNPolicy(Policy):
    name = "mod-n"

    def choose(self, request, load):
        return Decision(self.replicas[stable_hash(request.conversation_id) % len(self.replicas)], "home")


class ConsistentHashPolicy(Policy):
    name = "consistent"
    params = ("vnodes",)

    def __init__(self, replicas: Iterable[str], vnodes: int) -> None:
        self._ring = HashRing(vnodes)
        super().__init__(replicas)

    def update_replicas(self, replicas):
        super().update_replicas(replicas)
        self._ring.set_nodes(self.replicas)

    def choose(self, request, load):
        return Decision(self._ring.candidates(request.conversation_id)[0], "home")


class BoundedLoadsPolicy(ConsistentHashPolicy):
    """Go home unless home is above (1 + epsilon) x average load; then walk clockwise.

    Capacity includes the incoming request, as in the paper. ``load_metric``
    "requests" is the published algorithm; "tokens" weighs requests by size.
    """

    name = "bounded"
    params = ("vnodes", "epsilon", "load_metric")

    def __init__(self, replicas: Iterable[str], vnodes: int, epsilon: float, load_metric: str) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if load_metric not in ("requests", "tokens"):
            raise ValueError("load_metric must be 'requests' or 'tokens'")
        self._epsilon = epsilon
        self._load_metric = load_metric
        super().__init__(replicas, vnodes)

    def _load(self, load, replica: str) -> int:
        return load.requests(replica) if self._load_metric == "requests" else load.tokens(replica)

    def _capacity(self, load, cost: int, epsilon: float) -> float:
        total = sum(self._load(load, r) for r in self.replicas) + cost
        capacity = (1 + epsilon) * total / len(self.replicas)
        return math.ceil(capacity) if self._load_metric == "requests" else capacity

    @staticmethod
    def _fits(current: int, cost: int, capacity: float) -> bool:
        # An idle replica always accepts, so one oversized request can still be routed.
        return current == 0 or current + cost <= capacity

    def _least_loaded(self, load, order: list[str]) -> str:
        return min(order, key=lambda r: (self._load(load, r), order.index(r)))

    def choose(self, request, load):
        candidates = self._ring.candidates(request.conversation_id)
        cost = 1 if self._load_metric == "requests" else request.cost
        capacity = self._capacity(load, cost, self._epsilon)
        for position, replica in enumerate(candidates):
            if self._fits(self._load(load, replica), cost, capacity):
                return Decision(replica, "home" if position == 0 else "overflow")
        return Decision(self._least_loaded(load, candidates), "fallback-least-loaded")


class LLMAwarePolicy(BoundedLoadsPolicy):
    """Bounded loads plus three additions, each switchable for ablations:

    token_load        load = estimated tokens of actual work; on the replica holding
                      the cache a request costs only its new tokens, elsewhere its
                      whole history
    long_stickiness   long, probably-cached conversations tolerate up to
                      (1 + long_epsilon) x average before being redirected
    placement_memory  remember where each conversation was served (and its prompt
                      size) and keep it there instead of returning to its ring home
    """

    name = "llm-aware"
    params = ("vnodes", "epsilon", "long_epsilon", "long_threshold_tokens", "token_load",
              "long_stickiness", "placement_memory", "memory_max_entries")

    def __init__(self, replicas: Iterable[str], vnodes: int, epsilon: float, long_epsilon: float,
                 long_threshold_tokens: int, token_load: bool, long_stickiness: bool,
                 placement_memory: bool, memory_max_entries: int) -> None:
        if long_epsilon < epsilon:
            raise ValueError("long_epsilon must be >= epsilon")
        if long_threshold_tokens < 1 or memory_max_entries < 1:
            raise ValueError("long_threshold_tokens and memory_max_entries must be >= 1")
        self._long_epsilon = long_epsilon
        self._long_threshold = long_threshold_tokens
        self._long_stickiness = long_stickiness
        self._memory_enabled = placement_memory
        self._memory_max = memory_max_entries
        # conversation -> (replica, prompt tokens processed there), least recent first
        self._placement: OrderedDict[str, tuple[str, int]] = OrderedDict()
        super().__init__(replicas, vnodes, epsilon, "tokens" if token_load else "requests")

    def update_replicas(self, replicas):
        super().update_replicas(replicas)
        # A conversation whose replica disappeared has lost its cache anyway.
        for conversation, (replica, _) in list(self._placement.items()):
            if replica not in self.replicas:
                del self._placement[conversation]

    def _cost_on(self, request: RouteRequest, replica: str, placement) -> int:
        if self._load_metric == "requests":
            return 1
        if placement is not None and placement[0] == replica:
            return max(1, request.prompt_tokens - placement[1]) + request.max_tokens
        return request.cost

    def choose(self, request, load):
        candidates = self._ring.candidates(request.conversation_id)
        home = candidates[0]
        placement = self._placement.get(request.conversation_id) if self._memory_enabled else None
        if placement is not None and placement[0] not in self.replicas:
            placement = None
        preferred = placement[0] if placement else home
        # Without memory we can't know if a cache exists; assume it lives at home.
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
            if replica != preferred:
                other = self._cost_on(request, replica, placement)
                if self._fits(self._load(load, replica), other, self._capacity(load, other, self._epsilon)):
                    return Decision(replica, "overflow", other)
        fallback = self._least_loaded(load, candidates)
        return Decision(fallback, "fallback-least-loaded", self._cost_on(request, fallback, placement))

    def observe(self, request, replica, success):
        if self._memory_enabled and success:
            self._placement[request.conversation_id] = (replica, request.prompt_tokens)
            self._placement.move_to_end(request.conversation_id)
            while len(self._placement) > self._memory_max:
                self._placement.popitem(last=False)  # forget the least recently active


POLICIES: dict[str, type[Policy]] = {
    cls.name: cls for cls in (RoundRobinPolicy, ModNPolicy, ConsistentHashPolicy, BoundedLoadsPolicy, LLMAwarePolicy)
}
ALL_PARAMS = sorted({p for cls in POLICIES.values() for p in cls.params})


def build_policy(name: str, replicas: Iterable[str], settings: Mapping[str, Any],
                 overrides: Mapping[str, Any] | None = None) -> Policy:
    """Create a policy from the router settings, optionally overriding some of its params.

    Overrides must be params of that policy, so typos are reported, not ignored.
    """
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    cls = POLICIES[name]
    unknown = set(overrides or {}) - set(cls.params)
    if unknown:
        raise ValueError(f"policy {name!r} does not accept {sorted(unknown)}; accepted: {list(cls.params)}")
    values = {p: settings[p] for p in cls.params} | dict(overrides or {})
    return cls(replicas, **values)
