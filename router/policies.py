"""Routing policies and the registry that builds them by name.

round-robin  rotate through replicas (baseline, no affinity)
mod-n        hash(conversation) % N (affinity until N changes)
consistent   hash ring (scaling moves ~1/N of conversations)
bounded      ring + cap of (1 + epsilon) x average load (Mirrokni, Thorup, Zadimoghaddam)
llm-aware    bounded + our LLM additions (see LLMAwarePolicy)

Every decision carries a ``trace``: a short list of strings that record the numbers
and intermediate results the policy used. The router logs it with each request.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
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
    reason: str            # "home", "overflow", "sticky", ...
    cost: int | None = None  # load to charge; None means RouteRequest.cost
    trace: tuple[str, ...] = field(default=())  # steps + numbers that produced this decision


class Policy:
    """Base class. ``params`` lists the settings a policy accepts."""

    name = ""
    params: tuple[str, ...] = ()

    def __init__(self, replicas: Iterable[str]) -> None:
        self.update_replicas(replicas)

    def update_replicas(self, replicas: Iterable[str]) -> None:
        self.replicas = tuple(sorted(set(replicas)))

    def choose(self, request: RouteRequest, load: Any) -> Decision:
        raise NotImplementedError

    def observe(self, request: RouteRequest, replica: str, success: bool) -> None:
        """Called after the replica answered. Only llm-aware uses it."""


# ---------------------------------------------------------------------------
# Simple policies (no load awareness)
# ---------------------------------------------------------------------------

class RoundRobinPolicy(Policy):
    name = "round-robin"

    def __init__(self, replicas: Iterable[str]) -> None:
        self._counter = 0
        super().__init__(replicas)

    def choose(self, request, load):
        n, N = self._counter, len(self.replicas)
        replica = self.replicas[n % N]
        self._counter += 1
        return Decision(replica, "round-robin",
                        trace=(f"counter={n}, {n}%{N}={n % N} → {replica}",))


class ModNPolicy(Policy):
    name = "mod-n"

    def choose(self, request, load):
        h = stable_hash(request.conversation_id)
        N = len(self.replicas)
        idx = h % N
        replica = self.replicas[idx]
        return Decision(replica, "home",
                        trace=(f"stable_hash={h}, {h}%{N}={idx} → {replica}",))


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
        home = self._ring.candidates(request.conversation_id)[0]
        return Decision(home, "home",
                        trace=(f"ring → home={home}",))


# ---------------------------------------------------------------------------
# Load-aware policies
# ---------------------------------------------------------------------------

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
        cap = (1 + epsilon) * total / len(self.replicas)
        return math.ceil(cap) if self._load_metric == "requests" else cap

    @staticmethod
    def _fits(current: int, cost: int, capacity: float) -> bool:
        # An idle replica always accepts, so one oversized request can still be routed.
        return current == 0 or current + cost <= capacity

    def _least_loaded(self, load, order: list[str]) -> str:
        return min(order, key=lambda r: (self._load(load, r), order.index(r)))

    def _load_snapshot(self, load) -> str:
        return " ".join(f"{r}={self._load(load, r)}" for r in sorted(self.replicas))

    def _fmt(self, value: float) -> str:
        return str(int(value)) if self._load_metric == "requests" else f"{value:.1f}"

    def choose(self, request, load):
        candidates = self._ring.candidates(request.conversation_id)
        cost = 1 if self._load_metric == "requests" else request.cost
        cap = self._capacity(load, cost, self._epsilon)
        total = sum(self._load(load, r) for r in self.replicas) + cost
        t = [f"request: prompt={request.prompt_tokens} max={request.max_tokens} cost={cost} ({self._load_metric})",
             f"ring → home={candidates[0]}",
             f"load ({self._load_metric}): {self._load_snapshot(load)}; with this request total={total}, "
             f"cap=(1+{self._epsilon}) x {total}/{len(self.replicas)} = {self._fmt(cap)}"]
        for pos, replica in enumerate(candidates):
            cur = self._load(load, replica)
            fits = self._fits(cur, cost, cap)
            reason = "home" if pos == 0 else "overflow"
            if fits:
                note = "idle, always accepts" if cur == 0 and cur + cost > cap else f"{cur}+{cost}={cur + cost} ≤ {self._fmt(cap)}"
                t.append(f"{replica}: {note} → {reason}")
                return Decision(replica, reason, trace=tuple(t))
            t.append(f"{replica}: {cur}+{cost}={cur + cost} > {self._fmt(cap)} → skip")
        fallback = self._least_loaded(load, candidates)
        t.append(f"all over capacity → least loaded: {fallback}")
        return Decision(fallback, "fallback-least-loaded", trace=tuple(t))


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
        for conv, (replica, _) in list(self._placement.items()):
            if replica not in self.replicas:
                del self._placement[conv]

    def _cost_on(self, request: RouteRequest, replica: str, placement) -> int:
        """Estimated tokens of work if this request runs on ``replica``."""
        if self._load_metric == "requests":
            return 1
        if placement is not None and placement[0] == replica:
            # cache hit: only the new tokens need processing
            return max(1, request.prompt_tokens - placement[1]) + request.max_tokens
        return request.cost  # cache miss: full history + answer

    def choose(self, request, load):
        candidates = self._ring.candidates(request.conversation_id)
        home = candidates[0]
        placement = self._placement.get(request.conversation_id) if self._memory_enabled else None
        if placement is not None and placement[0] not in self.replicas:
            placement = None
        preferred = placement[0] if placement else home
        # Without memory we assume the conversation lives at its ring home.
        cache_likely = (placement is not None) if self._memory_enabled else True

        # --- build trace --------------------------------------------------------
        t: list[str] = []

        # 1. Request numbers
        t.append(f"request: prompt={request.prompt_tokens} max={request.max_tokens} "
                 f"cost_full={request.cost} tokens")

        # 2. Placement memory / ring home
        if placement:
            cached = placement[1]
            new_toks = max(1, request.prompt_tokens - cached)
            cost_on_preferred = new_toks + request.max_tokens
            t.append(f"placement memory: last served on {placement[0]} with {cached} prompt tokens → preferred={preferred}")
            t.append(f"cost on {preferred}: max(1, {request.prompt_tokens}-{cached})={new_toks} "
                     f"+ {request.max_tokens} = {cost_on_preferred} (new tokens only, cache hit)")
        else:
            reason_str = "memory disabled" if not self._memory_enabled else "new conversation"
            t.append(f"no placement ({reason_str}) → preferred={home} (ring home)")
            t.append(f"cost on {home}: {request.prompt_tokens}+{request.max_tokens}={request.cost} (full history, no cache)")
        # --- end placement section ----------------------------------------------

        cost = self._cost_on(request, preferred, placement)
        cap = self._capacity(load, cost, self._epsilon)
        total_load = sum(self._load(load, r) for r in self.replicas)

        # 3. Load snapshot and capacity
        t.append(f"load ({self._load_metric}): {self._load_snapshot(load)}; "
                 f"+{cost} → total={total_load + cost}, avg={(total_load + cost) / len(self.replicas):.1f}, "
                 f"cap={cap:.1f} (ε={self._epsilon})")

        # 4. Long-conversation stickiness
        limit = cap
        if self._long_stickiness and cache_likely and request.prompt_tokens >= self._long_threshold:
            limit = self._capacity(load, cost, self._long_epsilon)
            t.append(f"long stickiness: prompt_tokens={request.prompt_tokens} ≥ {self._long_threshold} "
                     f"and cache_likely={cache_likely} → long_cap={limit:.1f} (long_ε={self._long_epsilon})")
        elif self._long_stickiness:
            reasons = []
            if request.prompt_tokens < self._long_threshold:
                reasons.append(f"prompt_tokens={request.prompt_tokens} < {self._long_threshold}")
            if not cache_likely:
                reasons.append("cache_likely=False")
            t.append(f"long stickiness: not applied ({', '.join(reasons)}), using normal cap={cap:.1f}")

        # 5. Try preferred replica
        cur = self._load(load, preferred)
        if self._fits(cur, cost, limit):
            reason = "home" if preferred == home else "sticky"
            if not self._fits(cur, cost, cap):
                reason += "-long"
                t.append(f"{preferred}: {cur}+{cost}={cur+cost} ≤ long_cap={limit:.1f} "
                         f"(> normal cap={cap:.1f}) → {reason}")
            elif cur == 0 and cur + cost > limit:
                t.append(f"{preferred}: idle, always accepts (cost={cost} > cap={limit:.1f}) → {reason}")
            else:
                t.append(f"{preferred}: {cur}+{cost}={cur+cost} ≤ {limit:.1f} → {reason}")
            return Decision(preferred, reason, cost, trace=tuple(t))

        t.append(f"{preferred}: {cur}+{cost}={cur+cost} > {limit:.1f} → must redirect")

        # 6. Walk clockwise
        for replica in candidates:
            if replica == preferred:
                continue
            other_cost = self._cost_on(request, replica, placement)
            other_cap = self._capacity(load, other_cost, self._epsilon)
            other_cur = self._load(load, replica)
            cost_note = "full history, no cache there" if other_cost == request.cost else "new tokens only"
            if self._fits(other_cur, other_cost, other_cap):
                t.append(f"{replica}: cost={other_cost} ({cost_note}), "
                         f"{other_cur}+{other_cost}={other_cur+other_cost} ≤ {other_cap:.1f} → overflow")
                return Decision(replica, "overflow", other_cost, trace=tuple(t))
            t.append(f"{replica}: cost={other_cost} ({cost_note}), "
                     f"{other_cur}+{other_cost}={other_cur+other_cost} > {other_cap:.1f} → skip")

        fallback = self._least_loaded(load, candidates)
        t.append(f"all over capacity → least loaded: {fallback}")
        return Decision(fallback, "fallback-least-loaded",
                        self._cost_on(request, fallback, placement), trace=tuple(t))

    def observe(self, request, replica, success):
        if self._memory_enabled and success:
            self._placement[request.conversation_id] = (replica, request.prompt_tokens)
            self._placement.move_to_end(request.conversation_id)
            while len(self._placement) > self._memory_max:
                self._placement.popitem(last=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls for cls in (RoundRobinPolicy, ModNPolicy, ConsistentHashPolicy,
                               BoundedLoadsPolicy, LLMAwarePolicy)
}
ALL_PARAMS = sorted({p for cls in POLICIES.values() for p in cls.params})


def build_policy(name: str, replicas: Iterable[str], settings: Mapping[str, Any],
                 overrides: Mapping[str, Any] | None = None) -> Policy:
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    cls = POLICIES[name]
    unknown = set(overrides or {}) - set(cls.params)
    if unknown:
        raise ValueError(f"policy {name!r} does not accept {sorted(unknown)}; accepted: {list(cls.params)}")
    values = {p: settings[p] for p in cls.params} | dict(overrides or {})
    return cls(replicas, **values)