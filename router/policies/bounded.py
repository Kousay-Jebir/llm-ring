"""Consistent hashing with bounded loads (Mirrokni, Thorup, Zadimoghaddam, 2016/2018)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from ..load import LoadTracker
from ..ring import HashRing
from .base import Decision, Policy, RouteRequest

LOAD_METRICS = ("requests", "tokens")


class BoundedLoadsPolicy(Policy):
    """Go to the key's home on the ring unless it is above capacity; then walk clockwise.

    Capacity is ``(1 + epsilon) x average load``, computed including the incoming
    request. With ``load_metric="requests"`` this is the published algorithm.
    ``load_metric="tokens"`` weighs requests by their estimated size instead.
    """

    name = "bounded"

    def __init__(
        self,
        replicas: Iterable[str],
        vnodes: int = 100,
        epsilon: float = 0.25,
        load_metric: str = "requests",
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if load_metric not in LOAD_METRICS:
            raise ValueError(f"load_metric must be one of {LOAD_METRICS}")
        self._epsilon = epsilon
        self._load_metric = load_metric
        self._ring = HashRing(vnodes=vnodes)
        super().__init__(replicas)

    def update_replicas(self, replicas: Iterable[str]) -> None:
        super().update_replicas(replicas)
        self._ring.set_nodes(self.replicas)

    # -- helpers shared with subclasses ------------------------------------------

    def _cost(self, request: RouteRequest) -> int:
        return 1 if self._load_metric == "requests" else request.cost

    def _load(self, load: LoadTracker, replica: str) -> int:
        return load.requests(replica) if self._load_metric == "requests" else load.tokens(replica)

    def _capacity(self, load: LoadTracker, cost: int, epsilon: float) -> float:
        total = sum(self._load(load, r) for r in self.replicas) + cost
        capacity = (1 + epsilon) * total / len(self.replicas)
        # The published algorithm works with integer request counts.
        return math.ceil(capacity) if self._load_metric == "requests" else capacity

    @staticmethod
    def _fits(current: int, cost: int, capacity: float) -> bool:
        # An idle replica always accepts, so a single request larger than the
        # capacity (possible with token weights) can still be routed.
        return current == 0 or current + cost <= capacity

    def _least_loaded(self, load: LoadTracker, order: Sequence[str]) -> str:
        rank = {replica: i for i, replica in enumerate(order)}
        return min(order, key=lambda r: (self._load(load, r), rank[r]))

    # -- policy ------------------------------------------------------------------

    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        candidates = self._ring.candidates(request.conversation_id)
        cost = self._cost(request)
        capacity = self._capacity(load, cost, self._epsilon)
        for position, replica in enumerate(candidates):
            if self._fits(self._load(load, replica), cost, capacity):
                return Decision(replica, "home" if position == 0 else "overflow")
        return Decision(self._least_loaded(load, candidates), "fallback-least-loaded")
