from __future__ import annotations

from collections.abc import Iterable

from ..load import LoadTracker
from ..ring import HashRing
from .base import Decision, Policy, RouteRequest


class ConsistentHashPolicy(Policy):
    """Consistent hashing: affinity that survives scaling, but no load awareness."""

    name = "consistent"

    def __init__(self, replicas: Iterable[str], vnodes: int = 100) -> None:
        self._ring = HashRing(vnodes=vnodes)
        super().__init__(replicas)

    def update_replicas(self, replicas: Iterable[str]) -> None:
        super().update_replicas(replicas)
        self._ring.set_nodes(self.replicas)

    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        return Decision(self._ring.home(request.conversation_id), "home")
