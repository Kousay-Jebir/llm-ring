from __future__ import annotations

from collections.abc import Iterable

from ..load import LoadTracker
from .base import Decision, Policy, RouteRequest


class RoundRobinPolicy(Policy):
    """Baseline: even load, no cache affinity at all."""

    name = "round-robin"

    def __init__(self, replicas: Iterable[str]) -> None:
        self._counter = 0
        super().__init__(replicas)

    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        replica = self.replicas[self._counter % len(self.replicas)]
        self._counter += 1
        return Decision(replica, "round-robin")
