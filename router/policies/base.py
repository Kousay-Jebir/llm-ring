"""Common interface for routing policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

from ..load import LoadTracker


@dataclass(frozen=True)
class RouteRequest:
    """What a policy knows about a request when choosing a replica."""

    conversation_id: str
    prompt_tokens: int  # estimated size of the whole conversation sent in this request
    max_tokens: int  # generation budget requested by the client

    @property
    def cost(self) -> int:
        """Estimated work in tokens: prompt to process plus tokens to generate."""
        return self.prompt_tokens + self.max_tokens


@dataclass(frozen=True)
class Decision:
    replica: str
    reason: str  # why this replica: "home", "overflow", "sticky", ...
    # Tokens to charge to the replica's load while the request runs. None means
    # ``RouteRequest.cost``. Policies that know the cost depends on the replica
    # (because of caching) set it explicitly.
    cost: int | None = None


class Policy(ABC):
    """A routing policy. Implementations must be cheap and must not block."""

    name: str = "abstract"

    def __init__(self, replicas: Iterable[str]) -> None:
        self._replicas: tuple[str, ...] = ()
        self.update_replicas(replicas)

    @property
    def replicas(self) -> tuple[str, ...]:
        return self._replicas

    def update_replicas(self, replicas: Iterable[str]) -> None:
        # Sorted so that order-dependent policies (round-robin, mod-N) behave the
        # same regardless of the order in which discovery reports replicas.
        self._replicas = tuple(sorted(set(replicas)))

    @abstractmethod
    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        """Pick a replica. Only called when at least one replica exists."""

    def observe(self, request: RouteRequest, replica: str, success: bool) -> None:
        """Called after the replica answered. Stateless policies ignore it."""
