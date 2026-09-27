from __future__ import annotations

from ..hashing import stable_hash
from ..load import LoadTracker
from .base import Decision, Policy, RouteRequest


class ModNPolicy(Policy):
    """hash(conversation) % N: perfect affinity until N changes, then most keys move."""

    name = "mod-n"

    def choose(self, request: RouteRequest, load: LoadTracker) -> Decision:
        index = stable_hash(request.conversation_id) % len(self.replicas)
        return Decision(self.replicas[index], "home")
