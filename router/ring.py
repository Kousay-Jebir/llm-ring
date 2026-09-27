
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable

from .hashing import stable_hash


class HashRing:
    """A consistent hash ring.

    Each node is placed at ``vnodes`` points on the ring. A key belongs to the first
    node found clockwise from the key's hash. Adding or removing a node only moves
    the keys on the arcs that node gains or loses (about 1/N of all keys).
    """

    def __init__(self, nodes: Iterable[str] = (), vnodes: int = 100) -> None:
        if vnodes < 1:
            raise ValueError("vnodes must be >= 1")
        self._vnodes = vnodes
        self._nodes: set[str] = set(nodes)
        self._points: list[int] = []
        self._owners: list[str] = []
        self._rebuild()

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._nodes))

    def __len__(self) -> int:
        return len(self._nodes)

    def set_nodes(self, nodes: Iterable[str]) -> None:
        new_nodes = set(nodes)
        if new_nodes != self._nodes:
            self._nodes = new_nodes
            self._rebuild()

    def add(self, node: str) -> None:
        if node not in self._nodes:
            self._nodes.add(node)
            self._rebuild()

    def remove(self, node: str) -> None:
        if node in self._nodes:
            self._nodes.remove(node)
            self._rebuild()

    def _rebuild(self) -> None:
        # Sorting by (hash, node) makes the ring fully deterministic, even in the
        # (astronomically unlikely) case of a 64-bit hash collision.
        pairs = sorted(
            (stable_hash(f"{node}#{i}"), node)
            for node in self._nodes
            for i in range(self._vnodes)
        )
        self._points = [point for point, _ in pairs]
        self._owners = [owner for _, owner in pairs]

    def home(self, key: str) -> str:
        """Return the node that owns ``key``."""
        if not self._points:
            raise LookupError("ring is empty")
        index = bisect_right(self._points, stable_hash(key)) % len(self._points)
        return self._owners[index]

    def candidates(self, key: str) -> list[str]:
        """Return every node in the order met when walking clockwise from ``key``.

        The first element is ``home(key)``. Bounded-load routing walks this list
        until it finds a node with spare capacity.
        """
        if not self._points:
            return []
        total = len(self._points)
        start = bisect_right(self._points, stable_hash(key))
        seen: list[str] = []
        seen_set: set[str] = set()
        for offset in range(total):
            owner = self._owners[(start + offset) % total]
            if owner not in seen_set:
                seen.append(owner)
                seen_set.add(owner)
                if len(seen) == len(self._nodes):
                    break
        return seen
