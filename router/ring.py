"""Stable hashing and the consistent hash ring."""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections.abc import Iterable


def stable_hash(value: str) -> int:
    """64-bit hash, identical in every process. Python's hash() is salted per process,
    so a restarted router would reshuffle every conversation."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


class HashRing:
    """Each node sits at ``vnodes`` points on the ring; a key belongs to the first
    node clockwise from its hash. Adding a node moves only ~1/N of the keys."""

    def __init__(self, vnodes: int) -> None:
        if vnodes < 1:
            raise ValueError("vnodes must be >= 1")
        self._vnodes = vnodes
        self._nodes: set[str] = set()
        self._points: list[int] = []
        self._owners: list[str] = []

    def set_nodes(self, nodes: Iterable[str]) -> None:
        nodes = set(nodes)
        if nodes == self._nodes:
            return
        self._nodes = nodes
        # Sorting by (hash, node) keeps the ring deterministic even on a hash collision.
        pairs = sorted((stable_hash(f"{n}#{i}"), n) for n in nodes for i in range(self._vnodes))
        self._points = [p for p, _ in pairs]
        self._owners = [n for _, n in pairs]

    def candidates(self, key: str) -> list[str]:
        """Every node, in the order met walking clockwise from ``key``. The first is its home."""
        if not self._points:
            return []
        start = bisect_right(self._points, stable_hash(key))
        seen: list[str] = []
        for offset in range(len(self._points)):
            owner = self._owners[(start + offset) % len(self._points)]
            if owner not in seen:
                seen.append(owner)
                if len(seen) == len(self._nodes):
                    break
        return seen
