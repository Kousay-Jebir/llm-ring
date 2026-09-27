"""Stable hashing.

Python's built-in ``hash()`` is salted per process, so two router instances (or one
router after a restart) would disagree on where a conversation belongs. Every hash
used for routing must go through ``stable_hash`` instead.
"""

from __future__ import annotations

import hashlib


def stable_hash(value: str) -> int:
    """Return a 64-bit hash of ``value`` that is identical across processes and machines."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")
