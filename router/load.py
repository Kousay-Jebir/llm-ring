"""In-flight load accounting per replica."""

from __future__ import annotations

from collections.abc import Iterable


class LoadTracker:
    """Counts in-flight requests and in-flight estimated tokens per replica.

    Replicas can disappear while requests to them are still running (for example
    during a scale-down). Their counters are kept until those requests finish, so
    ``release`` never fails, but they are no longer reported as current replicas.
    """

    def __init__(self) -> None:
        self._current: set[str] = set()
        self._requests: dict[str, int] = {}
        self._tokens: dict[str, int] = {}

    def sync(self, replicas: Iterable[str]) -> None:
        self._current = set(replicas)
        for replica in list(self._requests):
            if replica not in self._current and self._requests[replica] == 0:
                del self._requests[replica]
                self._tokens.pop(replica, None)

    def acquire(self, replica: str, tokens: int) -> None:
        self._requests[replica] = self._requests.get(replica, 0) + 1
        self._tokens[replica] = self._tokens.get(replica, 0) + tokens

    def release(self, replica: str, tokens: int) -> None:
        self._requests[replica] = max(0, self._requests.get(replica, 0) - 1)
        self._tokens[replica] = max(0, self._tokens.get(replica, 0) - tokens)
        if replica not in self._current and self._requests[replica] == 0:
            del self._requests[replica]
            self._tokens.pop(replica, None)

    def requests(self, replica: str) -> int:
        return self._requests.get(replica, 0)

    def tokens(self, replica: str) -> int:
        return self._tokens.get(replica, 0)

    def snapshot(self) -> dict[str, dict[str, int]]:
        return {
            replica: {"requests": self.requests(replica), "tokens": self.tokens(replica)}
            for replica in sorted(self._current)
        }
