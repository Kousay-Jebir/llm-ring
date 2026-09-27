"""Policy registry."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from typing import Any

from .base import Decision, Policy, RouteRequest
from .bounded import BoundedLoadsPolicy
from .consistent import ConsistentHashPolicy
from .llm_aware import LLMAwarePolicy
from .mod_n import ModNPolicy
from .round_robin import RoundRobinPolicy

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls
    for cls in (RoundRobinPolicy, ModNPolicy, ConsistentHashPolicy, BoundedLoadsPolicy, LLMAwarePolicy)
}


def accepted_params(name: str) -> set[str]:
    cls = POLICIES[name]
    return {p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "replicas")}


def build_policy(
    name: str,
    replicas: Iterable[str],
    defaults: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Policy:
    """Create a policy.

    ``defaults`` (typically the router settings) are filtered to what the policy
    accepts. ``overrides`` (typically from the admin API) must all be valid for the
    chosen policy, so typos are reported instead of silently ignored.
    """
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    accepted = accepted_params(name)
    params = {k: v for k, v in (defaults or {}).items() if k in accepted}
    unknown = set(overrides or {}) - accepted
    if unknown:
        raise ValueError(f"policy {name!r} does not accept {sorted(unknown)}; accepted: {sorted(accepted)}")
    params.update(overrides or {})
    return POLICIES[name](replicas, **params)


__all__ = ["POLICIES", "Decision", "Policy", "RouteRequest", "accepted_params", "build_policy"]
