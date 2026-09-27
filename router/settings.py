"""Router configuration, read from ROUTER_* environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def _bool(value: str, name: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class Settings:
    # Routing
    policy: str = "consistent"
    vnodes: int = 100
    epsilon: float = 0.25
    load_metric: str = "requests"
    long_epsilon: float = 1.0
    long_threshold_tokens: int = 800
    token_load: bool = True
    long_stickiness: bool = True
    placement_memory: bool = True
    memory_max_entries: int = 50_000
    default_max_tokens: int = 256

    # Replica discovery
    discovery: str = "kubernetes"  # "kubernetes" or "static"
    static_replicas: tuple[str, ...] = field(default=("llm-0", "llm-1", "llm-2"))
    replica_url_template: str = "http://{name}.llm.llm-ring.svc.cluster.local:8080"
    k8s_namespace: str = "llm-ring"
    k8s_service: str = "llm"
    k8s_port_name: str = "http"
    discovery_interval_s: float = 2.0

    # Upstream and logging
    upstream_timeout_s: float = 300.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        defaults = cls()
        values: dict[str, Any] = {}
        for name, default in defaults.__dict__.items():
            key = f"ROUTER_{name.upper()}"
            if key not in env:
                continue
            raw = env[key]
            if isinstance(default, bool):
                values[name] = _bool(raw, key)
            elif isinstance(default, int):
                values[name] = int(raw)
            elif isinstance(default, float):
                values[name] = float(raw)
            elif isinstance(default, tuple):
                values[name] = tuple(p.strip() for p in raw.split(",") if p.strip())
            else:
                values[name] = raw.strip()
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.discovery not in ("kubernetes", "static"):
            raise ValueError("ROUTER_DISCOVERY must be 'kubernetes' or 'static'")
        if self.discovery == "static" and not self.static_replicas:
            raise ValueError("ROUTER_STATIC_REPLICAS must list at least one replica")
        if "{name}" not in self.replica_url_template:
            raise ValueError("ROUTER_REPLICA_URL_TEMPLATE must contain {name}")
        if self.discovery_interval_s <= 0 or self.upstream_timeout_s <= 0:
            raise ValueError("intervals and timeouts must be > 0")
        if self.default_max_tokens < 1:
            raise ValueError("ROUTER_DEFAULT_MAX_TOKENS must be >= 1")

    def policy_params(self) -> dict[str, Any]:
        """Settings that policies may accept; each policy keeps only its own."""
        return {
            "vnodes": self.vnodes,
            "epsilon": self.epsilon,
            "load_metric": self.load_metric,
            "long_epsilon": self.long_epsilon,
            "long_threshold_tokens": self.long_threshold_tokens,
            "token_load": self.token_load,
            "long_stickiness": self.long_stickiness,
            "placement_memory": self.placement_memory,
            "memory_max_entries": self.memory_max_entries,
        }
