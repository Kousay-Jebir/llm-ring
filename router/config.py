"""Router settings, read from ROUTER_* environment variables.

They come from the router-config ConfigMap (k8s/41-router-config.yaml). There are no
defaults in the code: a missing value stops the router at startup with its name, so
every value is set in exactly one place.
"""

import os
from dataclasses import dataclass, fields


def _bool(raw: str) -> bool:
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected a boolean, got {raw!r}")


@dataclass(frozen=True)
class Config:
    # Routing
    policy: str
    vnodes: int
    epsilon: float
    load_metric: str
    long_epsilon: float
    long_threshold_tokens: int
    token_load: bool
    long_stickiness: bool
    placement_memory: bool
    memory_max_entries: int
    default_max_tokens: int
    # Replica discovery
    k8s_namespace: str
    k8s_service: str
    k8s_port_name: str
    discovery_interval_s: float
    k8s_api_timeout_s: float
    # Connections to the replicas
    upstream_timeout_s: float
    upstream_connect_timeout_s: float
    upstream_max_connections: int
    upstream_max_keepalive: int
    log_level: str

    @classmethod
    def from_env(cls, env=os.environ) -> "Config":
        parsers = {bool: _bool, int: int, float: float, str: str}
        values, missing = {}, []
        for field in fields(cls):
            key = f"ROUTER_{field.name.upper()}"
            if key not in env:
                missing.append(key)
                continue
            try:
                values[field.name] = parsers[field.type](env[key].strip())
            except ValueError as exc:
                raise ValueError(f"{key}: {exc}") from None
        if missing:
            raise ValueError(f"missing router settings: {', '.join(missing)}")
        config = cls(**values)
        for name in ("discovery_interval_s", "k8s_api_timeout_s", "upstream_timeout_s",
                     "upstream_connect_timeout_s", "default_max_tokens", "upstream_max_connections"):
            if getattr(config, name) <= 0:
                raise ValueError(f"ROUTER_{name.upper()} must be > 0")
        return config
