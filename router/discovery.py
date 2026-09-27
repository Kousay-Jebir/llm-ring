"""Replica discovery: which LLM replicas exist and are ready right now."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any
from urllib.parse import quote

FetchJson = Callable[[str], Awaitable[dict[str, Any]]]


def _url(address: str, port: int) -> str:
    host = f"[{address}]" if ":" in address else address  # IPv6
    return f"http://{host}:{port}"


def parse_endpoint_slices(document: Mapping[str, Any], port_name: str = "http") -> dict[str, str]:
    """Turn an EndpointSliceList into ``{pod name: base URL}`` for ready endpoints only.

    Kubernetes semantics: ``ready`` unset means ready; terminating endpoints are
    excluded even if still serving, so traffic drains before a pod goes away.
    """
    replicas: dict[str, str] = {}
    for slice_ in document.get("items") or []:
        ports = slice_.get("ports") or []
        port = next((p.get("port") for p in ports if p.get("name") == port_name), None)
        if port is None and len(ports) == 1:
            port = ports[0].get("port")
        if port is None:
            continue
        for endpoint in slice_.get("endpoints") or []:
            conditions = endpoint.get("conditions") or {}
            if conditions.get("ready") is False or conditions.get("terminating") is True:
                continue
            name = (endpoint.get("targetRef") or {}).get("name") or endpoint.get("hostname")
            addresses = endpoint.get("addresses") or []
            if name and addresses:
                replicas[name] = _url(addresses[0], int(port))
    return replicas


class StaticDiscovery:
    """A fixed list of replicas (useful locally, but blind to scaling)."""

    def __init__(self, names: Iterable[str], url_template: str) -> None:
        self._replicas = {name: url_template.format(name=name) for name in names}

    async def fetch(self) -> dict[str, str]:
        return dict(self._replicas)


class KubernetesDiscovery:
    """Reads the EndpointSlices of the replicas' headless Service from the API server."""

    def __init__(self, fetch_json: FetchJson, namespace: str, service: str, port_name: str) -> None:
        self._fetch_json = fetch_json
        self._port_name = port_name
        selector = quote(f"kubernetes.io/service-name={service}", safe="")
        self._path = f"/apis/discovery.k8s.io/v1/namespaces/{namespace}/endpointslices?labelSelector={selector}"

    @property
    def path(self) -> str:
        return self._path

    async def fetch(self) -> dict[str, str]:
        return parse_endpoint_slices(await self._fetch_json(self._path), self._port_name)
