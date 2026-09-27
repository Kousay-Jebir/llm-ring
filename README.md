# llm-ring

Cache-aware routing for LLM replicas on Kubernetes, using consistent hashing
with bounded loads, adapted to LLM serving.

> Status: work in progress. Current stage: Kubernetes environment.

## The idea in one paragraph

In a chat, every message resends the whole conversation. LLM servers cache the
processed conversation, but only on the replica that handled it. Sending the next
message to a different replica means reprocessing everything. `llm-ring` routes
each conversation to the replica that holds its cache, keeps that true when the
cluster scales, and avoids overloading any single replica. It applies a known
distributed-caching technique (consistent hashing with bounded loads) and adapts
it to the fact that, for LLMs, a cache miss is expensive and grows with the
conversation's length.

## Repository layout

```
cluster/        kind cluster definition (1 control plane + 3 workers)
k8s/            Kubernetes manifests (namespace, config, replicas, PDB)
router/         the routing service (coming next)
traffic/        conversation traffic generator (coming)
experiments/    experiment runner and analysis (coming)
docs/           lab notes, design, results
tests/          unit tests for the hash ring (coming)
```

## Prerequisites

Docker, kind, kubectl, Python 3.11+, jq. Around 8 GB RAM (16 GB recommended),
4+ CPU cores, 5 GB free disk. On Windows, use WSL2.

## Getting started

### 1. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create the cluster

```bash
kind create cluster --config cluster/kind-config.yaml
kubectl apply -f k8s/00-namespace.yaml
kubectl config set-context --current --namespace=llm-ring
```

### 3. Deploy the LLM replicas

```bash
kubectl apply -f k8s/
kubectl rollout status statefulset/llm --timeout=10m
kubectl get pods -o wide
```

Each replica downloads the model once (init container) into its own persistent
volume, loads it, then becomes ready. Replicas are spread across the three
worker nodes.

### 4. Check a specific replica from inside the cluster

```bash
kubectl run debug --rm -it --image=curlimages/curl:8.10.1 --restart=Never -- \
  curl -s http://llm-1.llm.llm-ring.svc.cluster.local:8080/health
```

### 5. Scale (the operation the experiment will perform mid-traffic)

```bash
kubectl scale statefulset llm --replicas=4
kubectl get pods -o wide -w
kubectl scale statefulset llm --replicas=3
```

### Tear down

```bash
kind delete cluster --name llm-ring
```

## Kubernetes design notes

- **Headless Service**: each replica gets its own DNS name
  (`llm-N.llm.llm-ring.svc.cluster.local`), because the router, not
  Kubernetes, decides which replica serves a request.
- **StatefulSet + volumeClaimTemplates**: stable replica names (used as points
  on the hash ring) and a persistent volume per replica, so restarts skip the
  model download.
- **Init container with atomic rename**: the model is downloaded to a `.part`
  file and renamed when complete, so an interrupted download never looks like a
  valid model.
- **Startup / readiness / liveness probes**: the startup probe allows a long
  model-loading phase without liveness restarts; readiness gates traffic.
- **Topology spread constraints**: replicas land on different nodes.
- **PodDisruptionBudget**: node maintenance can't take down more than one
  replica at a time, since an evicted replica loses its cached conversations.
- **Memory limit, no CPU limit**: protects nodes without throttling generation,
  which would distort latency measurements.
