# llm-ring

Cache-aware routing for LLM replicas on Kubernetes, using consistent hashing
with bounded loads, adapted to LLM serving.

> Status: work in progress. Current stage: router deployed and manually tested.
> Next: traffic generator, experiment runner, analysis.

## The idea in one paragraph

In a chat, every message resends the whole conversation. LLM servers cache the
processed conversation, but only on the replica that handled it. Sending the next
message to a different replica means reprocessing everything. `llm-ring` routes
each conversation to the replica that holds its cache, keeps that true when the
cluster scales, and avoids overloading any single replica. It applies a known
distributed-caching technique (consistent hashing with bounded loads) and adapts
it to the fact that, for LLMs, a cache miss is expensive and grows with the
conversation's length.

First measurement on this setup: turn 2 of a conversation sent back to the same
replica reused 77 of its 94 prompt tokens from cache. See `docs/notes.md`.

## What exists so far

| Component | Status |
|---|---|
| kind cluster (1 control plane + 3 workers) | done |
| llama.cpp replicas (StatefulSet, Qwen2.5-0.5B on CPU) | done |
| Router with 5 routing policies and Kubernetes discovery | done, manually tested |
| Traffic generator, experiment runner, analysis | coming |
| Automated tests | coming |

The router's policies:

| Policy | Behavior |
|---|---|
| `round-robin` | Rotates through replicas. Baseline with no cache affinity. |
| `mod-n` | `hash(conversation) % N`. Sticky until N changes, then most conversations move. |
| `consistent` | Hash ring with virtual nodes. Scaling moves only ~1/N of conversations. |
| `bounded` | Consistent hashing with bounded loads: no replica above (1 + epsilon) x average. |
| `llm-aware` | Bounded loads plus our LLM-specific additions (see `docs/design.md`). |

## Repository layout

```
cluster/        kind cluster definition (1 control plane + 3 workers)
k8s/            Kubernetes manifests, numbered in apply order
router/         routing service: core.py (logic), app.py (HTTP), policies/, ring.py
docs/           notes.md (lab notebook), design.md (how and why)
traffic/        conversation traffic generator (coming)
experiments/    experiment runner and analysis (coming)
tests/          automated tests (coming)
```

## Prerequisites

Docker, kind, kubectl, Python 3.11+. Around 8 GB RAM (16 GB recommended),
4+ CPU cores, 5 GB free disk.

Docker must use **cgroup v2**: recent Kubernetes versions refuse to start on
cgroup v1. Check with `docker info` (look for `Cgroup Version: 2`). On Windows,
see the fix in `docs/notes.md`.

Commands below work in PowerShell and bash, except where marked.

## Getting started

### 1. Create the cluster

```bash
kind create cluster --config cluster/kind-config.yaml
kubectl apply -f k8s/00-namespace.yaml
kubectl config set-context --current --namespace=llm-ring
```

`set-context` only changes your local default namespace; the `apply` is what
creates the namespace in the cluster.

### 2. Deploy the LLM replicas

```bash
kubectl apply -f k8s/10-llm-config.yaml -f k8s/20-llm.yaml -f k8s/30-llm-pdb.yaml
kubectl rollout status statefulset/llm --timeout=10m
kubectl get pods -o wide
```

Each replica downloads the model once (init container) into its own persistent
volume, loads it, then becomes ready. Replicas are spread across the three worker
nodes. After changing `k8s/10-llm-config.yaml`, restart the replicas so they read
the new values: `kubectl rollout restart statefulset/llm`.

### 3. Build and load the router image

Run from the repository root (the Dockerfile copies `router/` relative to it):

```bash
docker build -f router/Dockerfile -t llm-ring-router:0.1.0 .
kind load docker-image llm-ring-router:0.1.0 --name llm-ring
```

kind nodes can't see images on your machine, so `kind load` copies the image into
every node. After changing router code, build with a new tag (e.g. `0.1.1`), load
it, and update the tag in `k8s/42-router.yaml`.

### 4. Deploy the router

```bash
kubectl apply -f k8s/40-router-rbac.yaml -f k8s/41-router-config.yaml -f k8s/42-router.yaml
kubectl rollout status deployment/router
kubectl logs deploy/router
```

The logs should contain a `replicas_changed` event listing `llm-0`, `llm-1` and
`llm-2`: the router found the replicas through the Kubernetes API.

### 5. Talk to the router

Keep a port-forward open in a separate terminal:

```bash
kubectl port-forward svc/router 8000:80
```

Port-forward targets and their ports differ:

| Target | Port inside | Health endpoint |
|---|---|---|
| `svc/router` | 80 | `/healthz`, `/readyz` |
| `pod/llm-N` | 8080 | `/health` |

Send a message (PowerShell):

```powershell
$body = '{"messages":[{"role":"user","content":"What is a pod?"}],"max_tokens":40}'
$r = Invoke-WebRequest -Uri http://localhost:8000/v1/chat/completions -Method Post `
     -ContentType "application/json" -Headers @{"X-Conversation-ID"="demo-1"} -Body $body
$r.Headers["x-routed-to"]; $r.Headers["x-route-reason"]
($r.Content | ConvertFrom-Json).timings
```

The `X-Conversation-ID` header tells the router which conversation a message
belongs to. The response headers say which replica served it and why; `timings`
shows how many prompt tokens came from cache (`cache_n`) and how many were
processed (`prompt_n`).

Inspect the router's state (current policy, replicas, in-flight load):

```powershell
Invoke-RestMethod http://localhost:8000/admin/state | ConvertTo-Json -Depth 5
```

Switch policy at runtime (resets the policy's in-memory state):

```powershell
Invoke-RestMethod -Method Put -Uri http://localhost:8000/admin/policy `
  -ContentType "application/json" -Body '{"policy":"bounded","params":{"epsilon":0.5}}'
```

### 6. Scale and watch the router follow

```bash
kubectl scale statefulset llm --replicas=4
kubectl logs deploy/router -f
```

When `llm-3` becomes ready, the router logs `"added": ["llm-3"]`. With
`consistent`, only conversations that move to `llm-3` change replica. Scale back
with `kubectl scale statefulset llm --replicas=3`.

### Restarting after stopping Docker containers

kind nodes are Docker containers, so a stopped cluster can be restarted:

```bash
docker start llm-ring-control-plane llm-ring-worker llm-ring-worker2 llm-ring-worker3
kubectl get nodes
```

If nodes stay `NotReady`, delete and recreate the cluster (models will download
again).

### Tear down

```bash
kind delete cluster --name llm-ring
```

## Kubernetes design notes

LLM replicas:

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
- **PVC retention on scale-down**: a replica removed by scaling keeps its volume,
  so when it returns it finds the model on disk.

Router:

- **Least-privilege RBAC**: a dedicated ServiceAccount that may only read
  EndpointSlices, only in the `llm-ring` namespace.
- **Discovery through EndpointSlices**: only ready, non-terminating replicas are
  routed to, so a new replica joins after its model is loaded and a leaving one
  stops receiving traffic before it shuts down.
- **Single instance, `Recreate` strategy**: load counters and placement memory
  live in the process; two routers at once would make inconsistent decisions.
- **Readiness vs liveness**: `/readyz` fails until a replica is known;
  `/healthz` only checks the process, so losing all replicas never triggers
  pointless router restarts.
- **Hardened container**: non-root user, read-only root filesystem, no Linux
  capabilities, no privilege escalation, default seccomp profile.
- **Separate runtime requirements**: the image installs only
  `router/requirements.txt` (FastAPI, uvicorn, httpx); `.dockerignore` keeps
  everything else out of the build.