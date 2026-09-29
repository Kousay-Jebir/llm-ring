# llm-ring

Cache-aware routing for LLM replicas on Kubernetes: consistent hashing with bounded
loads, adapted to the way LLM servers cache conversations.

## The problem

In a chat, every message resends the whole conversation. LLM servers keep a
**prompt cache** (the KV cache): after answering, the processed conversation stays
in memory, so the next turn only needs its new tokens processed. But that cache
lives on **one replica**. Send the next turn to a different replica and the whole
history is processed again.

Measured on this project's setup (Qwen2.5-0.5B, CPU):

- A 28-token prompt: **376 ms** cold, **32 ms** from cache.
- A second turn routed back to the same replica reused **77 of its 94** prompt tokens.
- Under round-robin routing, **91% of follow-up cache misses were avoidable**: the
  history existed in a cache, just on another replica.

So with several replicas, *where* each request goes decides how much work is wasted.

## The idea

This is the problem distributed caches (CDNs, Memcached clusters) solved long ago:
send each key to the server that holds it. `llm-ring` applies the known solution,
**consistent hashing with bounded loads** (Mirrokni, Thorup, Zadimoghaddam, Google,
2016), and adapts it to what makes LLM serving different: requests vary enormously
in size, and a cache miss is expensive and grows with the conversation's length.

| Policy | Idea | Weakness |
|---|---|---|
| `round-robin` | Rotate through replicas | Consecutive turns land on different replicas |
| `mod-n` | `hash(conversation) % N` | When N changes, ~75% of conversations move (3 → 4 replicas) |
| `consistent` | Hash ring with virtual nodes | Scaling moves only ~1/N, but blind to load |
| `bounded` | Ring + cap of `(1 + ε) × average load`, load counted in requests (default) or tokens | Ignores where the cache is: counts every request as 1, or (tokens mode) charges the full history everywhere; forgets redirects |
| **`llm-aware`** | `bounded` + three LLM-specific additions | This project's contribution |

`llm-aware` adds, each switchable for ablations:

- **Cache-aware token load.** Load is counted in tokens of actual work. On the
  replica holding a conversation's cache, a request costs only its new tokens;
  anywhere else, its whole history.
- **Placement memory.** The router remembers where each conversation was last served
  and keeps it there after a redirect, instead of sending it back to its ring home
  and paying a second full reprocess.
- **Long-conversation stickiness.** Long, cached conversations tolerate up to
  `2 × average` load before being redirected, since moving them costs the most.

## Results

Standard profile (50 concurrent conversations, 30% pasting 600-word documents,
5 to 10 turns each), 3 replicas, same traffic for both policies (seed 1):

| | `bounded` | `llm-aware` | |
|---|---|---|---|
| Cache hit rate (follow-up turns) | 88.5% | **94.2%** | |
| Cache miss rate | 11.5% | **5.8%** | misses halved |
| Redirects away from the cache | 64 | **17** | 4× fewer |
| p50 latency | 111.6 s | **91.4 s** | −18% |
| p95 latency | 217.0 s | **144.9 s** | −33% |
| Work imbalance (busiest / average) | **1.008** | 1.025 | both near even |

Why, from the routing decisions (`llm-aware`, 732 requests):

- **17 redirects** instead of 64: a busy replica still accepts its own cached
  conversations, because they cost only their new tokens there.
- **154 `sticky` decisions**: after a redirect, the conversation stayed where its
  cache now lives. `bounded` would have sent each of these back home.
- **85 `-long` decisions**: long conversations kept with their cache despite their
  replica being above the normal cap.

The latency gain is larger than the hit-rate gain because the replicas were
saturated: when requests queue, halving the reprocessing work shrinks the queues
disproportionately.

**Caveats, stated plainly:**

- One run per policy so far. In both comparisons made, `llm-aware` ran first;
  confirmation runs with more seeds and alternated order are in progress.
- These runs did not include a scale-up; that comparison is separate.
- Everything runs on one laptop. Hit rates and redirect counts are solid; absolute
  latencies are pessimistic and noisier.
- Under **light** traffic, all load-aware policies route almost everything home and
  give nearly identical results. The policies only differ when replicas are
  regularly over their cap.

## How it works

```
traffic generator ──(x-conversation-id, x-turn)──▶ router ──▶ llm-0 │ llm-1 │ llm-2 (│ llm-3)
        ▲                                             │           llama.cpp replicas
        └──(answer + timings; x-routed-to, x-route-reason)───┘
```

- **Replicas**: a StatefulSet of llama.cpp servers, each with its own volume holding
  the model, spread across worker nodes.
- **Router**: a FastAPI service. For each request it identifies the conversation,
  asks the policy for a replica, tracks in-flight load, forwards the request, and
  logs the decision. It discovers ready replicas from the Kubernetes API
  (EndpointSlices), so a new replica joins the ring as soon as its model is loaded.
- **Traffic generator**: simulated users holding multi-turn chats; every turn resends
  the whole conversation, like a real chat client. Deterministic from a seed.
- **Experiment runner**: replays the same traffic under each policy, optionally adds
  a replica mid-run, and records every request.

### Decision traces

Every routing decision is logged with the numbers and steps that produced it. The
same situation, seen by both policies:

```
llm-aware → llm-0 (home-long)
  placement memory: last served on llm-0 with 820 prompt tokens → preferred=llm-0
  cost on llm-0: max(1, 860-820)=40 + 48 = 88 (new tokens only, cache hit)
  load (tokens): llm-0=2400 llm-1=900 llm-2=900; +88 → total=4288, avg=1429.3, cap=1786.7
  long stickiness: prompt_tokens=860 ≥ 800 and cache_likely=True → long_cap=2858.7
  llm-0: 2400+88=2488 ≤ long_cap=2858.7 (> normal cap=1786.7) → home-long

bounded (token mode) → llm-2 (overflow)
  load (tokens): llm-0=2400 llm-1=900 llm-2=900; with this request total=5108, cap=2128.3
  llm-0: 2400+908=3308 > 2128.3 → skip
  llm-2: 900+908=1808 ≤ 2128.3 → overflow
```

## Repository layout

```
cluster/kind-config.yaml   kind cluster: 1 control plane + 3 workers
k8s/                       manifests, numbered in apply order
  00-namespace.yaml
  10-llm-config.yaml       all replica settings
  20-llm.yaml              llama.cpp StatefulSet + headless Service
  30-llm-pdb.yaml          PodDisruptionBudget
  40-router-rbac.yaml      ServiceAccount allowed to read EndpointSlices only
  41-router-config.yaml    all router settings
  42-router.yaml           router Deployment + Service
router/
  config.py                reads the router's settings (no defaults in code)
  ring.py                  stable hashing and the hash ring
  policies.py              the five policies, their decision traces, the registry
  core.py                  request handling, load tracking, logging
  app.py                   HTTP endpoints and replica discovery
traffic/
  generator.py             conversation traffic generator
  profiles/*.json          traffic shapes (quick, standard)
experiments/
  run.py                   runs the policy comparison
  analyze.py               summary table and charts
settings.toml              settings of the tools you run locally
docs/
  design.md                design, data flow, metrics, limitations
  notes.md                 lab notebook: measurements and problems met
```

## Configuration

Every setting lives in exactly one place:

| What | Where | Applied by |
|---|---|---|
| Replicas (context, slots, threads, RAM cache, model) | `k8s/10-llm-config.yaml` | `kubectl apply`, then `kubectl rollout restart statefulset/llm` |
| Router (policy, ε, timeouts, discovery) | `k8s/41-router-config.yaml` | `kubectl apply`, then `kubectl rollout restart deployment/router` |
| Generator, runner, analysis | `settings.toml` | read at every run; command-line options override it |
| Traffic shapes | `traffic/profiles/*.json` | chosen with `--profile` |

The router has no defaults in its code: a missing key stops it at startup, with the
key's name in `kubectl logs deploy/router`.

One manual link to keep: a profile's `max_context_tokens` must stay below the
replicas' per-slot context (`LLAMA_ARG_CTX_SIZE / LLAMA_ARG_N_PARALLEL`, 4096 tokens
here), with a margin, since the generator only estimates token counts.

## Prerequisites

Docker, kind, kubectl, Python 3.11+. About 8 GB RAM (16 GB recommended), 4+ CPU
cores, 5 GB free disk. Docker must use **cgroup v2** (`docker info` shows
`Cgroup Version: 2`). On Windows, use Docker Desktop with the WSL2 engine; see
`docs/notes.md` if the cluster fails to start.

Commands below work in PowerShell.

## Setup

### 1. Python environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Cluster and replicas

```bash
kind create cluster --config cluster/kind-config.yaml
kubectl apply -f k8s
kubectl config set-context --current --namespace=llm-ring
kubectl apply -f k8s/10-llm-config.yaml -f k8s/20-llm.yaml -f k8s/30-llm-pdb.yaml
kubectl rollout status statefulset/llm --timeout=15m
```

Each replica downloads the model once into its own volume, loads it, then becomes
ready.

### 3. Router

```bash
docker build -f router/Dockerfile -t llm-ring-router:0.1.3 .
kind load docker-image llm-ring-router:0.1.3 --name llm-ring
kubectl apply -f k8s/40-router-rbac.yaml -f k8s/41-router-config.yaml -f k8s/42-router.yaml
kubectl rollout status deployment/router
kubectl logs deploy/router
```

The logs should show a `replicas_changed` event listing `llm-0`, `llm-1`, `llm-2`.
After changing router code, build with a new tag, load it, and update the tag in
`k8s/42-router.yaml`.

## Running experiments

### Smoke test

```bash
python -m experiments.run --policies consistent --profile traffic/profiles/quick.json --prepare
python -m experiments.analyze --bin-s 3
```

`--prepare` starts the extra replica once so it downloads the model before any
measurement; after that, every run's scale-up costs a model load, never a download.
It's needed once per cluster.

### Comparing policies

```bash
python -m experiments.run --policies bounded llm-aware --seed 1 --out-dir experiments/results/raw-main
python -m experiments.run --policies llm-aware bounded --seed 2 --out-dir experiments/results/raw-main
python -m experiments.analyze --results-dir experiments/results/raw-main --out-dir experiments/results/main
```

Use several seeds (different conversation placements) and alternate the policy
order (so neither always runs on a cold or hot laptop). `--scale-at 0` disables the
mid-run scale-up (default value is 0). The analysis combines all runs of the same policy in a folder;
keep separate experiments in separate folders.

Outputs: `summary.md`, `per_run.csv`, `summary.png` (hit rate, p95 latency,
imbalance per policy) and `hit_rate_over_time.png` (one line per policy, all runs
pooled). Check the `errors` column first.

## What gets measured

Each request produces one CSV row: which replica served it and why, prompt tokens
reused from cache (`cache_n`) and processed (`prompt_n`), timings, latency. Each run
produces one JSON with what was run and when the new replica became ready. The
analysis computes, per policy: hit rate, hit rate right after the scale-up, share of
conversations moved by the scale-up, p50/p95 latency, and work imbalance.
Definitions and data flow: `docs/design.md`, section 5.

## Limitations

- **Single machine.** All nodes share one CPU; adding a replica adds a cache and a
  routing target, not compute. Latency results are indicative.
- **Single router instance.** Load counters and placement memory live in one
  process; several routers would agree on ring homes but not on load or memory.
- **Estimated token counts** (~4 characters per token). The cached-prefix estimate
  also ignores the previous answer, which is cached too, so it slightly overstates
  the new work of a follow-up turn.
- **Load is released only when the full answer arrives**; no streaming support.
- **Answers are not bit-exact across runs** (batching and cache state change the
  floating-point path), so a few conversations can end a turn earlier or later under
  the context limit.
- **Generous server cache.** With 512 MiB of RAM prompt cache per replica, caches
  were never evicted for lack of space; with smaller caches, routing would matter
  more, not less.
