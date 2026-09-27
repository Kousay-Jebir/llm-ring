# Design

This document covers the parts built so far: the replicas' setup and the router.
Measurement method and results will be added with the experiment tooling.

## 1. Problem

LLM servers keep a prompt cache: after answering, the processed conversation stays
in memory, so the next message of the same conversation only needs its new tokens
processed. The cache is local to one replica. On this setup, a 28-token prompt took
376 ms cold and 32 ms cached, and through the router a second turn reused 77 of its
94 prompt tokens.

So routing decides cost: send a conversation back to the replica that holds its
cache and the reply starts fast; send it elsewhere and its whole history is
reprocessed.

Sharing the cache between replicas is possible in principle but expensive: the
cache is large (hundreds of MB for a long conversation on bigger models) and must
be in the serving replica's memory to be used. Moving it pays off only with very
fast interconnects. This project takes the other approach: move the request to
where the cache already is.

## 2. The known solution, step by step

| Policy | Idea | Weakness |
|---|---|---|
| `round-robin` | Rotate through replicas | Consecutive turns land on different replicas |
| `mod-n` | `hash(conversation) % N` | When N changes, ~(N-1)/N of conversations move (75% for 3 -> 4) |
| `consistent` | Hash ring with virtual nodes | Scaling moves only ~1/N, but no load awareness: hot spots |
| `bounded` | Consistent hashing with bounded loads | Needs a definition of "load"; built for equal-cost requests |

**Stable hashing.** Python's built-in `hash()` is salted per process, so a
restarted router would reshuffle every conversation. The router uses BLAKE2b
(`router/hashing.py`).

**Virtual nodes.** Each replica is placed at 100 points on the ring, so three
replicas split it evenly instead of by chance.

**Bounded loads** (Mirrokni, Thorup and Zadimoghaddam, 2016). Every replica has a
capacity of `(1 + epsilon) x average load`, computed including the incoming
request. A conversation goes to its home on the ring unless the home is at
capacity; then the router walks clockwise to the next replica with room. An idle
replica always accepts, so a single request larger than the capacity can still be
routed.

## 3. Our additions (`llm-aware`)

The original algorithm assumes requests and cache misses cost about the same. For
LLMs neither holds. Each addition can be switched off through the admin API, for
ablation experiments (`token_load`, `long_stickiness`, `placement_memory`).

### 3.1 Cache-aware token load

Load is measured in estimated tokens of work rather than request counts, since one
request can be ten times heavier than another.

The naive version, charging a conversation's whole history as work, backfires:
on the conversation's own replica most of that history is cached, so the real work
is small. Charging the full history makes long conversations look expensive at
home, pushes them over capacity, and redirects them, destroying the cache that
made them cheap. So **cost depends on where the request goes**: on the replica
holding the cache, only the new tokens; anywhere else, the full history. The policy
returns the cost it used with its decision (`Decision.cost`), so load accounting
matches the reasoning exactly.

### 3.2 Long-conversation stickiness

A redirect costs a full reprocessing of the history, which grows with the
conversation. A conversation above `long_threshold_tokens` is only redirected when
its replica exceeds `(1 + long_epsilon) x average`, instead of
`(1 + epsilon) x average`. It applies only when a cache probably exists (the
conversation was served before); a brand-new long conversation has nothing to
protect.

### 3.3 Placement memory

In plain bounded loads, a conversation redirected during an overload returns to
its ring home once the overload ends, which costs another full reprocessing. The
router instead remembers where each conversation was last served, and the size of
its prompt there (bounded LRU, `memory_max_entries`), and keeps it where its cache
lives. That memory also feeds the cost estimate in 3.1. Entries for replicas that
disappear are dropped.

Expected side effect after a scale-up: existing conversations stay on the replicas
holding their caches, and the new replica receives mostly new conversations and
overflow, instead of a burst of cache misses all at once.

## 4. How the router works

**Request path** (`router/core.py`):

1. Parse the body; reject invalid requests (400) and streaming requests.
2. Identify the conversation: the `X-Conversation-ID` header, or a hash of the
   first system and user messages when the header is missing.
3. Estimate the conversation's size in tokens (~4 characters per token).
4. The policy chooses a replica using the current load.
5. Record the request's load on that replica (**acquire**).
6. Forward the request to the replica.
7. Remove the load (**release**, in a `finally` block, so failures never leave
   load counted forever).
8. Let the policy observe the outcome (placement memory), log a JSON record, and
   return the response with `x-routed-to`, `x-route-reason`, `x-route-policy`
   and `x-conversation-id` headers.

Steps 4 and 5 run with no `await` in between. The router is a single asyncio event
loop that only switches between requests at `await` points, so every routing
decision sees the load of all requests routed before it.

**Load tracker** (`router/load.py`). Counts in-flight requests and in-flight
estimated tokens per replica. `bounded` reads request counts, `llm-aware` reads
tokens. A replica removed while it still has requests in flight keeps its counters
until they are released, then is cleaned up.

**Discovery** (`router/discovery.py`, `router/app.py`). Every 2 seconds the router
reads the EndpointSlices of the `llm` headless Service through the Kubernetes API,
authenticating with its ServiceAccount token, which is re-read on each call because
Kubernetes rotates it. Only ready, non-terminating endpoints count. On errors, the
last known replicas are kept.

**Admin API.** `GET /admin/state` shows policy, replicas, in-flight load and
counters. `PUT /admin/policy` swaps the policy; unknown parameters are rejected
rather than silently ignored.

## 5. Current limitations

- **One laptop.** All kind nodes share the same CPU: scaling to 4 replicas adds a
  cache and a routing target, not real compute. Absolute latencies are pessimistic.
- **Approximate token estimates** (~4 characters per token). Routing only needs
  relative sizes.
- **Single router instance.** Load counters and placement memory live in one
  process. Several routers would agree on ring homes (stable hashing) but not on
  load or memory.
- **No streaming support.**
- **Conversation ID fallback.** Without the header, two users starting with
  identical messages share a key.
