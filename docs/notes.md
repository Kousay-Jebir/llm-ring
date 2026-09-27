# Lab notes

## Environment
- Machine: Windows (PowerShell), Docker Desktop with WSL2 backend
- CPU / RAM:
- kind version:
- Kubernetes version (kind node image): v1.37.0
- llama.cpp build: b11176 (`system_fingerprint` in responses)
- Model: Qwen2.5-0.5B-Instruct, Q4_K_M quantization, CPU only, 2 threads per replica

## Cluster setup
- Time for a fresh replica to become ready (download + load):
- Time for a restarted replica (model already on disk): model loads in about 4 s
  (from the server log: `model loaded` at ~4.2 s)
- Node placement of llm-0 / llm-1 / llm-2:
- Time for llm-3 to become ready after scaling to 4:
- Server slots: 4 parallel slots per replica (`n_slots = 4`)

## Prompt cache, single replica (by hand, llm-0 directly)

Same 28-token request sent twice:

| Request | cache_n | prompt_n | prompt_ms |
|---|---|---|---|
| First (cold) | 0 | 28 | 375.8 |
| Second (identical) | 27 | 1 | 31.8 |

- Prompt processing about 12x faster with the cache.
- Cold prompt processing: ~74 tokens/s. Generation: ~32 tokens/s.
- The server always reprocesses at least the last token, hence 27 of 28.
- With `temperature` unset, the answer text differs between identical requests.
  Experiments must set `temperature: 0` and a fixed `seed`.

## Prompt cache through the router (Test 4)

Two-turn conversation `test-conv-1`, `consistent` policy, `max_tokens = 40`:

| Turn | Replica | Reason | cache_n | prompt_n |
|---|---|---|---|---|
| 1 | llm-2 | home | 0 | 38 |
| 2 | llm-2 | home | 77 | 17 |

- The router kept the conversation on its home replica; 77 of 94 prompt tokens
  (~82%) came from cache.
- Turn 1 cached 38 prompt + 40 generated = 78 tokens; turn 2 reused 77 (last
  token always reprocessed). The 17 new tokens are the new question plus chat
  template markers.
- A one-line question costs 38 tokens because of the chat template (role markers,
  and Qwen's default system message when none is sent).
- The turn-2 answer was cut off: it hit `max_tokens = 40`.

## Spread of conversations (Test 5)

Nine single-message conversations, `consistent` policy:

| Replica | Conversations |
|---|---|
| llm-0 | spread-4, 5, 7, 8 |
| llm-1 | spread-6, 9 |
| llm-2 | spread-1, 2, 3 |

- Uneven split is expected with 9 samples (an exact 3-3-3 happens ~8.5% of the time).
- Rerunning gives the same mapping: the hash is deterministic.
- All reasons were `home`: requests were sequential, so no replica ever had
  in-flight load. Overflow only appears with concurrent traffic.

## Scale-up (Test 7)
- Conversations that moved after 3 -> 4 with `consistent`:
- Conversations that moved after 3 -> 4 with `mod-n`:
- Time from `kubectl scale` to `"added": ["llm-3"]` in router logs:

## Configuration changes
- `LLAMA_ARG_CTX_SIZE` 8192 -> 16384: 4096 tokens per slot instead of 2048, so
  long conversations (pasted documents, many turns) don't exceed their slot.
- `LLAMA_ARG_CACHE_RAM: 512`: host-memory prompt cache (MiB), so conversations
  pushed out of a slot keep their cache. Bounded to stay under the 2Gi pod limit.
- `persistentVolumeClaimRetentionPolicy: Retain` on the StatefulSet: llm-3 keeps
  its model when scaled away and back.

## Surprises / things that didn't work first time
- **Cluster creation failed (kubeadm: connection refused on :6443).** Cause: Docker
  was on cgroup v1, which recent kubelets refuse. Fix on Windows: `wsl --update`,
  WSL2 engine enabled in Docker Desktop, and if still v1, add
  `kernelCommandLine = cgroup_no_v1=all` under `[wsl2]` in `%UserProfile%\.wslconfig`,
  then `wsl --shutdown` and restart Docker Desktop. Verify with `docker info`.
  Debug tip: `kind create cluster --retain`, then read the kubelet log with
  `docker exec <node> journalctl -u kubelet`.
- **Namespace missing after `set-context`.** `kubectl config set-context
  --namespace` only changes a local default; the namespace itself is created by
  applying `00-namespace.yaml`.
- **Metrics endpoint silently disabled.** The variable was first written as
  `LLAMA_ARG_ENDPOINTS_METRICS`; the correct name is `LLAMA_ARG_ENDPOINT_METRICS`.
  llama.cpp ignores unknown variables without any error. Check names with
  `llama-server --help`.
- **Same numbers twice in PowerShell.** Reading `$r` again doesn't resend the
  request; the `Invoke-RestMethod` line must run again.
- **Port-forward connection refused.** Forwarded `pod/llm-0 8000:80`, but replicas
  listen on 8080; the router is `svc/router 8000:80`.