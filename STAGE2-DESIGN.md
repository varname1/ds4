# CUDA token-graph Stage 2: replay without re-capture

Status: DESIGN (Stage 1 implemented in this branch; see the
`cuda: single-launch token decode via CUDA graph capture` commit).

## Where Stage 1 landed (measured, H200 NVL, DeepSeek V4 Flash q2-imatrix)

| Mode | decode t/s | ms/token |
|------|-----------|----------|
| Normal path (launch-per-op) | 34.9–36.3 | ~28 |
| Stage 1: capture+ExecUpdate+launch per token (`DS4_CUDA_GRAPH=1`) | 30.5–31.4 | ~32 |

Stage 1 is **correct** (identical greedy output, 1,718-node graph, transparent
fallback, zero request failures) but **slower**, because re-capturing every
token serializes what the normal path overlaps: the CPU encode/capture pass
(~23 ms — ds4.c arg logic + 1,718 capture-intercepted launches + ExecUpdate)
runs to completion before the GPU starts the ~9 ms dense execution.
Normal path: ~max(23, 9)+tail ≈ 28 ms. Stage 1: ~23 + 9 ≈ 32 ms. QED.

The prize: eliminate the per-token CPU pass → ~9–10 ms/token → **~100 t/s**
(11× the H200's bandwidth-efficiency deficit vs Metal finally closed).

## Why re-capture is needed today

Per-token scalars are baked into kernel args. Exactly these ops consume them
in the decode layer (verified by arg audit):

| Op (wrapper) | per-token scalars |
|--------------|-------------------|
| `ds4_gpu_embed_token_hc_tensor` | token |
| `ds4_gpu_rope_tail_tensor` | pos |
| `ds4_gpu_head_rms_norm_rope_tail_tensor` | pos |
| `ds4_gpu_kv_fp8_store_raw_tensor` | raw_row |
| `ds4_gpu_router_select_tensor` | token (hash routing) |
| `ds4_gpu_compressor_update_tensor` | pos |
| `ds4_gpu_attention_decode_heads_tensor` | n_raw, n_comp[il] |
| `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` | pos, n_raw, n_comp[il], n_selected[il] |
| `ds4_gpu_indexer_score_one_tensor` / `indexer_topk_tensor` chain | pos, n_index_comp[il] |

Everything else (13 q8 matmuls/layer, norms, swiglu, hc ops, output head,
argmax) takes only stable dims/pointers.

## Stage 2 design

### 1. Pinned step-state struct (no param patching!)

```c
struct ds4_step_state {
    uint32_t token, pos, raw_row, n_raw;
    uint32_t n_comp[DS4_MAX_LAYER];
    uint32_t n_index_comp[DS4_MAX_LAYER];   /* n_selected = min(TOP_K, n_index_comp) derivable in-kernel */
};
```

- Host side: ONE persistent `cudaMallocHost` (pinned) instance.
- Device side: `__device__ ds4_step_state g_step_state_dev;`
- The FIRST op inside every captured token is
  `cudaMemcpyToSymbolAsync(g_step_state_dev, pinned, ..., cudaStreamPerThread)`.

Key property: a graph **memcpy node re-reads its host source on every
launch**. Replay = write fresh values into the pinned struct + relaunch the
SAME exec. No node introspection, no `cudaGraphExecKernelNodeSetParams`.

### 2. Kernel templatization (byte-identical default path)

For each kernel behind the wrappers above (~20–25 including variants:
attention_decode_mixed / heads8_online / static, indexer scores
direct/wmma/wmma32/wmma64, router select ×3, compressor ×4, rope ×2,
store_raw_kv, embed, topk size variants):

```cpp
template <bool STEP>
__global__ static void rope_tail_kernel(..., uint32_t pos0_arg, ...) {
    const uint32_t pos0 = STEP ? g_step_state_dev.pos : pos0_arg;
    /* body unchanged */
}
```

Per-layer counters index by an added stable `il` arg:
`STEP ? g_step_state_dev.n_comp[il] : n_comp_arg`.

Wrappers select `<true>` when `g_token_graph_capturing` is set (capture only
ever happens on the single-token decode path, so prefill/batch callers are
untouched by construction). `<false>` instantiation keeps today's binary
behavior bit-for-bit.

### 3. Replay eligibility (exact, no heuristics)

Replay the cached exec for a token iff ALL hold:

1. **No emit topology this token.** Flash emit cadence is pure in pos:
   layers 0–1 never compress; even layers emit at `(pos+1) % 4 == 0`; odd
   layers at `(pos+1) % 128 == 0` (`ds4_expected_layer_compress_ratio`).
   Non-emit tokens also mutate NO counters → zero bookkeeping to replicate
   on replay (only the cur_hc/after_ffn_hc parity swap — 43 layers = odd →
   one swap — and pos advances in the caller).
   Emit tokens (~25%+~1%) take the existing Stage-1 capture path, which
   also refreshes the cached exec via ExecUpdate.
2. **Variant signature matches.** Kernel-variant choices depend on the same
   tracked scalars via wrapper predicates. Compute a signature from the SAME
   predicates (single source of truth — export tiny query helpers from
   ds4_cuda.cu rather than duplicating thresholds):
   - per-layer sparse-indexer gate: `n_comp[il] > decode_sparse_threshold` (ds4.c)
   - attention score-buffer fits: `cuda_attention_score_buffer_fits(n_comp)`
   - indexer top-k size bucket (1024/2048/8192 kernel variants)
   Signature change → capture (rebuilds exec + stores new signature).
3. **Alloc epoch matches** the epoch at capture (already tracked by Stage 1:
   any lazy cudaMalloc/cudaFree bumps it). This catches `cuda_tmp_alloc`
   growth in the indexed-attention path (its pointer is baked in the exec).

Expected mix: ~75% replay tokens at ~9–10 ms + ~25% capture tokens at ~32 ms
→ **~15 ms avg ≈ 65–70 t/s**.

### 4. Extension to ~105 t/s (later)

Add a second cached exec for the `emit4` topology. Requirements beyond
Stage 2: compressor/indexer emit-row writes use `ds4_gpu_tensor_view`s with
per-emit ring offsets baked as pointers — convert `compressor_store` /
`compressor_set_rows` (and the indexer row write) to take (base, row) with
row read from the step-state, and replicate the counter increments on
replay (pure in pos, same cadence rule). Then only every-128th token
recaptures: ~9.3 ms avg ≈ **~105 t/s**.

### 5. Hazards already handled by Stage 1 machinery (keep them)

- Warmup: no capture until 4 consecutive alloc-free tokens.
- Any capture failure → return 2 → token transparently re-runs uncaptured
  (compressor counters + hc pointers snapshot/restored around the attempt).
- `allow_split_flush=false` during capture (mid-encode deviceSync).
- `--default-stream per-thread` + cuBLAS pinned to `cudaStreamPerThread`.
- Sync `cudaMemset` in routed_moe batch path already converted to Async.

### 6. Validation plan

1. Graphs off vs on: identical greedy 512-token output (Stage 1 already
   passes this).
2. Replay correctness: compare full logits of a replayed token vs the same
   token forced through capture (env knob to force-capture every token).
3. Long-context soak past variant thresholds (n_index_comp crossing 1024,
   2048; n_selected saturation at 512; ctx > raw window) — signature must
   force recaptures at each crossing.
4. kv-disk-cache save/restore across replayed tokens.
5. Benchmark matrix: 512-token greedy at depth 0 / 2k / 8k.
