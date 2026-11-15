# Phased Development Plan — Strata Experts Management for NInfer Orin

**Branch:** `feat/add-strata-experts-management` (from `6670926f`)
**Base architecture:** `sm_87` (Jetson Orin AGX)
**Target models:** Qwen3.8 Flash-Next variants → Qwen4 MoE (roadmap)
**Created:** 2026-11-15

---

## Context & Objectives

This branch enriches `ninfer-jetson-orin`'s existing MoE foundation — currently a working but narrow `SparseMoeOp` for exact Qwen3.6 35B-A3B geometry — with production-grade experts management infrastructure adapted from upstream `ninfer`. The goal is a complete, configurable, measurable MoE execution pipeline on `sm_87` supporting Qwen3.8 Flash-Next model families and extensible to Qwen4 MoE.

### Design Principles

1. **Backward compatible** — existing `SparseMoeOp` remains usable as fallback/compatibility path.
2. **Configuration-driven** — all new behavior opt-in via `MoeOptions`; defaults preserve current behavior.
3. **Architecture-aware** — every upstream component evaluated for `sm_87` compatibility; no TMA, no NVFP4 hardware intrinsics, no sm_120a-specific optimizations leak in.
4. **Measurable** — each phase ships with correctness verification and paired benchmark controls per strata-ninfer benchmark protocol.
5. **Reference-preserving** — upstream code used as reference material; ported components re-written for sm_87 with source origin documented.

### Product Contract Alignment

The current product contract targets single-GPU, single-model, 1–8 concurrent requests on RTX 5090 (`sm_120a`). This branch extends the runtime for `sm_87` Jetson Orin, which has different memory topology, bandwidth, and SM characteristics. No changes to the host-side engine contracts (`include/ninfer/engine.h`, `include/ninfer/types.h`) beyond adding MoE-specific config structs that are purely additive.

---

## Phase 1 — Weight Binding Abstraction & Execution Wrapper

**Gap items:** 1, 2, 3 | **Estimate:** 2–3 weeks

### Deliverables

1. **Affine bank description layer** — Port `ExpertBinding`, `BankDescription`, `ExpertDescription` from `ninfer/src/models/common/moe/weights.h/cpp` with sm_87 layout adjustments. Define rank-3 logical bindings over affine stored banks for gate_up and down weight matrices.

2. **ResidentExpertBank class** — Implement the resident expert bank abstraction with device view, mask, and subset support. Wraps a collection of individual expert weight tensors with geometric metadata.

3. **Execution wrapper around SparseMoeOp** — Create `Parameters` struct (profile + control + shared + bank + activation_calibration) and an `execute()` function that maps `SparseMoeOp` internal calls through the upstream execution abstraction while preserving current CUDA kernel launches.

4. **Configurable epilogue** — Replace hardcoded AddResidual with configurable `MoeEpilogue` enum (Write / AddResidual), using existing conditional logic already partially present in the sparse_moe kernels.

5. **Token validity support** — Wire masked token handling through to the op level via `MoeTokenValidity`, enabling non-contiguous token batches.

### Files to add/modify

| Action | File | Source Origin |
|--------|------|---------------|
| Add | `src/models/common/moe/weights.h/cpp` | ninfer src/models/common/moe/weights* |
| Add | `src/models/common/moe/execution.h/cpp` | ninfer src/models/common/moe/execution* |
| Add | `include/ninfer/core/expert_cache.h` | Partial: stub only (full cache in Phase 2) |
| Modify | `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` | Existing: add Epilogue parameter |
| Modify | `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu` | Existing: add Epilogue parameter |
| Modify | `src/ops/wrapper/sparse_moe.cpp` | Existing: route through new execute() |
| Add | `include/ninfer/types.h` additions | MoeOptions struct (stub, full version in Phase 2) |

### Verification

- **Parametric:** Correctness tests at fixed geometry (256 routed + 1 shared, top-8, profile matching Qwen3.6 35B-A3B). Compare output against current implementation byte-for-byte. Verify both Write and AddResidual epilogues produce correct results.
- **Multivariate:** Performance comparison (current vs. wrapped execution) shows <5% regression from indirection overhead. Throughput across batch sizes 1–8 matches within measurement noise.

### Risks & sm_87 Adaptations

- Shared memory limits: sm_87 has 64 KB/block vs. sm_120a's larger limit. Kernels using >32 KB must be verified not to exceed this.
- No architectural change needed for this phase since we wrap existing working kernels.

---

## Phase 2 — ExpertCache & Configuration Infrastructure

**Gap items:** 5, 7, 8 | **Estimate:** 3–4 weeks

### Deliverables

1. **ExpertCache core** — Implement `ExpertCache` class from `ninfer/src/core/expert_cache.h/cpp` adapted for sm_87. Supports prepare/progress/acquire/release lifecycle. Fixed entries occupy `[0, fixed_bytes)`; streaming uses bounded ring. Resident-only mode (zero-cache) where all experts pre-loaded is the default for initial deployment.

2. **Layout tracking** — `ExpertCacheRequirements` and `ExpertCacheConfig` structs with capacity_bytes, fixed_bytes, max_batch_bytes/copies, upload_slots, max_in_flight_uses. Adjust default values for Orin PCIe bandwidth.

3. **MoeOptions configuration** — Add full `MoeOptions` struct to `include/ninfer/types.h`: enable flag, resident_capacity, fixed_experts_per_layer, cache_priority, prefetch flags, cpu_workers, cpu_queue_depth. All fields have sensible defaults that preserve current behavior when disabled.

4. **MoeMemorySummary observability** — Track fixed/target/ready expert counts per layer, PLE device/pinned/host bytes, cache usage metrics. Integrate with engine-level observability without changing existing snapshots.

5. **Batch execution plan integration** — Port `BatchExecutionPlan` and `PhysicalBatchShape` from `ninfer/src/models/common/batch.h/cpp` to give MoE-aware scheduling proper physical token geometry tracking.

### Files to add/modify

| Action | File | Source Origin |
|--------|------|---------------|
| Add | `src/core/expert_cache.h/cpp` | ninfer src/core/expert_cache* |
| Add | `src/models/common/batch.h/cpp` | ninfer src/models/common/batch* |
| Modify | `include/ninfer/types.h` | Append MoeOptions struct |
| Modify | `include/ninfer/observability/resources.h` | Append MoeMemorySummary |
| Modify | `src/artifact/bindings.cpp` | Use ExpertDescription for MoE weights |
| Modify | `src/targets/qwen3_6_35b_a3b/impl/config.h` | Wire MoeOptions into model config |

### Verification

- **Parametric:** ExpertCache invariant tests — prepare creates correct layout, acquire returns valid device pointers, release marks entry reusable. Resident-only mode (all experts pre-loaded) produces identical results to Phase 1.
- **Multivariate:** Memory accounting accuracy — total VRAM reported matches `nvidia-smi` readings within 1%. Cache hit rate measurements are deterministic under repeated identical workloads.

### Risks & sm_87 Adaptations

- pinned_memory allocation on Orin may fail if system has limited pinned memory budget. Need graceful degradation.
- max_batch_bytes (default 96 MB) needs tuning for Orin's slower PCIe. May reduce to 32 MB or lower initially.

---

## Phase 3 — Schedule Orchestration

**Gap items:** 6, 12 | **Estimate:** 3–4 weeks

### Deliverables

1. **Schedule class** — Implement the full `Schedule` orchestration from `ninfer/src/models/common/moe/schedule.h/cpp`. Activity phase dispatches layers in priority order; Preparation phase handles cache population ahead of compute. Each layer executes asynchronously via `LayerGraphExecutable` start/poll/cancel cycle.

2. **Layer graph definition/executable hierarchy** — Create `LayerGraphDefinition` and `LayerGraphExecutable` classes wrapping CUDA Graph capture and replay per layer. Enable async execution with proper stream ordering.

3. **Hybrid policy** — CPU workers + GPU co-execution when batch is undersubscribed. Initially conservative: CPU worker threads perform minimal staging prep while GPU processes other tokens.

4. **Priority-based expert loading** — Layers with higher priority get their experts loaded first into the resident bank. Priority derived from transformer depth (deeper layers scheduled last during decoding).

### Files to add/modify

| Action | File | Source Origin |
|--------|------|---------------|
| Add | `src/models/common/moe/schedule.h/cpp` | ninfer src/models/common/moe/schedule* |
| Add | `src/models/common/moe/graph.h/cpp` (new) | Hybrid of schedule + cuda_graph patterns |
| Modify | `src/targets/qwen3_6_35b_a3b/impl/variant.cpp` | Replace direct sparse_moe calls with schedule.execute() |
| Modify | `src/runtime/engine.cpp` | Initialize and manage Schedule instance |

### Verification

- **Parametric:** Schedule ordering tests — given N layers with known priorities, verify execution order matches priority list. Async start/poll/cancel respects layer dependency boundaries.
- **Multivariate:** End-to-end latency at batch size 1 should match or slightly improve over current synchronous path (GPU utilization improves from idle time between layer launches being reduced).

### Risks & sm_87 Adaptations

- CUDA Graph APIs may behave differently between Orin's CUDA 12.x and desktop CUDA 12.8+. Must validate capture/replay on target hardware.
- Graph capture overhead is significant on slow GPUs. Batch sizes ≥4 recommended before enabling graph capture.

---

## Phase 4 — Offload Planning & CPU Staging

**Gap items:** 9, 10, 11 | **Estimate:** 4–5 weeks

### Deliverables

1. **Offload planning** — Implement `Offload::Plan` from `ninfer/src/models/common/moe/offload/plan.h/cpp`. Determines which experts fit in cache, which require PCIe staging, timing budgets for transfers vs. compute overlap.

2. **Execution bridge** — `offload/execution.h/cpp` with `ExecutionRequirements`: device_bytes, pinned_bytes, cpu_workspace_bytes. Manages pinned host buffers for efficient PCIe transfers.

3. **CPU worker pool** — Thread pool with NUMA affinity control for expert data staging. Handles async host-to-device copies coordinated with GPU compute streams.

4. **Staging buffer management** — Queue-based staging with configurable depth (`cpu_queue_depth` in MoeOptions). Multiple outstanding transfers supported up to queue limit.

### Files to add/modify

| Action | File | Source Origin |
|--------|------|---------------|
| Add | `src/models/common/moe/offload/plan.h/cpp` | ninfer src/models/common/moe/offload/plan* |
| Add | `src/models/common/moe/offload/execution.h/cpp` | ninfer src/models/common/moe/offload/execution* |
| Add | `src/models/common/moe/offload/resources.h/cpp` | ninfer src/models/common/moe/offload/resources* |
| Modify | `src/core/expert_cache.cpp` | Integrate offload transfer coordination |
| New | `src/core/cpu_thread_pool.h/cpp` | Project-owned: simple NUMA-aware thread pool |

### Verification

- **Parametric:** Pinned memory allocation succeeds and can back a cudaMemcpyAsync transfer. Worker pool correctly dispatches staging tasks and synchronizes before use.
- **Multivariate:** With partial resident cache (e.g., 32 of 256 experts resident), show that staged expert fetch completes within expected PCIe time. No regressions when cache fully resident (Phase 3 baseline still optimal).

### Risks & sm_87 Adaptations

- Orin's integrated CPU/GPU shares unified memory but also supports discrete pinned allocations. Must test both paths.
- PCIe bandwidth on Orin (~16 GB/s theoretical) is significantly lower than desktop platforms. Transfer-heavy workloads may show poor overlap efficiency.

---

## Phase 5 — Routing Enhancement

**Gap items:** 4 | **Estimate:** 2–3 weeks

### Deliverables

1. **Generic routing dispatcher** — Port shape-specific routing routines from `ninfer/src/ops/moe/kernels/cuda/topk_softmax_route/` with sm_87 kernel adaptation. Currently `sparse_moe_route.cuh` handles top-8 selection; extend to support configurable top-k and additional quantization formats.

2. **Score normalization improvement** — Use upstream's softmax-based scoring with proper numerical stability. Current implementation does basic score normalization.

3. **Routing kernel stream binding** — Ensure routing computation binds to the correct CUDA stream for async scheduling integration.

### Files to add/modify

| Action | File | Source Origin |
|--------|------|---------------|
| Modify | `src/ops/sparse_moe/sparse_moe_route.cuh` | Extend with configurable top-k and softmax dispatch |
| Add | `src/ops/moe/routing.h/cpp` | Generic routing interface (thin wrapper) |
| Modify | `src/ops/sparse_moe/decode/sparse_moe_decode_kernels.cu` | Accept routing result from generic dispatcher |
| Modify | `src/ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu` | Same |

### Verification

- **Parametric:** Route output (top-k expert indices and corresponding attention scores) compared against naive CPU softmax-topk at multiple input distributions. Numerical tolerance: 1e-5 FP32.
- **Multivariate:** Routing overhead measured as fraction of total layer latency. At top-8, should add <2% overhead. At top-1, <1%.

### Risks & sm_87 Adaptations

- Shared memory optimization strategies may differ from sm_120a. Block geometries optimized for Ada Lovelace need profiling and adjustment.
- No major architectural blockers expected — routing is mostly reduction + scatter operations well-supported on sm_87.

---

## Phase 6 — PLE Kernel API (Deferred/Optional)

**Gap item:** 13 | **Estimate:** 2–3 weeks if pursued

### Status: Lower priority for sm_87

PLE (Prefetch Loop Enhancement) leverages high-bandwidth memory copy engines (TMA on sm_90a+, register-file transfers on sm_120a) to overlap expert loading with compute. On sm_87 Orin, the lack of TMA means the primary benefit vector is unavailable. The fallback is PCIe-based prefetching, already addressed by Phase 4's staging infrastructure.

### Decision rationale

- **Not prioritized** unless Qwen4 MoE requires >256 active experts causing PCIe contention that exceeds compute parallelism.
- If implemented later, simplify to CPU-initiated async prefetch into staged buffers rather than replicating upstream TMA/register-based approaches.

### Files (future)

If pursued, would touch `src/ops/ple.h` concepts and integrate with Phase 4 staging infrastructure. No direct upstream file imports needed.

---

## Integration Milestones

### Milestone 1 (Phase 1 complete) — ~5–6 weeks
Working weighted expert abstraction. Models load with `ResidentExpertBank`, execute through the new wrapper, epilogue is configurable. No regressions vs. current behavior.

### Milestone 2 (Phases 1–2 complete) — ~8–10 weeks
Full configuration surface. ExpertCache supports resident-only mode (default) and cache-mode (future). Memory tracking operational. Ready for multi-layer experiments.

### Milestone 3 (Phases 1–3 complete) — ~11–14 weeks
Asynchronous schedule with CUDA Graph capture. End-to-end inference through the new pipeline. Latency comparable to or better than current synchronous path.

### Milestone 4 (Phases 1–4 complete) — ~15–19 weeks
Complete experts management stack including offload planning and CPU staging. Scalable to configurations exceeding available VRAM (expert eviction/staging).

### Milestone 5 (All phases complete) — ~18–24 weeks
With optional Phase 6 excluded by decision. Full production-grade MoE infrastructure adapted for sm_87, ready for Qwen3.8 Flash-Next testing and eventual Qwen4 MoE support.

---

## Measurement Protocol

Each phase follows the strata-ninfer measurement rules:

1. **Same-day paired controls** — Baseline and experiment measured on same hardware, same thermal state, sequential order randomized.
2. **Identical workload** — Same tokenizer, same prompt length, same sampling strategy, same number of tokens generated.
3. **Report format** — Per-phase summary includes: metric names, units, sample count, mean ± stddev, relative change %, p-value threshold if applicable.
4. **Opt-in experiments** — All new functionality gated behind `MoeOptions.enable = true`; default preserves current behavior.
5. **Physical memory authority** — Single authoritative memory tracker. The 6 GiB host headroom and workspace reserve preserved from strata rules.

### Benchmark templates

For each phase, maintain a consistent benchmark script covering:
- Batch sizes: 1, 2, 4, 8
- Token counts: 64, 128, 256, 512 generate
- Quantization modes: groupwise-int (primary), nvfp4 (secondary, if sm_87 kernels available)

---

## References

- Gap analysis: [GAP_ANALYSIS.md](./GAP_ANALYSIS.md)
- Upstream MoE weights: [ninfer/src/models/common/moe/weights.h](../../../ninfer/src/models/common/moe/weights.h)
- Upstream MoE execution: [ninfer/src/models/common/moe/execution.h](../../../ninfer/src/models/common/moe/execution.h)
- Upstream MoE schedule: [ninfer/src/models/common/moe/schedule.h](../../../ninfer/src/models/common/moe/schedule.h)
- Upstream expert cache: [ninfer/src/core/expert_cache.h](../../../ninfer/src/core/expert_cache.h)
- Upstream ops/moe: [ninfer/src/ops/moe/](../../../ninfer/src/ops/moe/)
- Upstream types.h MoeOptions: [ninfer/include/ninfer/types.h](../../../ninfer/include/ninfer/types.h)
- Strata NInfer execution plan: [strata-ninfer-jetson-orin/docs/integration/NINFER_EXPERT_EXECUTION_PLAN.md](../../strata-ninfer-jetson-orin/docs/integration/NINFER_EXPERT_EXECUTION_PLAN.md)
- Strata AGENTS.md rules: [strata-ninfer-jetson-orin/AGENTS.md](../../strata-ninfer-jetson-orin/AGENTS.md)
