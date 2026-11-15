# MoE Experts Management — Gap Analysis

**Created:** 2026-11-15
**Scope:** Compare upstream `ninfer` MoE capabilities against `ninfer-jetson-orin` and strata variants.
**Purpose:** Identify what needs backporting/adapting for `sm_87`/Orin from upstream's production-grade MoE infrastructure.

---

## Executive Summary

Upstream `ninfer` has **complete, production-grade MoE experts management** covering model loading, expert residency, physical memory management, routing, request scheduling, GPU-resident execution, bounded staging, offloading, PLE prefetch, telemetry, and configuration. It targets Qwen3.5 MoE (35B-A3B) and Qwen4 Exp with `sm_120a`.

`ninfer-jetson-orin` currently has only an exact-geometry sparse-MoE Op (`SparseMoeOp`) for the Qwen3.6 35B-A3B architecture, executing resident experts inline with no cache, no offload, no scheduling, no PLE, and no telemetry. The gap is substantial but well-documented — it maps to a clear incremental migration path.

Strata variants maintain separate MoE systems entirely: `strata-inference-orin` runs Strata's own fused MOE kernels with its expert residency model; `strata-ninfer-jetson-orin` documents E0–E7 stages for integration but no implementation has started.

---

## Capability Matrix

| Capability | Upstream `ninfer` | `ninfer-jetson-orin` | `strata-inference-orin` | `strata-ninfer-jetson-orin` |
|---|---|---|---|---|
| **Target architectures** | `sm_120a` (RTX 5090/DGX Spark) | `sm_87` (Jetson Orin AGX) | `sm_87` (or `sm_90a` with CUDA 12.x+) | `sm_87` |
| **Model types** | `Qwen3_5Moe`, `Qwen4Exp` | Qwen3.6 35B-A3B only | Multiple Strata Flash-Next variants | Multiple Strata variants |
| **Expert count support** | Configurable via geometry | Hardcoded 256 routed + 1 shared | Configurable (up to 24,576 total) | TBD |
| **Weight quantization** | NVFP4, FP8, BF16, Q4/Q5/Q6/Q8 | W4/W5 routing, W8 shared | UD-Q4_K_XL etc. (GGUF-style) | GGUF-style via Strata |

### 1. Model Loading & Expert Weight Binding

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Binding layer** | `models/common/moe/weights.h/cpp` — `ExpertBinding`, `ResidentExpertBank`, `bind_banks()` | `targets/qwen3_6_35b_a3b/impl/load/bindings.cpp` — direct weight assignment per tensor |
| **Affine rank-3 banks** | Yes — three rank-3 logical `[E,I,H]`, `[E,I,H]`, `[E,H,I]` bindings over affine stored banks | No — single tensor per weight slot |
| **Resident expert bank abstraction** | `ResidentExpertBank` class with device view, mask, subset | None — weights are plain tensors in config |
| **Multi-bank support** | Yes — multiple `ExpertBankId` across layers | N/A (single layer type) |
| **Geometry parameters extracted** | gate_up/down shapes + expert count + subset kind | Hardcoded in source files |

**Gap:** Missing weight-binding abstraction layer, affine bank description, resident expert bank class. All three can be ported directly from upstream but must handle `sm_87` memory layout differences.

### 2. Expert Residency Management

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Cache system** | `core/expert_cache.h/cpp` — full `ExpertCache` class | None |
| **Fixed vs. streaming experts** | Yes — fixed entries occupy `[0, fixed_bytes)`; streaming uses bounded ring | All experts always resident |
| **Cache capacity config** | `ExpertCacheConfig` — capacity_bytes, fixed_bytes, max_batch_bytes/copies, upload_slots, max_in_flight_uses | N/A |
| **Resident-only mode** | Yes — zero-cache mode where all experts pre-loaded | Implicit (equivalent) |
| **Placement tracking** | `ExpertPlacement` with offset, key-based lookup | N/A |
| **Device/pinned/host bytes** | Tracked via `MoeMemorySummary` | Not tracked |

**Gap:** Complete absence of `ExpertCache` infrastructure. The entire cache lifecycle (prepare → progress → acquire → release) must be ported. Key adaptation: `sm_87` lacks TMA/Tensor Core DMA available on `sm_120a`, affecting upload patterns.

### 3. Physical Memory Management

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Memory bookkeeping** | `MoeMemorySummary` — tracks fixed/target/ready experts per layer; PLE device/pinned/host bytes | None (implicit VRAM) |
| **Offload planning** | `offload/plan.h/cpp` — `Plan` struct with cache, bridge, team, timing budgets | None |
| **Bridge (staging) memory** | `ExecutionRequirements` — device_bytes, pinned_bytes, cpu_workspace_bytes | None |
| **Pinned memory for PCIe transfers** | Yes — explicit pinned allocation | No pinned buffers |
| **Window management** | Multi-window expert staging with window_count | Single implicit window |
| **Headroom strategy** | Engine-level headroom | JetPack/system limits |

**Gap:** Entire physical memory authority layer is missing. This includes `describe_expert_cache()`, `ExpertCacheRequirements`, `Offload::Plan`, `ExecutionRequirements`, and pinned memory allocation. Critical for any non-trivial expert scale (>256).

### 4. Routing & Token Selection

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Top-k routing kernel** | `ops/moe/topk_softmax_route/` and `topk_sigmoid_route/` — arch-specific implementations | `sparse_moe_route.cuh` — basic top-8 selection, score normalization |
| **Routing quantization** | Top-k with softmax/sigmoid gating, arch-specific dispatch | Simple softmax normalization |
| **Route computation** | GPU kernel with proper stream binding | Inline CUDA in `sparse_moe_*_kernels.cu` |
| **Score handling** | Full score matrix with per-expert scoring | Simplified scoring |

**Gap:** The sparse-MoE route in `ninfer-jetson-orin` is functional for the specific Qwen3.6 geometry but lacks the generic routing infrastructure. Upstream provides shape-specific dispatched routines for multiple quantizations — this is reusable with `sm_87` kernel adaptation.

### 5. Request Scheduling & Batch Execution

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Batch execution plan** | `models/common/batch.h/cpp` — `BatchExecutionPlan`, `PhysicalBatchShape`, `graph_rows` | `variant.h` — flat batch token count |
| **Schedule orchestration** | `moe/schedule.h/cpp` — `Schedule` class with activity/preparation phases, layer traversal | No schedule layer |
| **Per-layer invocation** | `start(ExpertBankId, Tensor&, Tensor&, ...)` with poll/cancel async interface | Direct synchronous call per layer |
| **Async GPU execution** | `LayerGraphExecutable` — start/poll/cancel cycle with Graph support | CUDA Graph not used |
| **Hybrid policy** | CPU workers + GPU co-execution when batch is undersubscribed | CPU fallback not implemented |
| **Priority ordering** | Per-layer priority list determines fixed vs. cache experts | None |

**Gap:** Complete absence of scheduling layer. The `Schedule` class, `Activity`, `Preparation`, `LayerGraphDefinition/Executable` hierarchy must be built or adapted.

### 6. GPU-Resident Expert Execution

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Execution driver** | `moe/execution.h/cpp` — `execute()` function accepting profile, parameters, hints | Direct kernel launches in `sparse_moe_*_kernels.cu` |
| **Epilogue support** | Write / AddResidual via `MoeEpilogue` enum | AddResidual only |
| **Token validity** | `MoeTokenValidity` — masked tokens handled at op level | T assumed contiguous |
| **Architecture-specific kernels** | Separate kernels for DGX Spark (`sm_120a`) and RTX 5090 (`sm_120a`) | Single kernel set for `sm_87` |
| **Workspace management** | Profile-driven workspace calculation | Fixed workspace in plan.cpp |
| **SWiGLU ops** | `ops/moe/swiglu/` — dedicated SWiGLU implementations per quantization | SWiGLU fused into sparse_moe kernels |

**Gap:** The existing `SparseMoeOp` already implements resident execution correctly for its target geometry. The main task is wrapping it with the upstream execution abstraction (`Parameters`, `MoeProfile`, `MoeExecutionHints`) and adding epilogue/token-validity flexibility.

### 7. Bounded Staging / Offloading

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Offload planning** | `offload/plan.h/cpp` — `plan_cache()`, `resolve_plan()` decisions | Not implemented |
| **CPU-to-GPU transfer** | `offload/execution.h/cpp` — `CpuTeamPlan`, `ExecutionRequirements` | Not implemented |
| **Host pinning** | Explicit pinned memory allocation for PCIe | No pinned buffers |
| **CPU worker pool** | Thread pool with affinity control | No CPU expert workers |
| **Queue depth** | Configurable per-model | N/A |
| **Hybrid execution** | CPU+GPU concurrent when batch undersubscribed | N/A |

**Gap:** Entire offload pipeline absent. This is the most complex new infrastructure — CPU pinning, thread pools, staging buffers, queue management. Lower priority since Qwen3.8 35B-A3B fits in Orin VRAM.

### 8. PLE (Prefetch Loop Enhancement) Kernel API

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **PLE kernels** | `ops/ple.h` — `ple_kernels_ready()`, `get_device_expert_ptrs()`, `host_to_ple_transfer()` | Not implemented |
| **PLE prefetch strategy** | `topk_sigmoid_swiglu_moe/` — proactively loads next-layer experts | Not implemented |
| **PLE data structure** | Pre-loaded expert pointers for overlapping compute/transfer | Not implemented |
| **Arch-specific PLE** | DGX Spark TMA-based, RTX 5090 register-based | Not implementable on `sm_87` (no TMA) |

**Gap:** PLE is primarily useful on architectures with high-bandwidth memory copy engines. On `sm_87` (Orin), PCIe bandwidth is the bottleneck, so PLE overlap benefits are smaller. May be deferred or simplified.

### 9. Telemetry & Observability

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **Resource summary** | `observability/resources.h` — `MoeMemorySummary` struct | None |
| **Telemetry fields** | `expert_upload_bytes`, `ple_upload_bytes` | None |
| **Cache usage metrics** | `ExpertCacheUsage` — ready/expert counts, timing, byte counters | None |
| **Performance profiling** | Integration point with engine-level metrics | No MoE-specific metrics |

**Gap:** Missing observability layer. Important for the multi-phase verification approach to measure performance gains.

### 10. Configuration

| Aspect | Upstream `ninfer` | `ninfer-jetson-orin` |
|---|---|---|
| **MoeOptions struct** | `types.h` — `enable`, `resident_capacity`, `fixed_experts_per_layer`, `cache_priority`, `prefetch`, `expert_priority`, `cpu_workers`, `cpu_queue_depth` | Not present |
| **Model-specific defaults** | Configured per-architecture entry point | No MoE-specific config |
| **Runtime overrides** | Supported via engine config parsing | Not supported |

**Gap:** Zero configuration surface for MoE beyond compile-time geometry constants.

---

## Architecture-Specific Adaptation Notes

The following `sm_120a`-specific features in upstream need re-implementation for `sm_87`:

1. **TMA-based uploads** — DGX Spark kernels use TMA (Tensor Memory Accelerator) for bulk transfers. Orin `sm_87` has no TMA; use `cudaMemcpyAsync` with pinned host memory instead.
2. **NVFP4 kernel intrinsics** — Many upstream NVFP4 kernels use `wmma.mma.sync` instructions specific to `sm_90a`. Need software-emulated paths for `sm_87`.
3. **RTX 5090 SHARED memory layouts** — Optimized for L2 cache topology different from Orin. Needs tuning.
4. **CUDA Graph capture patterns** — Graph APIs may behave differently between CUDA 12.x (Orin) and 12.8+ (DGX Spark). Must validate.
5. **Thread block geometries** — Block sizes optimized for Ada Lovelace SMs; may be suboptimal for Hopper-like Orin SMs. Profiling required.
6. **Shared memory limits** — `sm_87` has 64 KB shared memory per block vs. `sm_120a`'s larger limit. Kernels using >32 KB shared must be reworked.

---

## Strata Variant Assessment

### strata-inference-orin

- Own independent MoE system with up to 24,576 total experts
- Prefill-only fused kernels (`moe_fused.cu`: 727 lines)
- Native MOE kernel (`native_moe.cu`: 170 lines)
- Resident expert caching with CPU/GPU offloading
- Uses GGUF-style quantization (UD-Q4_K_XL, etc.)
- Shares `experts = 71.7 MB` constant with other strata variants

This is a **parallel implementation**, not a component we can import from. It demonstrates real-world Orin MoE operation but uses completely different internals. Useful as a reference for expected VRAM profiles and performance numbers, not as code to reuse.

### strata-ninfer-jetson-orin

- Documents E0–E7 integration stages in `NINFER_EXPERT_EXECUTION_PLAN.md`
- E0–E7 remain planned but unimplemented
- Integration status shows Phases 1–8 complete for their scopes, but NInfer compute backend not established
- Has NInfer reference snapshot at `reference/ninfer/`
- Plan calls for applying NInfer's proven design to Strata's Flash-Next Orin runtime

The strata-ninfer plan describes a similar migration but from Strata toward NInfer compute. Our branch does the reverse direction: enriching NInfer's existing MoE foundation with the broader management infrastructure.

---

## Summary of Gaps by Priority

### Phase 1 — Essential (blocking for any expansion)
1. Weight binding abstraction (`ResidentExpertBank`, affine banks)
2. Execution abstraction wrapper around existing SparseMoeOp
3. Epilogue support (AddResidual as configurable, not hardcoded)
4. Basic telemetry hooks (byte counters)

### Phase 2 — Important (needed for multi-layer/batch correctness)
5. ExpertCache infrastructure (prepare/progress/acquire/release)
6. Schedule orchestration (Activity/Preparation/LayerGraph pattern)
7. Memory management tracking (`MoeMemorySummary`)
8. Configuration surface (`MoeOptions`)

### Phase 3 — High-value (needed for scale beyond resident)
9. Offload planning and CPU staging buffers
10. Pinned memory allocation for PCIe transfers
11. CPU worker pool for hybrid execution
12. Asynchronous layer graph with polling

### Phase 4 — Future/Advanced
13. PLE kernel API (limited benefit on sm_87)
14. Advanced telemetry and observability integration
15. CUDA Graph optimization for layer pipelines
16. Quantization format expansion (NVFP4 software emulation on sm_87)
