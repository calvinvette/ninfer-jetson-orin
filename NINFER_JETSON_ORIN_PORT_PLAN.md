NInfer Jetson Orin Port Plan

Objective

Port calvinvette/ninfer-jetson-orin to NVIDIA Jetson AGX Orin in controlled phases.

The target environment is:

* NVIDIA Jetson AGX Orin
* ARM64 / aarch64
* Ampere SM87 / compute capability 8.7
* JetPack 6.x
* CUDA 12.6
* native Jetson unified physical LPDDR memory architecture

The port MUST NOT assume that CUDA Unified Memory (cudaMallocManaged) is beneficial merely because Orin has physically unified CPU/GPU DRAM.

Preserve the existing explicit-device-memory execution model initially. Unified Memory, mapped host memory, and other Jetson-specific allocation strategies are optimization experiments to be evaluated only after a correct SM87 implementation exists.

The work should proceed through independently verifiable phases. Do not combine CUDA 12.6 compatibility, SM87 enablement, and memory-architecture optimization into one change.

⸻

Phase 0: Establish the Baseline and Porting Inventory

Before modifying code, inventory assumptions inherited from ninfer-3090.

Identify all code and build logic dependent on:

* CUDA >=12.8 or CUDA 13.x
* x86_64
* SM86
* SM89
* NINFER_SM8X_COMPAT
* compile-time __CUDA_ARCH__
* architecture-specific PTX
* architecture-specific MMA instructions
* Tensor Core tile shapes
* SM count
* warp occupancy
* shared-memory capacity
* register assumptions
* launch bounds
* persistent-kernel scheduling
* memory-bandwidth assumptions
* PCIe/discrete-GPU assumptions
* CUDA Graph behavior
* FP8/NVFP4 availability
* INT8/BF16 paths
* host/device transfer assumptions
* Windows-specific or x86-specific SIMD/intrinsics

Classify findings into four categories:

1. CUDA-toolkit dependency
2. CPU architecture dependency
3. GPU ISA/capability dependency
4. performance-tuning dependency

Do not change kernel schedules merely because the target changes from SM86 to SM87.

Produce a concise porting inventory before implementation.

Record the current SM86 behavior as the regression baseline.

⸻

Phase 1: CUDA 12.6 Compatibility

Goal

Make the existing codebase compile correctly with CUDA 12.6 without yet claiming Jetson or SM87 support.

This phase is a CUDA toolchain backport.

Do NOT introduce SM87-specific kernel changes here.

1.1 Remove the artificial CUDA 12.8 floor

The top-level CMake currently contains:

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)

Determine WHY 12.8 became the minimum.

Do not simply change 12.8 to 12.6.

Search the implementation for APIs, headers, compiler behavior, PTX features, library features, and template/compiler constructs introduced after CUDA 12.6.

For every incompatibility:

* determine whether CUDA 12.6 already provides an equivalent facility;
* use the CUDA 12.6-compatible implementation where practical;
* isolate unavoidable toolkit differences behind compile-time compatibility code only when necessary.

Once demonstrated compatible, change the minimum supported toolkit to CUDA 12.6.

1.2 Preserve SM86 during this phase

Build:

CUDA 12.6
CMAKE_CUDA_ARCHITECTURES=86

The purpose is to separate:

CUDA 13.1 -> CUDA 12.6

from:

SM86 -> SM87

If an SM86 machine with CUDA 12.6 is available, run the existing correctness tests and a real-model smoke test there.

If it is not available, at minimum establish successful CUDA 12.6 compilation and clearly identify runtime validation as pending.

1.3 CUDA 12.6 acceptance gate

Phase 1 is complete when:

* the complete relevant project builds with CUDA 12.6;
* CUDA 12.8+ remains supported;
* existing SM86 source semantics are unchanged;
* no SM87-specific workaround has been introduced;
* CUDA 12.6 compatibility differences are documented where materially relevant.

Do not begin optimization work in this phase.

⸻

Phase 2: Establish Native Jetson/aarch64 Build Support

Goal

Make the project a first-class native ARM64/Linux build before changing GPU execution behavior.

Jetson is not simply an RTX 3090 with CMAKE_CUDA_ARCHITECTURES=87.

2.1 Native Jetson build environment

Add a Jetson-native build path targeting:

Linux
aarch64
CUDA 12.6

Prefer the CUDA/JetPack installation supplied by the Jetson platform.

Do not assume that the existing nvidia/cuda:13.1.2-* Docker images are appropriate for Jetson.

Container support, if added, should use an NVIDIA Jetson/L4T-compatible base corresponding to the target JetPack release.

Native compilation should remain the reference development path initially.

2.2 Audit dependencies

Verify ARM64 availability and behavior of:

* CMake
* Ninja
* GCC/G++
* FFmpeg
* libcurl
* pthreads
* CUDA Runtime
* CUDA Driver API dependencies
* any optional vcpkg dependencies

Prefer Jetson/Ubuntu system packages where appropriate rather than introducing unnecessary dependency builds.

2.3 CPU portability

Find and remove or conditionally isolate genuine x86 assumptions.

Do not create generic portability abstractions unless an actual architecture dependency exists.

Acceptance gate

A native aarch64 CUDA 12.6 build must reach the point where SM87 becomes the remaining GPU-specific blocker.

⸻

Phase 3: Enable SM87 Correctness

Goal

Execute NInfer correctly on Jetson AGX Orin.

Performance is secondary in this phase.

3.1 Add SM87 as an explicit supported architecture

Extend the CMake architecture contract from:

86 | 89

to include:

87

Do NOT silently treat SM87 as SM86 merely because both are Ampere.

Review the meaning of:

NINFER_SM8X_COMPAT

If it genuinely represents capabilities shared by SM86/SM87/SM89, retaining it is reasonable.

If it actually means “RTX 3090 implementation”, rename or subdivide the capability appropriately.

Architecture selection should describe capabilities, not accidentally encode a specific board.

3.2 Kernel-by-kernel qualification

For every registered CUDA Op used by the target model, determine:

* does it compile for SM87?
* does it launch?
* does it produce numerically correct output?
* does it rely on an SM86-specific instruction sequence?
* does it assume an RTX-specific SM count?
* does it assume discrete-GPU memory behavior?
* does it contain a schedule specifically tuned for GA102?

Do not optimize yet.

Where the SM86 implementation is mathematically and architecturally valid on SM87, reuse it as the initial SM87 implementation.

Where it is not, create an explicit SM87 implementation.

3.3 Tensor Core capability

Establish the usable Orin execution paths for:

* BF16
* FP16
* INT8
* supported MMA instructions

Explicitly reject unsupported Blackwell/Ada-only paths rather than allowing accidental compilation or runtime selection.

Do not attempt to emulate FP8/NVFP4 Tensor Core functionality merely to preserve an upstream feature.

3.4 Runtime hardware discovery

Inspect all places where scheduling depends upon device properties.

Runtime properties such as:

* SM count
* shared memory
* maximum active blocks
* memory capacity

must not retain RTX 3090 constants where the algorithm logically depends on the actual device.

At the same time, do not replace explicitly tuned kernel schedules with a generic runtime scheduler merely for portability.

The desired design is:

SM87 has explicit qualified schedules

rather than:

one generic schedule tries to fit every GPU

Acceptance gate

Phase 3 requires:

1. real model loading;
2. short prompt prefill;
3. autoregressive decode;
4. correct output;
5. BF16 KV operation;
6. INT8 KV operation if the current SM8x implementation is valid;
7. MTP operation;
8. CUDA Graph execution;
9. repeated request stability.

Correctness must be checked against the existing mathematical/reference mechanisms rather than merely judging generated text.

⸻

Phase 4: Establish an Orin Performance Baseline

Do this BEFORE changing memory allocation strategy.

Capture at least:

prefill tokens/sec
decode tokens/sec
MTP acceptance rate
effective speculative tokens/step
time-to-first-token
resident memory
maximum practical context
memory bandwidth utilization
GPU utilization
power mode
GPU clocks
memory clocks

Test at least:

MTP off
MTP draft 2
MTP draft 3
MTP draft 4

Do not assume the RTX 3090’s preferred MTP depth transfers to Orin.

Use the existing Qwen artifact and otherwise keep workload parameters constant.

Record Jetson power mode and clocks because Orin performance comparisons are meaningless without them.

This becomes the SM87 baseline.

⸻

Phase 5: Tune SM87 Kernel Schedules

Only after correctness and baseline measurements exist should SM87 diverge materially from SM86.

Profile end-to-end execution first.

Identify the kernels actually responsible for significant wall-clock time.

Likely areas include:

* quantized linear kernels
* attention
* GDN/SSM operations
* RMSNorm
* RoPE
* MoE routing if applicable
* KV append/read
* MTP proposal execution

Tune only kernels shown to matter.

Investigate:

* CTA dimensions
* warps per block
* occupancy
* register pressure
* shared-memory usage
* Tensor Core tile shape
* pipeline depth
* vectorized memory operations
* L2 behavior
* memory-bandwidth saturation

Orin has substantially different compute/memory balance from GA102. A schedule optimized for the RTX 3090 should therefore be treated as an initial candidate, not as the expected optimum.

Retain explicit SM87 schedules when measurements justify them.

⸻

Phase 6: Jetson Memory Architecture Experiments

Critical design rule

Do NOT globally replace:

cudaMalloc(...)

with:

cudaMallocManaged(...)

simply because Jetson has unified physical memory.

Jetson’s CPU and GPU share physical LPDDR, but CUDA allocation type still determines important virtual-memory, residency, caching, synchronization, and access semantics.

Physical memory unification does not imply that CUDA Unified Memory is the fastest allocation model.

The existing device-allocation path is the control case.

6.1 Memory strategies to compare

Evaluate separately:

A. Existing device allocation

cudaMalloc

This remains the baseline and default until another strategy demonstrates an end-to-end advantage.

B. CUDA Managed Memory

cudaMallocManaged

Test only selected allocation classes.

Do not globally convert the engine.

Measure:

* page faults
* migration activity
* synchronization overhead
* GPU throughput
* CPU access cost
* CUDA Graph compatibility
* startup/materialization time

C. Pinned/mapped host memory

Investigate:

cudaHostAlloc
cudaHostRegister
cudaHostGetDevicePointer

for objects genuinely shared between CPU and GPU.

Zero-copy access may be useful for infrequently accessed structures but should not be presumed appropriate for hot model weights or KV storage.

6.2 Evaluate allocation classes independently

Classify NInfer memory into:

1. model weights;
2. KV cache;
3. persistent sequence state;
4. temporary workspace;
5. CUDA Graph-stable buffers;
6. host-visible metadata;
7. model-loading/materialization buffers.

Do not assume the optimal CUDA allocation type is the same for all seven.

My initial hypothesis to test is:

Weights:             cudaMalloc
KV cache:            cudaMalloc
hot workspace:       cudaMalloc
persistent GPU state:cudaMalloc
small shared metadata:mapped/pinned candidate
loading buffers:     pinned/mapped candidate

Managed Memory should have to beat this arrangement empirically.

6.3 Memory acceptance criterion

Adopt a different allocation strategy only if it improves a meaningful end-to-end metric without compromising:

* deterministic execution;
* CUDA Graph address stability;
* memory capacity;
* latency;
* throughput;
* correctness.

Lower copy counts alone are not sufficient evidence.

⸻

Phase 7: Orin-Specific Capacity and Context Optimization

Once memory behavior is understood, retune NInfer’s capacity model for unified system memory.

Do not interpret “64 GB Orin” as “64 GB VRAM available to NInfer.”

Account for:

* Linux
* filesystem cache
* Jetson services
* CPU-side model representation
* CUDA runtime/driver allocations
* multimedia components
* NInfer workspace
* CUDA Graph overhead
* model weights
* KV cache

Determine safe operating headroom empirically.

Then calculate practical context limits for:

* BF16 KV
* INT8 KV

Do not enable a more aggressive KV format unless there is an SM87-qualified implementation with acceptable numerical behavior.

⸻

Phase 8: Final Performance Qualification

Compare:

llama.cpp baseline
NInfer initial SM87 port
NInfer tuned SM87

using identical:

* model
* quantization where reasonably comparable
* prompt
* context
* output length
* MTP configuration
* Jetson power mode
* clocks

Report separately:

prefill tok/s
decode tok/s
TTFT
memory consumption
maximum context
power
tokens/joule

Because Orin is power-constrained, tokens/joule is a first-class metric alongside tokens/sec.

Test MTP depths 2, 3, and 4 explicitly.

Do not select the largest speculative depth by default.

⸻

Implementation Discipline

Each phase should be independently reviewable.

Recommended branch/commit progression:

build: support cuda 12.6
build: support native jetson aarch64
feat(cuda): add sm87 execution target
fix(cuda): qualify sm87 operators
perf(sm87): establish orin schedules
perf(memory): evaluate jetson allocation strategies
perf(sm87): tune context and speculative decoding
docs: document jetson orin qualification

Do not mix speculative optimization with compatibility work.

After each phase:

1. build;
2. run the smallest meaningful correctness gate;
3. record the result;
4. stop and assess before proceeding.

The final implementation should retain the project’s existing philosophy of explicit hardware-specific high-performance paths rather than evolving into a generic CUDA inference framework.




## Execution status: Phase 0 inventory (2026-09-21)

Phase 0 source inventory is complete. The SM86 regression baseline below is recorded
from the existing implementation and published validation, not a new runtime measurement.
Phase 1 investigation has reproduced a real CUDA 12.6 incompatibility. Its acceptance
gate remains open; no toolkit floor, GPU schedule, architecture selection, or allocation
policy has been changed. Assess these prerequisites before continuing implementation.

### Regression baseline

The checkout defaults to SM86 and accepts SM89 explicitly. Both use
`NINFER_SM8X_COMPAT`. Preserve BF16/INT8 group-64 KV, MTP, prefix reuse, CUDA
Graphs, and the existing A16 dequantizing FP8/NVFP4 weight routes. FP8 A8,
NVFP4 A4, FP8 KV and RotorQuant KV are unavailable in this build. Disabling
NVFP4 weight loading to avoid a toolkit dependency would regress this baseline.

`README.md` records the v0.6.0 Windows gate (Qwen3.8 generation, materialization,
request memory, admission, paged KV, prefix reuse, speculative rounds and W8
Linear) and the v0.6.1 Linux CUDA 13.1 compile/link gate (245 steps and application
help). It explicitly leaves Linux real-artifact generation and performance open.
No new SM86 correctness or throughput result is claimed here.

### Porting inventory

| Category | Evidence and implication | Owning phase |
| --- | --- | --- |
| CUDA-toolkit dependency | `CMakeLists.txt` rejects CUDA below 12.8. History shows the RTX 3090 release change lowered 13.1 to 12.8; that diff supplies no separate compatibility rationale. | 1 |
| CUDA-toolkit dependency | `src/ops/linear/nvfp4/nvfp4_codec.cuh` includes `cuda_fp4.h` and uses `__nv_fp4x2_e2m1` in active A16 decode. CUDA 12.6 lacks this header; direct SM86 compilation fails. Replace the decode dependency with an exact E2M1 decoder and qualify all 256 packed bytes, including signed zero, against an independent exact oracle. Keep unsupported A4 quantization separate from supported A16 decode. | 1 |
| CUDA-toolkit dependency | `cuda_fp8.h` is present in the installed 12.6 toolkit. FP8 decode is used by embedding, KV codecs and A16 weight execution; header presence alone does not qualify those translation units. Complete compilation must also check C++20 templates, cooperative launch APIs, graph instantiate/update and device linking. | 1 |
| CPU architecture dependency | No x86 SIMD headers, x86 architecture preprocessor guards or AVX/`-march` flags were found in project-owned source. Windows branches exist in build, file/network and application code; isolate actual compilation failures rather than introducing a CPU abstraction. | 2 |
| CPU architecture dependency | Native build uses CMake >=3.28, C++20, pthreads, pkg-config FFmpeg >=60/60/58/7 and libcurl >=7.85. vcpkg is optional; Docker uses CUDA 13.1.2 Ubuntu 24.04 images and is not the Jetson reference path. CUDA Driver linkage is in the excluded SM120 TMA archive; ordinary core uses Runtime and NVTX. Native dependency availability still needs qualification. | 2 |
| GPU ISA/capability dependency | Top-level architecture validation and source exclusion in `src/CMakeLists.txt` accept only 86/89. `NINFER_SM8X_COMPAT` controls source stubs, ordinary stream launches in `src/core/pdl.cuh`, A16 policies, KV rejection, W8 routes, tests and workspace allowances. It mixes capability restrictions and tuning; adding 87 requires reviewing both. | 3 |
| GPU ISA/capability dependency | `src/ops/common/mma.cuh` has BF16/FP16 m16n8k16, INT8 m16n8k32, TF32 m16n8k8 and shared-memory `ldmatrix`; `memory.cuh` has `cp.async`. These are initial SM87 qualification candidates, not yet qualified routes. Blackwell FP8/FP4 MMA, FP4 conversion PTX, TMA and register-transfer kernels must remain unreachable. | 1, 3 |
| GPU ISA/capability dependency | The project-owned `__CUDA_ARCH__` branch in GDN `common.cuh` distinguishes host division from device fast division, rather than selecting a GPU generation. Launch and numerical qualification remain necessary. | 3 |
| Performance-tuning dependency | GDN `bf16_gdn_gating_proj_plan.cpp` uses runtime SM count but fixed measured CTAs/SM, a minimum/fallback of 82 SMs, 100 KiB shared-memory assumptions and register-derived occupancy. Cooperative grid residency is correctness-critical, not merely tuning. Requalify occupancy and fallback behavior for Orin before launching these routes. | 3 |
| Performance-tuning dependency | Linear configuration/dispatch, attention tiles, GDN routes and many `__launch_bounds__` declarations encode tile dimensions, warp counts, register pressure, pipeline stages and shared-memory residency. Cooperative/global synchronization schedules need residency review; retain existing schedules until correctness or measurements justify changes. | 3, 5 |
| Performance-tuning dependency | `src/runtime/engine/context_cost_defaults.cpp` supplies transfer cost estimates; target workspace allowances and serving ECC/bandwidth guidance contain RTX-derived values. They are not Orin performance evidence. Retune only at the appropriate measurement phase. | 4–7 |
| GPU ISA/capability dependency | `src/core/decode_graph.cpp` owns capture, instantiate, update, upload and replay; family runtime owns graph scheduling. Verify graph execution and stable addresses on the actual driver. No graph-specific workaround is established by this inventory. | 1, 3 |
| Performance-tuning dependency | `src/core/arena.cu` uses explicit `cudaMalloc`, pinned `cudaMallocHost` and copies; artifact materialization stages pinned H2D transfers. Device properties and `cudaMemGetInfo` supply capacity information. No managed-memory replacement is warranted. Physical LPDDR sharing changes capacity competition and transfer economics, not ownership or synchronization requirements. | 4, 6, 7 |

### Local evidence and outstanding gates

Host: Linux aarch64, GCC 11.4, CMake 4.0.2, CUDA 12.6.77 selected explicitly
from `/usr/local/cuda-12.6/bin/nvcc`. The default CUDA symlink was not used to
choose the compatibility toolchain.

Configuration attempted with applications, tests and benchmarks enabled:

```bash
cmake -S . -B build/port-cuda126-sm86 -G Ninja \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
```

C/C++/CUDA detection succeeded; configuration stopped at the existing 12.8
floor. A direct `nvcc -std=c++20 -arch=sm_86 -DNINFER_SM8X_COMPAT=1
-Iinclude -Isrc -Ithird_party -c src/ops/linear/nvfp4/nvfp4_gemv.cu`
probe independently failed on missing `cuda_fp4.h`. No full build or tests ran.

Installed FFmpeg pkg-config versions are 58.76.100 / 58.134.100 / 56.70.100 /
5.9.100, below the required versions. `libcurl.pc` is absent. These are additional
full-build prerequisites, not grounds to silently lower dependency requirements.
`nvidia-smi` failed with NvRm initialization/driver access errors; runtime validation
is pending. No SM86 machine or real-model runtime baseline was established.

Next: implement and exactly qualify the E2M1 decode backport, resolve native build
prerequisites, then compile the complete relevant SM86 project using CUDA 12.6
with `cmake --build build/port-cuda126-sm86 -j`. Lower the advertised CUDA floor
only after compatibility evidence exists. Keep SM87 changes and memory experiments
out of that phase.

## Execution status: completed SM87 experiment campaign (2026-09-23)

The historical inventory and phased criteria above remain the porting rationale.
The SM87 experiment campaign is complete and its result ledger is
[`NINFER_JETSON_ORIN_PORT_EXPERIMENTS.md`](NINFER_JETSON_ORIN_PORT_EXPERIMENTS.md).

- Fixed `MAXN`/`jetson_clocks` matrices establish short-workload INT8 draft-3
  at 235.32 PP tok/s and 11.00 TG tok/s (`pp512+tg64`), and long-workload BF16
  draft-4 at 238.25 PP tok/s and 17.86 TG tok/s (`pp2048+tg128`).
- Trace-directed tuning selected the Q4/Q5 attention-input R64C128S2 route for
  T>=21. Its 13.570 ms T=1024 public-op median is 37.3% below the original
  R32C64S4 control, and the numerical gate passes.
- Both BF16 and INT8 KV complete real 32,768-token prefill gates. Guarded
  higher-context and long INT8 extension attempts are pressure-limited during
  setup at the retained 2 GiB `MemAvailable` safety floor, not inference
  failures or capacity classifications.
- Explicit `cudaMalloc` remains the device-allocation control. A memory-pool
  experiment is not admitted without an Engine-owned allocation-class design
  that preserves CUDA Graph address stability and has a meaningful end-to-end
  measurement route.
