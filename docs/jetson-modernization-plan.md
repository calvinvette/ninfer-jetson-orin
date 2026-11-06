# Jetson Orin modernization plan

Date: 2026-11-06. Status: implementation in progress; milestone gates remain open.

This is the temporary execution plan for moving the Orin port onto modern NInfer while
retaining JetPack 6.x / CUDA 12.6 support. It supersedes the instruction in the earlier
[port plan](../NINFER_JETSON_ORIN_PORT_PLAN.md) to defer modernization and Flash-Next work
until a JetPack upgrade. Integrate completed requirements into the active references and
remove this plan when the work is complete.

## Decision and scope

Use current `Neroued/ninfer` as the architectural base. Bring across the required Ampere
implementations from `Don-Chad/ninfer-3090` and the existing Orin port, adapting them to
current ownership and interfaces. Preserve the original engine's V3 container, model
definitions, binding semantics, and public Engine route.

CUDA 12.6 is a required build and qualification target throughout the work. Add JP7.x /
CUDA 13 qualification later using the same SM87 backend. This plan explicitly extends the
existing port's supported execution model to the modern engine and, in a later milestone,
to disk-backed Qwen3.8-Flash-Next expert execution on one Orin GPU.

The first usable deliverable is the modern engine running the two existing benchmark
models on JP6. The subsequent deliverable is the supplied native V3 Flash-Next artifact
running with bounded memory. A successful reader test or build does not establish either
deliverable's model correctness or performance.

## Integration base and working method

The inspected original checkout is `~/workspace/ninfer`, at `a754fcf3` when this plan was
written. The Don-Chad snapshot is available in `~/workspace/ninfer-3090-sync`, at
`115dee6b4`. The existing Orin implementation remains the control for numerical and
performance comparisons. These are starting points, not permanent version pins.

The Orin implementation delta from the shared base touches 11 source/build files; much of
the remaining branch history is qualification documentation. Reimplement the relevant
changes against current upstream interfaces rather than replaying every historical commit.
Ordinary merge conflicts are integration work, not evidence of a platform blocker.

Create the implementation branch/worktree from the selected original upstream base and
carry this plan into it. The branch initially publishing this plan still contains the old
engine; publishing the plan does not perform the engine migration. Keep the existing
checkout and binaries usable as controls until the replacement qualifies. At cutover,
remove superseded project-owned execution paths and update affected active authorities.

Use the 3090 fork as a source of qualified Ampere implementations and as an upstream
contribution destination. Its Flash-Next target currently binds a GGUF/GGML representation;
the supplied artifact follows the original engine's NVIDIA recipe. Reusing the matching
original runtime avoids translating those different tensor conventions inside a reader.

## Platform and toolkit boundaries

Add an explicit `jetson_orin` execution target selecting Linux, AArch64, and SM87 alongside
the original engine's existing platform targets. Keep GPU capabilities separate from
toolkit/API availability. CUDA 13 does not give SM87 native Blackwell FP4 instructions.

Required compatibility work includes:

- Exact E2M1 decoding without `cuda_fp4.h`, using the existing port's implementation as a
  candidate and preserving signed zero, stored scales, and divisor semantics.
- Ordered asynchronous-copy fallbacks for toolkits without the batched-copy API, preserving
  stream ordering, source lifetime, completion, and error behavior.
- Compilation boundaries that exclude unsupported GPU instructions and optional x86 code
  from the Orin build while retaining valid implementations for other targets.
- Cooperative-launch residency checks and appropriate SM87 route selection. Existing tile
  tuning is evidence to revisit at the current shapes, not a universal route table.
- Native AArch64 dependencies and explicit allocation/accounting suitable for shared
  physical LPDDR. Mapped memory or Unified Memory requires its own measured justification.

Build the JP6 target with the selected CUDA 12.6 compiler and use unrestricted repository
build parallelism: `cmake --build <build-dir> -j`. Use a confirmed Python 3.11 interpreter
for converter and verification tools. Keep separate build directories for toolchains.

## Artifact representation and numerical execution

Retain the native V3 artifact and packed NVFP4 experts. The supplied artifact records the
`qwen3_8_flash_next` recipe: Q8 main projections, NVIDIA NVFP4 main experts, block-FP8 MTP
experts, NF4 PLE, and BF16/FP32 auxiliary tensors. The original upstream has this recipe and
its Qwen4Exp runtime; exact binding compatibility must still be verified on the chosen base.

Implement qualified SM87 operations that decode packed tiles into BF16/FP32 operands as
needed during computation. The generic reader owns framing and descriptors; it does not
silently requantize weights or select model arithmetic. Full-model expansion at loading
time is not the selected design.

For each changed numerical operation, use an independent FP32/FP64 mathematical oracle
from represented public inputs, decoding exact stored codes, scales, and divisors. Preserve
explicit casts, persistent state, and observable output boundaries. Compare floating-point
outputs using justified criteria at real shapes; compare codecs and exact transforms
exactly. Blackwell A4 and Orin A16 implementations need not produce bit-identical results.

Qualify the block-FP8 MTP path separately, including acceptance/commit and state restoration.
NF4 PLE lookup must preserve its specified BF16 output boundary. Final generated text alone
is insufficient evidence for any of these operations.

## Flash-Next capacity and storage

The inspected host exposes about 29 GiB of physical RAM. Directory metadata from
`~/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` reports approximately:

| Stored data | Size |
|---|---:|
| Main NVFP4 expert weights | 63.3 GiB |
| NF4 PLE table | 27.1 GiB |

These figures exclude the other model objects and runtime allocations. FP8/INT8 expert
codes alone would occupy about 112.5 GiB before scales; BF16 experts would occupy about
225 GiB. Re-encoding the entire bank into a wider format increases the capacity problem.

The original upstream already provides disk-backed PLE lookup. Its documented expert
offload path retains a complete host expert backing and uses x86 CPU kernels for part of
the computation. Neither assumption provides a working route on this host.

Add disk-backed expert storage with bounded caches and staging. Plan CPU and GPU
allocations together against one physical RAM budget, including pinned buffers, runtime
state, workspaces, and filesystem cache pressure. Moving an allocation from device to host
does not create additional physical capacity on Orin.

The storage layer owns reads and encoded buffers; model scheduling owns required expert
IDs and readiness; Ops own arithmetic. Publish cache entries only after their contents are
ready. Reuse buffers only after all GPU consumers complete, including error and cancellation
paths. CUDA Graph consumers require stable addresses or an explicit supported update
mechanism. Prefill and decode must both obey the bounded-memory contract.

Measure storage misses, bytes read, cache behavior, and inference throughput on the actual
drive. No throughput target is established by the current source inspection. Do not assume
that a discrete-GPU host/device cache policy is efficient on shared LPDDR.

## Delivery sequence and acceptance gates

## Current execution status (2026-11-06)

The first JP6 implementation pass is on local branch `feat/jetson-orin-sm87-jp6` in
`~/workspace/ninfer`, based on the planned `a754fcf3` source revision. Commits
`a8f27562` and `6376058d` add CUDA 12.6/SM87 build compatibility, explicit unsupported
routes, target runtime admission, and a predicated BF16 fallback for row/K tails. The
full build succeeds on Jetson AGX Orin (aarch64, SM87, CUDA 12.6). The push to the
configured `Neroued/ninfer` remote was rejected because the authenticated account has
read-only access; the commits remain local.

Focused artifact reader, backing, and materialization tests pass serially. The BF16
projection oracle passes 197 cases for the previously failing `324x10240` shape; the
`14336x5120`, T=129 route passes its two selected cases. The broader CTest run was
started concurrently and is not a valid suite result: GPU resource/capture contention
and target-independent tests that exercise deliberately unavailable compressed KV and
block-scaled MoE routes caused failures. The Qwen3.5 runtime-mechanisms test also still
needs target-specific expectations for FP8/NVFP4/K8V4 KV. Missing Python packages and
the local FFmpeg runtime path affected unrelated tests.

The public benchmark now completes short autoregressive runs on both original artifacts
with CUDA Graphs and INT8-G64 KV:

| Artifact | Workload | Result |
|---|---|---|
| Qwen3.8-27B groupwise-int | 57 prompt tokens, 4 output tokens; context cap 1 GiB, max context 1024; no warmup, 1 repetition | 58.9 prefill tok/s; 3 decode steps in 137.1 ms |
| Qwen3.6-35B-A3B groupwise-int | 15 prompt tokens, 6 output tokens; context cap 1 GiB, max context 1024; MTP-2, 1 warmup, 1 repetition | 85.5 prefill tok/s; 2 decode steps in 49.7 ms; 3/3 drafts accepted |

These are bring-up checks, not matched replacements for the existing `pp512+tg64` and
`pp2048+tg128` campaigns. The host currently exposes about 29 GiB physical RAM, so a
12 GiB context reservation does not fit beside the 27B weights and required runtime
resources; the smoke runs use a deliberately small 1 GiB cap. The plan's matched
benchmark gate remains open.

The generic V3 reader parses the supplied Flash-Next artifact (1,319 objects and
108,440,379,396 payload bytes). This proves container parsing only. The current resident
materializer cannot fit its NVFP4 expert bank and NF4 PLE table in this host's physical
memory, and no bounded disk-backed expert execution path has been implemented. There is
no Flash-Next inference or performance result yet.

### 1. Modern JP6 platform and original models

Bring up CUDA 12.6/SM87 builds of the modern engine, applications, and relevant tests.
Port the necessary Ampere operators and existing Orin fixes. Restore Qwen3.8-27B
groupwise-int and Qwen3.6-35B-A3B groupwise-int through the public Engine.

Use the official V2-to-V3 migration on the pinned baseline artifacts, preserving represented
weights and resources. Keep the original files available for the old engine. Do not substitute
new publisher artifacts with potentially different quantization in a performance comparison.
Do not add a legacy container execution lane to the modern engine.

Gate: successful JP6 build, exact migration checks, relevant numerical qualification, and
real-model inference for both artifacts. Verify supported product paths affected by the
migration, including serving, prefix reuse, and speculation where applicable.

### 2. Qualify the replacement port

Run focused operator tests, real Engine integration tests, and protocol/report tests for
changed contracts. Reproduce representative existing short and long benchmark scenarios
with matching artifacts, corpus, KV format, MTP window, context reservation, graph mode,
prefix policy, warmups, and repetitions.

Use MAXN and `jetson_clocks` for controlled measurements; passwordless sudo is authorized.
Record actual CPU/GPU/EMC clocks and relevant thermal/power conditions. Investigate material
differences before claiming preservation or improvement. Operator timings support operator
claims; end-to-end claims require inference measurements.

Gate: no known in-scope correctness regression, explained performance results, and clear
limitations for any supported path that could not be exercised. Only then select the modern
engine as the replacement baseline.

### 3. Native Flash-Next execution on SM87

Verify the supplied V3 directory and bindings against the selected original runtime.
Qualify NVFP4 expert operations, indexed attention, HC, GDN, and PLE at the actual shapes.
Start with ordinary text execution. Small operator/layer fixtures can establish arithmetic
before the full artifact fits the storage execution path.

Gate: direct mathematical-oracle evidence for changed operations and correct model/state
composition. Whole-artifact execution on this host also depends on milestone 4.

### 4. Bounded expert streaming and real-artifact qualification

Implement the storage/cache design above. Exercise miss, hit, eviction, prefetch, and consumer
lifetime transitions, then run the supplied artifact through the public Engine on this host.
Measure cold and warm behavior separately and account for disk traffic and physical memory.

After ordinary execution qualifies, enable and verify MTP, then qualify additional supported
features such as Vision and prefix reuse before advertising them. Gate: correct real-artifact
execution within the explicit physical-memory budget and usable benchmark results with
storage conditions stated. Do not count parsing or model initialization as completion.

### 5. JP7/CUDA13 qualification

When the upgraded environment is available, build and qualify the same SM87 backend with
the selected JP7/CUDA13 toolchain. First repeat workloads at the same clocks as JP6; then
measure the highest stable supported clock profile separately. Keep toolchain and clock
effects distinct. CUDA 12.6 remains a required regression target.

## Upstream contributions and completion

Prepare separately reviewable changes for toolkit compatibility, Ampere operators, the
Orin platform, and expert streaming. Preserve attribution for imported implementations.
Contribute Ampere work to `ninfer-3090` and submit shared changes directly to original NInfer
where appropriate. Do not make acceptance by one fork a technical dependency of another.
Repository pushes and PR scope follow the user's authorization for the relevant work.

Update active architecture, build, artifact, and product documentation with each completed
contract change. Keep this plan as the single temporary coordination document; remove it
when requirements have been integrated and the requested milestones are complete. Do not
claim upstream acceptance, JP7 qualification, or Flash-Next support before their gates pass.
