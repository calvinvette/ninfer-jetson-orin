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

## Current execution status (2026-11-07)

### Milestone 1: Modern JP6 platform and original models

The modern Orin source branch `feat/jetson-orin-sm87-jp6-sync` is based on the original NInfer
`dev` revision `62674bae`, rebased from `a754fcf3`. The full applications and benchmarks build on
Jetson AGX Orin (AArch64, SM87, JetPack 6.x / CUDA 12.6). CUDA 12.6 remains the active target;
JP7/CUDA 13 is not available for qualification yet.

Both original groupwise-int models complete public Engine inference with the modern V3 runtime.
The matched 512/2,048-token prompt and 64/128-token generation matrix is recorded below; it shows
small prefill gains and small decode regressions, not a general performance win. The short bring-up
runs and approximate context-allocation differences are retained in the benchmark reports and
summarized in the earlier milestone record where needed for comparison.

### Milestone 2: Replacement port qualification

The full JP6 build succeeds. Focused Q8 projection, block-scaled MoE, runtime admission, artifact
reader/backing/materialization, serving schema, and public Engine checks passed on their supported
Orin routes. The compressed FP8, NVFP4, and K8V4 KV-cache consumers remain unavailable on Orin;
the target rejects them explicitly. The supported paged KV path is BF16 and INT8-G64.

The real-artifact QSA planner had a serial-versus-MTP discrepancy at the 64/65-token boundary:
serial decode selected short-context SIMT, while a verify window containing the next position
selected MMA. The Orin build now retains MMA for short contexts; the RTX 5090 route is unchanged.
The exact T=1 versus T=2 QSA regression passes for BF16 and INT8-G64, as do oracle cases at
histories 64 and 65. A matched warm-cache public-Engine benchmark on the supplied artifact showed
about 1% lower prefill/decode rates under this route policy, within three-run variation; it does
not demonstrate a material performance change.

### Milestone 3: Native Flash-Next execution on SM87

The explicit artifact `~/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer` parses 1,319
objects and 108,440,379,396 payload bytes. The real-artifact loader binds 48 main expert banks,
the optional MTP bank, native geometry, and disk-backed PLE. Resident materialization would require
75.37 GB without MTP and 78.06 GB with MTP, exceeding the host's approximately 29 GiB physical
RAM. The selected execution route keeps packed NVFP4 experts on disk and stages bounded cache
entries; it does not expand the model to FP8/BF16.

Representative changed arithmetic has independent oracle evidence: the real artifact's first
NVFP4 MoE layer passes FP64 comparison (relative L2 3.32e-7, maximum absolute error 3.73e-9),
QSA input projections pass the artifact-bound A16 oracle, and the PLE gate/convolution, HC, and
GDN state checks pass their focused fixtures. These checks qualify selected real shapes and
operators, not all 48 layers or complete-model accumulated error.

### Milestone 4: Bounded expert streaming and real-artifact execution

The public Engine completes text inference on the supplied V3 artifact with a 5.27 GiB bounded
expert cache and a 512 MiB Device context. The first-layer file-backed NVFP4 cache test covers
miss, hit, eviction, and refill, and compares one real layer against an independent FP64 oracle.
A public Engine cache-cold/cache-warm pair on `Hello` staged 23.81 GB then 21.90 GB of expert
payload for repeated 53-token prompts. Prefill rose from 2.46 to 3.40 tok/s; the second request
still loaded substantial expert data. These are ExpertCache-cold/warm measurements, not cold
physical-drive measurements. An instrumented repeat sampled `/sys/block/nvme0n1/stat` after Engine
startup and at each repetition boundary: repetition 1 recorded 23.89 GB read and 3.58 MB written;
repetition 2 recorded 21.97 GB read and 0.39 MB written (512-byte sector counters). These
per-request block-layer totals closely track the corresponding expert-payload counts, with any
concurrent device activity included. `O_DIRECT` bypasses the page cache but does not control
SSD/controller caches. The full benchmark context and timings are recorded below.

MTP greedy equivalence now passes against serial decode for the real artifact on `hi` and `Hello`
through 128 output tokens, with draft windows 1 and 4 tested at shorter horizons and MTP-4 at 128.
The default 64-token `hi` case exercises one fallback; the 128-token `hi` case completes 123 drafts
and accepts 96 without fallback. The two-request MTP cohort also matches non-speculative token IDs
and exercises a two-row decode batch. An additional 32-token case on “A small robot explores a
forest at sunrise” matched exactly with MTP-4 (38 drafted, 21 accepted, no fallback). This resolves
the previously reproduced token-13 and token-32 mismatches for the tested cases. A separate
64-token arithmetic explanation prompt matched with MTP-2 (81 drafted, 21 accepted, one fallback).
The `hi` prompt also matched through 256 output tokens with MTP-4 (236 drafted, 195 accepted,
no fallback). Other prompts and horizons above 256 remain unqualified. A follow-up on a new prompt
with a five-token draft window did not complete: 512-token and then 128-token serial/MTP comparisons
remained in direct artifact reads and were stopped before producing token results. During both
attempts the kernel logged repeated NVMe I/O timeouts on `nvme0n1`; SMART reported 0 media errors,
51 C, and 90 error-log entries. These runs are inconclusive, not parity failures. Further long-
horizon artifact tests need a stable storage run; the previous parity evidence remains valid for
its recorded prompts and horizons. A new Graph-enabled serial/MTP comparison on `hi` at 64 output
tokens with a four-token draft window stalled in direct-I/O wait before producing token output;
the worker was interrupted after about 100 seconds. A shorter Graph-enabled `hi` run completed
with eight output tokens and draft window four: serial and MTP token IDs matched exactly, with five
drafts, four accepted tokens, and one fallback. A second Graph-enabled case on “A small robot
explores a forest at sunrise” matched through 32 output tokens with MTP-4 (40 drafted, 20 accepted,
one fallback). Graph-enabled MTP-4 also matched serial output for `hi` through 64 tokens (58
drafted, 47 accepted, one fallback) and 128 tokens (123 drafted, 96 accepted, no fallback). A
Graph-enabled two-request, two-row MTP cohort matched serial generation exactly; the requests
reported 9/4 and 5/5 drafted/accepted tokens, and the test observed a two-row decode step. The
cohort test now exercises three concurrent requests: all three matched serial token IDs,
observed a three-row decode batch, and reported 9/4, 5/5, and 5/5 drafted/accepted tokens. To make this
supported batch shape work on Orin, the Qwen4Exp MoE shape policy now slices ordinary decode input
at the qualified per-format kernel width (two rows for block-FP8 and one for NVFP4), as it already
did for masked MTP subsets. The Graph-enabled real-artifact CTest passed, including the 64-token
text parity case, in 427.8 s; NVMe timeouts caused long waits. An eight-request attempt did not qualify:
the 10-minute CTest run timed out amid further NVMe I/O timeouts. The real-artifact test runner
accepts `NINFER_TEST_CUDA_GRAPH=1` to run these parity cases with the production Graph route. A prior
64-token attempt was inconclusive after direct-I/O stalled; a subsequent rerun completed and
supersedes that result.

A follow-up four-request, 16-token cohort exposed a deterministic parity failure at generated token
14 for the second prompt: ordinary cohort decode produced 4145 while MTP produced 2716. It
reproduced with Graphs both enabled and disabled, and with draft windows two and four. A three-row,
16-token run reproduced the same mismatch at the same prompt/token. The same prompt passed a
single-row serial/MTP comparison through 16 tokens (draft window two; 11 drafted, 9 accepted, no
fallback); both single-row outputs produced 2716, while the ordinary three- and four-row cohorts
produced 4145. This narrows the discrepancy to batched execution but does not distinguish a
batch-shape numerical change from a state/dispatch defect. Cohort parity is established only through
three rows at the previously tested eight-token horizon; longer cohort decoding remains unqualified.

A real-artifact image-conditioned MTP comparison also passed on the public Engine with Graphs
enabled: one resized 224x224 image, 83 prompt tokens, greedy output horizon 8, and draft window 4.
Serial and MTP generated token IDs matched exactly; MTP drafted 8 tokens, accepted 3, and took one
fallback across three rounds. The Vision stream was 35.4 ms. This qualifies one image-conditioned
accept/commit path. A video-only comparison matched serial token IDs for four greedy output
tokens on a 49-token prompt with Graphs enabled and MTP-4; MTP accepted two drafts in one round
without fallback, and the Vision stream was 16.2 ms. The same video prompt also matched through 32
greedy output tokens with Graphs and MTP-4 enabled: MTP drafted 41 tokens, accepted 20 across 11
rounds, and had no fallback. The serial run took 39.3 s and the MTP run 70.4 s, so this establishes
state correctness at a longer media horizon, not a speedup. A follow-up with a 64-token cap ended
naturally on the model stop token after 35 output tokens; serial and MTP IDs matched exactly. MTP
drafted 44 tokens, accepted 23 over 11 rounds, and had no fallback. That run took 41.6 s serial and
40.3 s with MTP, which is only one diagnostic sample. Longer Vision/MTP horizons and cohorts larger
than three remain open.

The public Engine completes a single-image request, a mixed image/video fixture (1,615 prompt
tokens, output `NIFER-9`), text prefix reuse, and repeated image prefix reuse. The mixed-media
request took 413.1 s and exercised six Vision stream intervals totaling 1.20 s. Five focused CUDA
oracle suites pass on Orin for Vision constituent Ops, including packed BF16 attention at
head-dimension 72 / 16 heads, 2D RoPE with the same geometry, 1152-wide position interpolation,
1152-wide layer normalization, and GELU. A composed encoder test passes an independent FP64
reference through all 27 Vision layers at artifact geometry (hidden 1152, intermediate 4304, patch
width 1536, merger width 4608, output width 2560), with two separate four-patch image segments.
With one deterministic synthetic BF16 block repeated across layers, relative L2 is 0.00963 and
maximum absolute error is 0.02344. The same oracle also loads the artifact's distinct learned BF16
Vision weights and passes the full 27-layer encoder for one four-patch feature: relative L2 is
0.03122 and maximum absolute error is 0.00586. The test compares the final represented BF16 output
with the independent FP64 computation and uses relative-L2 <= 0.04 and max-absolute <= 0.01
acceptance criteria. An isolated production `VisionContext` benchmark now measures the encoder
with the artifact's learned weights (synthetic deterministic patch tensors and position metadata),
excluding image decode and preprocessing. On AGX Orin in MAXN with `jetson_clocks` (CPU 2.2016 GHz,
GPU 1.3005 GHz, EMC 3.199 GHz; CUDA runtime 12.6.0), two warmups and five repetitions gave median
CUDA-stream / synchronized wall times of 8.615 / 8.621 ms for 1 feature (4 patches), 10.067 /
10.076 ms for 4 features (16 patches), and 12.859 / 12.865 ms for 16 features (64 patches). The
corresponding wall-time throughput is 464, 1,588, and 4,975 patches/s. These are encoder-only
measurements; end-to-end mixed-media latency is still not isolated from Text and expert-storage
work. Command:

```bash
./build/jetson-orin-jp6-rebase/bench/ninfer_qwen4_exp_vision_bench \
  /home/calvin/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer 2 5
```

The public Engine completed a mixed image/video smoke input with a resized 224x224 image and a
two-second 84x56 video fixture. A one-token run took the eager Prefill route (Graphs enabled but
not used for Prefill), producing 156 prompt tokens in 36.9 s with a 48.2 ms Vision stream interval.
A separate one-repetition `ninfer_bench` run requested two output tokens, so it also exercised one
file-backed Decode Graph step (reported as `compute_paths.mixed=1`; the file-backed host handoff
keeps it out of the resident-only `graph` count). Total request wall time was 67.67 s: 67.09 s
through the first token and 0.589 s for the decode step. The report attributed 49.1 ms to the
Vision stream, 61.89 s to cumulative expert-cache planning (including synchronous direct range
reads into staging), 0.246 s to cache submission, and 43.22 GB to expert uploads. CUDA 12.6 on AGX
Orin with `jetson_clocks` was enabled (CPU 2.201 GHz, GPU 1.298 GHz, EMC 3.199 GHz). This separates
public mixed-media latency from the cache's synchronous range-read work for one run; repeated NVMe
timeouts and single-run variability mean these values are diagnostic, not a stable benchmark.

A focused CUDA cache test captures a consumer against the stable expert-cache view, stages two
different file-backed expert IDs into the cache, and verifies graph replay reads each updated expert
exactly. The file-backed schedule now captures routing separately from expert compute: the route
Graph completes before route IDs are copied to the host and experts are staged, then the compute
Graph launches after cache readiness. CUDA Graphs are enabled for this Engine route. On the real
Qwen3.8-Flash-Next artifact, a one-layer captured file-backed NVFP4 MoE run passed the independent
decoded-weight FP64 oracle (relative L2 5.38e-6, maximum absolute error 5.96e-8) while changing the
routed expert set. A public Engine smoke run also completed Graph calibration (5.7 s), runtime
initialization, and one-token generation with Graphs enabled: 53 prompt tokens, 16.1 s prefill,
3.3 tokens/s. This exercises the full Engine path but is a smoke result, not a latency benchmark.
Both runs used the local V3 artifact, CUDA 12.6, and AGX Orin; `jetson_clocks` was enabled for the
schedule/oracle run.

### Remaining gates

- Resolve the reproducible long-horizon cohort parity mismatch before qualifying longer cohort
  decoding. Text Graph/MTP has exact single-request serial parity through 128 output tokens and
  exact cohort parity through three requests at an eight-token horizon. At 16 tokens, three- and
  four-request cohorts diverged on row two at output token 14 with or without Graphs; that prompt
  passed single-row serial/MTP comparison, whose output at that position matched cohort MTP. This
  suggests a batched execution difference but does not identify its source. Continue longer
  Vision/video horizon qualification after resolving this issue. Image-conditioned runs cover 8 and 32 tokens, and
  video-conditioned runs cover 4, 32, and 35 generated tokens (the last run ended on the stop token
  with a 64-token cap).
- Repeat the public mixed-media/ExpertCache breakdown under stable storage conditions. The current
  mixed-media Graph report is one diagnostic sample with NVMe timeouts; Vision numerical and
  encoder-throughput qualification covers artifact weights at up to 64 patches.
- Distinguish SSD/controller cache state for physical cold/warm storage measurements; `O_DIRECT`
  isolates requests from the host page cache but does not flush or measure device caches.
- Storage diagnosis: the 4 TB Seagate FireCuda 530 (`SU6SM001`) reports no critical warning,
  100% spare, 0 media errors, and 44 C after the latest run. PCIe is 16 GT/s ×4 with ASPM disabled.
  A temporary APST-disable run still logged NVMe I/O timeouts; APST and its original PS3/PS4
  transition table have been restored. In that run the 64-token text parity case passed, then one
  three-request cohort submission expired while waiting for admission after an NVMe timeout. APST
  alone therefore does not explain the storage stalls, and the cohort result remains qualified by
  its earlier successful run rather than this repeat. The timeout source is still unidentified.
- Graph execution beyond the one-token smoke is now verified with text MTP through 128 tokens,
  image MTP through 32 tokens, video MTP through 35 generated tokens, a three-request MTP cohort,
  and a file-backed mixed-media Decode step. These are correctness/smoke checks; stable end-to-end
  Graph performance remains unmeasured under the current storage conditions.
- Re-run CUDA 12.6 regression after changes and qualify JP7/CUDA 13 when that environment becomes
  available.
- Submit source changes upstream when a repository with push access is available. The local source
  branch is `feat/jetson-orin-sm87-jp6-sync`; authenticated GitHub permissions report push access
  disabled for both `Neroued/ninfer` and `Don-Chad/ninfer-3090`. The documentation-plan branch is
  pushed separately.

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

Matched CUDA 12.6/SM87 and modern-engine measurements were run on the same Orin in MAXN with
`jetson_clocks` enabled (CPU 2.2016 GHz, GPU 1.3005 GHz, EMC 3.199 GHz). The comparison used
the original Qwen3.8-27B and Qwen3.6-35B-A3B V2 groupwise-int artifacts for the SM87 control and
exactly migrated V3 copies for the modern engine. Both consumed exact 512-token and 2,048-token
prefixes of `bench/fixtures/bench_corpus.ids`, greedy generation, INT8 group-64 KV, 1,024-token
prefill chunks, no prefix reuse, CUDA Graphs, MTP off, one warmup, and three repetitions. All
repetitions processed the full prompt and committed 64 or 128 decode tokens (65 or 129 outputs
including the first prefill token). `tegrastats` sampled the runs; GPU temperature peaked at
69.5 C.

| Target/workload | SM87 V2 prefill/decode tok/s | Modern V3 prefill/decode tok/s | Change |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B, pp512+tg64 | 244.03 / 7.672 | 249.00 / 7.299 | +2.0% / -4.9% |
| Qwen3.8-27B, pp2048+tg128 | 244.71 / 7.606 | 249.39 / 7.270 | +1.9% / -4.4% |
| Qwen3.6-35B-A3B, pp512+tg64 | 1,185.82 / 37.420 | 1,222.16 / 36.740 | +3.1% / -1.8% |
| Qwen3.6-35B-A3B, pp2048+tg128 | 1,358.23 / 36.826 | 1,431.20 / 36.404 | +5.4% / -1.2% |

The benchmark driver now accepts `--token-ids FILE`, which prepares the supplied IDs through
the public Engine and preserves the old benchmark's raw-token behavior (no model default stop
sequences and raw output). This allowed the two engines to consume identical prompts. Its parser
test passes and the complete modern JP6 build succeeds. Context allocation cannot be made
identical byte-for-byte: the modern Qwen3.8 runtime requires a 308.3 MB context pool (plus a
103.9 MB workspace), while the old engine reserves 235.6/304.9 MB sequence storage plus a
160.0 MB workspace. For Qwen3.6-35B-A3B, the modern pool was reduced to 130.0 MB, just above
its 128.8 MB minimum, plus a 72.1 MB workspace; old sequence plus workspace reservations were
201.7/223.3 MB. These runs are closely matched in workload and approximate total runtime
allocation, not identical in internal reservation policy. The pressure guard did not abort any
run. Results indicate small prefill gains and small decode regressions; they do not establish a
general performance win. Flash-Next and full feature qualification remain separate open gates.

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
