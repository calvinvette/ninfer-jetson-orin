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
`a8f27562`, `6376058d`, `b72ba228`, `3209e283`, `0962c076`, and `f7ad8ac6` add CUDA
12.6/SM87 build compatibility, explicit unsupported routes, target
runtime admission, a predicated BF16 fallback for row/K tails, Orin-specific expectations for
compressed KV formats without qualified consumers, and decode-only NVFP4/block-FP8 vector
routes. The
full build succeeds on Jetson AGX Orin (aarch64, SM87, CUDA 12.6). The push to the
configured `Neroued/ninfer` remote was rejected because the authenticated account has
read-only access. A push to `Don-Chad/ninfer-3090` was also rejected (HTTP 403 for the
authenticated `calvinvette` account); the implementation commits remain local.

The NVFP4 route reuses the exact E2M1/E4M3 unpack and FP32 vector contractions without
instantiating Blackwell MXFP4 MMA. The block-FP8 route decodes its stored E4M3 codes and BF16
scales directly and uses FP32 vector contractions. Their complete-MoE CUDA Graph oracle cases
pass on the Orin for NVFP4 T=1 and block-FP8 T=1/2, including routed subsets and both output
epilogues. The block-scaled CTest now runs those supported cases on Orin; the artifact reader,
backing, materialization, and runtime-mechanisms focused tests also pass.

Focused artifact reader, backing, and materialization tests pass serially. The BF16
projection oracle passes 197 cases for the previously failing `324x10240` shape; the
`14336x5120`, T=129 route passes its two selected cases. The broader CTest run was
started concurrently and is not a valid suite result: GPU resource/capture contention
and target-independent tests that exercise deliberately unavailable compressed KV and
block-scaled MoE routes caused failures. The Qwen3.5 runtime-mechanisms test now passes
with target-specific expectations for FP8/NVFP4/K8V4 KV rejection on Orin. Missing Python
packages and the local FFmpeg runtime path affected unrelated tests.

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
benchmark gate is closed by the short/long matrix below.

The modern Engine has now also completed three measured repetitions on each pinned original
artifact after the official V2-to-V3 migration. The measured request used one fixed generated
text message, greedy sampling, 64 output-token budget, INT8 KV, a 1 GiB context capacity,
2,048 maximum context, CUDA Graphs enabled, prefix reuse disabled, one warmup, and one active
request. MAXN/`jetson_clocks` readback was CPU 2,201.6 MHz, GPU 1,300.5 MHz, EMC 3,199 MHz
with the override active.

| Pinned model and modern Engine route | Prompt / committed output | Mean prefill / decode |
|---|---:|---:|
| Qwen3.8-27B groupwise-int, MTP off | 707 / 63 tokens | 206.64 / 7.30 tok/s |
| Qwen3.6-35B-A3B groupwise-int, MTP-2 | 665 / 63 tokens | 1,192.76 / 53.60 tok/s; 39/46 draft tokens accepted (84.8%) |

The source V2 files remain intact; migrated benchmark copies were written under `/tmp`. The
benchmark API accepts text/messages rather than token IDs, so these runs use the same message
text but do not reproduce the old curated 512-token corpus. They establish successful
modern-Engine inference and a representative fixed-clock benchmark for both pinned original
models. They do not replace the matched-corpus/short-and-long comparison gate above. Raw JSON
reports are local under `/tmp/ninfer-jp6-qwen38-27b-chunk1024.json` and
`/tmp/ninfer-jp6-qwen36-35b-mtp2.json`.

The generic V3 reader parses the supplied Flash-Next artifact (1,319 objects and
108,440,379,396 payload bytes). That initial check established container parsing only. The
resident materializer cannot fit its NVFP4 expert bank and NF4 PLE table in this host's
physical memory; the bounded disk-backed execution path and its current results are recorded
below.

The Qwen4Exp real-artifact loading test passes against that Flash-Next file without
materializing weights. It verifies the 48 Main expert banks, the optional MTP bank, native
expert geometry, and PLE-backed load planning; it also confirms that the resident artifact
would require 75.37 GB without MTP and 78.06 GB with MTP, beyond this host's physical RAM.
That test covers binding and metadata only. The GPU-only expert schedule and later Engine
execution evidence below supply the separate streamed-weight, text-inference, and short MTP
benchmark checks.

The disk-streaming foundation includes `FileBacking::read_range`, committed in the Orin
source branch as `4106d05a`. It reads object subranges spanning continuation files with
aligned direct I/O and per-fragment host scratch; an exact cross-file subrange check passes
in `ninfer_artifact_backing_test`. Commit `d83f2847` adds metadata-only expert-bank views,
per-slot pinned staging, file-range cache fills, and Qwen4Exp disk-expert load planning.

The GPU-only file-backed scheduler now routes on GPU, stages only selected NVFP4 experts from
the file, keeps fixed hot experts and bounded per-bank slots, and executes the qualified
tokenwise NVFP4 MoE Op. It fits a 5.27 GiB expert cache and plans a 16 GiB device budget
alongside resident weights; the resident model itself is about 70.2 GiB and is not
materialized. On Jetson AGX Orin (SM87, JP6/CUDA 12.6), the focused expert-cache,
artifact-backing, Qwen4Exp-loading, and block-scaled MoE tests pass. The real V3 artifact
loads its metadata and executes one NVFP4 MoE layer for two different hidden vectors; each
repeated route reuses cached experts (55,296,000 bytes uploaded across the two distinct
routes, with no repeat uploads). The changed route also exercises dynamic cache-slot
replacement.

The public Engine also completes Flash-Next runs using the explicit artifact
`~/models/Qwen3.8-Flash-Next-NInfer/qwen3_8_flash_next.ninfer`, prompt `hi`, greedy sampling,
prefill chunk 128, 512 MiB context reservation, and one active request. A ten-token CLI smoke
with MTP off reported 17.0 s prefill (3.1 tok/s), 5.9 s decode (1.5 tok/s), and 22.9 s total.
The rebuilt public-Engine benchmark then completed one warmup and three measured runs each
with MTP off and MTP-1. All runs used a 53-token prompt and committed 9 generated tokens
under the MAXN/`jetson_clocks` profile recorded above.
MTP off averaged 3.46 prefill / 1.60 decode tok/s; MTP-1 averaged 3.52 / 1.83 tok/s. MTP-1
accepted all four drafts in each run, with one fallback step. Startup reported 6.91 GiB GPU
weights without MTP and 9.42 GiB with the resident MTP bank; the bounded expert cache is
5.27 GiB. These very short runs establish that the public benchmark path executes MTP with
the disk-backed main experts, but are not stable throughput claims or full MTP state
qualification. The benchmark process was rebuilt after the first attempt exposed a stale
binary still using resident-only planning.

A deterministic public-Engine check with the same Flash-Next artifact, prompt `hi`, greedy
sampling, seed 0, and a ten-token limit found that MTP-1 changes the emitted sequence. With
CUDA Graphs enabled, MTP off emits
`1596 1144 310 5707 310 1156 328 5834 3158 8722`; MTP-1 emits
`1596 1144 310 5707 310 1156 25 328 5834 3158`. The mismatch begins at token seven,
despite all four drafts being accepted in the earlier short benchmark. This is a correctness
failure for greedy speculative equivalence, so MTP is not qualified for Flash-Next. A second
run with CUDA Graphs disabled emits the same sequence through token seven, across three MTP
rounds with all three drafts accepted. This confirms the mismatch is not caused by graph replay.
A two-token eager request completed but did not enter an MTP round. Do not use the short MTP
throughput result as a supported-path claim until greedy equivalence is resolved.

The captured BF16 output-head logits identify the immediate divergence at the mismatching
position: single-token decode scores token 328 at 23.25 and token 25 at 23.0, while the batched
verify column scores both at 23.25. The greedy argmax's deterministic lower-ID tie break selects
25 in the verify route. This is evidence of a batch-shape-dependent numerical boundary in the
target forward result; it does not by itself distinguish which upstream operation causes that
difference or establish state corruption. Keep the sampler's tie rule unchanged. Qualification
needs a target-forward/state comparison at this boundary and a defined way to preserve greedy
equivalence when verify and single-token paths disagree.

A temporary capture of the main model's post-HC hidden vector for input token 1156 found only
62/2,560 BF16 elements identical between the single-token and two-column verify paths (mean
absolute element difference 0.132; maximum 1.75). This places the discrepancy before the output
head, at or before the final hidden representation. It is a cross-route diagnostic, not an
independent numerical oracle: the two executions also have different recurrent-state histories.
The next correctness investigation must compare the relevant target state and intermediate
boundaries against the model's mathematical semantics before attributing the difference to a
specific kernel or changing MTP scheduling.

A layerwise capture of the same input token found the first post-block difference at layer 0
(mean absolute residual difference 0.0031), growing to 0.0844 by layer 35 and 0.7279 by layer
47; the mean difference increases from 0.0844 at layer 35 to 0.1801 at layer 36. The focused
GDN replay-record and GDN replay-fold tests both pass on SM87, including replay output equality
and accepted-prefix state selection. These synthetic Op checks do not qualify the full Qwen4
execution, but they do not support changing the generic GDN replay/fold implementation as the
first response. The next useful comparison is at the real model's first-layer projection/mixer
boundaries and around layer 36, with the stored operator inputs and state held explicit.

That boundary comparison found bit-identical layer-0 GDN inputs. The FP32 control projection
differs by at most 7.7e-6; projected query/key/value tensors differ slightly, while the replay
recurrent output differs by a mean 7.3e-7. The GDN output projection has a mean absolute
difference of 2.7e-4. At transformer layer 36 (GDN state index 27), its input already differs
by a mean 0.011, while the recurrent output remains within a mean 1.8e-5. This supports an
accumulating cross-shape projection/activation difference rather than a large recurrence replay
error. It still needs independent operator-oracle measurements at the real artifact's shapes;
these temporary captures were removed after analysis.

To test the identified single-token versus multi-column projection split, the 48x2560 GDN
control projection now uses one tiled GEMV route for T=1 and T=2+, and the Q8 2560x6144 output
projection uses one sliced route for T=1 and T=2+. New exact first-column tests pass for both
operators; the Q8 linear oracle and complete GDN control suite also pass on SM87. Repeating the
public ten-token greedy check with the real artifact still produces the original MTP-off sequence
`1596 1144 310 5707 310 1156 328 5834 3158 8722` and MTP-1 sequence
`1596 1144 310 5707 310 1156 25 328 5834 3158`. The route unification therefore does not
resolve the end-to-end mismatch, which still begins at token seven. The corresponding runs took
16.9/5.9 s (prefill/decode) with MTP off and 16.6/4.5 s with MTP-1; MTP-1 accepted four drafts
with one fallback. Do not claim greedy MTP correctness or a throughput win from this short
smoke. The next comparison must isolate later layer, attention, or head boundaries while holding
the exact starting model state and token input fixed.

A temporary commit trace hashed every layer's GDN convolution and recurrent state. At the initial
prefill commit, MTP-off and MTP-1 matched (`4aa95da671a407ce`); after the first shared generated
token 1596 they still matched (`91aa782b97cdc47b`). After MTP accepted tokens 1144 and 310 as a
pair, the committed state hash was `a4b067f6e98b2051`, whereas serial decoding has distinct
per-token states after 1144 (`5e54fccfb5670514`) and 310 (`b60a4505f6d953c1`). Thus, GDN state
already diverges by the first two-token verify commit, before the later token-7 mismatch. This
hash is a diagnostic comparison, not a numerical oracle, and does not identify whether the source
is verify-time state evolution or accepted-prefix commit selection. The trace copied all GDN state
to host at each commit and expanded this ten-token run to 3m06, so it was removed. A narrower
capture below resolves the prefix-selection question and measures the numerical state deltas.

A narrower capture compared the actual state tensors. The prefill state and state after token 1596
were bit-identical between modes. After the first accepted MTP pair (1144, 310), the committed
state aligns with the MTP-off state after token 310, rather than the earlier state after 1144,
which supports correct accepted-prefix selection. Relative to serial decode after token 310,
GDN state 0 has 2/30,720 BF16 convolution elements changed (mean absolute decoded BF16 delta
1.9e-10, max 3.8e-6) and FP32 recurrent state mean/max deltas of 2.2e-6/0.0046. At GDN state 27
(transformer layer 36), convolution deltas are 0.034 mean / 0.461 max and recurrent deltas are
1.23e-4 mean / 0.0188 max. The discrepancy therefore accumulates along the multi-token target
execution while the selected commit prefix is correct; this still does not distinguish replay
record inputs from replay recurrence/fold arithmetic. Next, compare the real artifact's per-token
GDN replay records and folded state against the single-token route and an independent recurrence
oracle at state 0 and state 27. The temporary capture was removed after this comparison.

The disk schedule runs eagerly: CUDA Graph capture is disabled for this route until stable
cache contents and graph replay are supported. Numerical comparison of a real layer against
an independent oracle, MTP acceptance/commit and state-restoration qualification, Vision,
prefix reuse, and graph capture remain open. The matched original-model performance matrix is
recorded below; it does not qualify Flash-Next MTP or the other feature gates.

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
