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

## Current execution status (2026-11-08)

The Orin branch was rebased onto the refreshed `origin/dev` tip `41da21f47` after ten upstream
commits arrived. The merge retained the Orin exact-token benchmark input and expert-cache timing
fields while taking upstream's newer benchmark warmup and runtime scheduling changes. The complete
JP6/SM87 build passed (1,012 targets). Focused bench-report, Qwen4Exp state, GDN control, and QSA
BF16/INT8 tests passed. The first CTest invocation lacked the Jetson dependency library path and
could not start the real-artifact executable; rerunning it directly with that path loaded the V3
artifact. The initial three-request, 16-token, draft-1, graphs-off cohort reached decode batch three
but failed exact token parity at row 2, output token 2; rows 0 and 1 matched. A four-token repeat
also reproduced the mismatch and reached decode batch two. The repeated-prompt trace and correction
are recorded below.

The `NINFER_TEST_COHORT_PROMPT` diagnostic override now repeats one prompt across the cohort. With
`The largest ocean on Earth is`, a four-token, three-request, graphs-off, draft-1 public-Engine
comparison reproduced the mismatch on all rows at output token 2 (serial `1596 1144 4087 1156`,
MTP `1596 1144 310 4087`); the largest decode batch was two. Thus different prompt histories are
not required for this failure. The test harness change compiled and this real-artifact run reached
the expected parity assertion; a layer trace using the repeated-prompt case is the next diagnostic.
That trace now follows the same prompt and pins the first mismatch to QSA layer 23, an indexed
attention layer. For input token 1144 at position 59, Step and Verify matched at the layer input,
query/gate/value/index-key projections, and prepared query; the projected key differed in one of
512 BF16 values (maximum absolute difference `9.765625e-4`, relative L2 `7.42e-5`). The sparse
attention result then differed in 1,055 values (relative L2 `2.16e-3`). The cause was the Verify
path projecting each column separately while decode projected the two-column batch together.
QSA now uses the common batched projection association for Verify. After this change, the original
three-prompt, 16-token, draft-1 cohort passed exact parity at decode batch 3 with Graphs off; the
repeated-prompt eight-token cohort passed at batch 3 both with Graphs off and Graphs on. Focused
QSA prepare tests passed for BF16 (65 cases) and INT8 (50 cases), the QSA indexer suite passed
(59 cases), and the Qwen4Exp state suite passed. A post-fix rerun of the original three-prompt,
16-token, draft-1, graphs-off real-artifact cohort again passed exact parity at decode batch 3.
The exhaustive sparse-GQA executable was stopped after running for over four minutes without
returning; sparse-GQA arithmetic itself was not changed. Post-fix end-to-end Flash-Next measurements
are recorded below.

The eight-request, 16-token, draft-1, graphs-off real-artifact rerun reached decode batch 8 but did
not pass exact parity: row 1 first differed at output token 14 (one fallback-free mismatch), and
row 7 first differed at output token 4 (one fallback). The other six rows matched. An eight-row,
repeated-prompt, eight-token follow-up could not start inference because Engine startup reported
`Device budget cannot hold model weights and required execution resources`; a concurrent unrelated
CUDA build was consuming substantial host CPU and memory, so this is not evidence for or against
same-history parity. The eight-row gate remains open.

A follow-up candidate kept QSA batched for both paths and forced PLE projection to use the same
per-column association for Step and Verify. Its build passed and the three-request, 16-token,
draft-1, graphs-off cohort passed exact parity at batch 3. The eight-request, 16-token run entered
uninterruptible block I/O while reading the file-backed artifact; kernel logs showed recurring
NVMe request timeouts. After 14 minutes without a result, the test was terminated. This is not
parity evidence for that candidate. A common batched PLE candidate instead failed the three-request
gate at row 0, output token 7. Both PLE route experiments were reverted; the common batched QSA
route remains the only retained correction, and the eight-request gate remains unresolved.

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
produced 4145. A temporary device-to-host diagnostic captured the represented BF16 sample for the
same decode context (sequence 2, frontier 71, input token 14862). Single-row ordinary and
MTP samples were bitwise equal. In the cohort run, ordinary and MTP samples differed in 2,448 of
2,560 channels (relative L2 3.30%, maximum absolute difference 0.75); the ordinary cohort sample
also differed from single-row ordinary by relative L2 2.54%. The corresponding top-two output logits
were only 0.125 apart: ordinary cohort ranked token 4145 at 23.125 over 2716 at 23.0, while MTP
ranked 2716 at 23.0 over 4145 at 22.875. This confirms that the mismatch is already present at the
BF16 sample boundary and is magnified into an argmax change; it does not identify whether the
boundary difference comes from batch-shape arithmetic or batched state handling. Cohort parity is
established only through three rows at the previously tested eight-token horizon; longer cohort
decoding remains unqualified pending that attribution.

The first temporary layer-boundary capture was invalid: it sliced the residual stream axis instead
of the token-column axis, so its reported layer-0/layer-1 comparison must be ignored. A corrected
token- and position-filtered capture compared the same anchor, input 14862 at position 71. Before
changing arithmetic, ordinary batched PLE projection differed from MTP's per-column projection in
17 of 12,800 BF16 values (relative L2 0.000240); the PLE gate differed in 92 of 10,240 values,
and the post-PLE residual differed in 20 of 10,240. A temporary experiment made ordinary decode
use MTP's per-column projection: projection and gate then matched bitwise, but the post-PLE
residual still differed in 5 values (relative L2 2.62e-5, maximum absolute difference 6.10e-5),
and GDN output differed in 377 of 2,560 values (relative L2 0.000918). A separate capture of the
nine-column PLE source history at the same anchor found 548 of 92,160 BF16 values different
(relative L2 0.000446, maximum absolute difference 0.0078125). With the temporary per-column
projection change, source-history differences fell to 201 values (relative L2 0.000195), but
remained nonzero and the generated-token mismatch remained at output token 14. Thus projection
arithmetic contributes to accumulated state drift, while PLE state or convolution behavior still
differs before the failing anchor. No production change is retained; temporary instrumentation and
the arithmetic experiment were removed, and the JP6 build completed successfully.

A short-horizon three-request run then aligned an earlier anchor at position 65 (sequence 2,
input token 1324): ordinary decode used physical row 1 / column 0 and MTP verification used row 1 /
column 1. The ten-token cohort passed exact token parity. At this anchor, stem embedding, stem
residual, and GDN input matched bitwise, while GDN layer-0 output first differed in 17 of 2,560
BF16 values (relative L2 0.000196, maximum absolute difference 0.000977). Later mixer, MoE, and
PLE inputs also differed. This localizes an early batch-versus-verify arithmetic difference to
within or at the GDN layer-0 boundary for this anchor; it does not establish that this is the cause
of the later position-71 mismatch. A second temporary attempt to capture GDN sub-stages did not
produce those captures, so control projection, recurrent update, normalization, and output
projection remain unlocalized. No code change was retained, and the source worktree is clean.

A further Orin-only route experiment tested that attribution on the same real artifact. For a
three-row ordinary decode, GDN control projection used tiled GEMV at three columns; MTP verification
used merged MMA at nine columns. Temporarily selecting merged MMA for both made GDN layer 0 match
bitwise through its output at the position-65 anchor, but the 16-token cohort still diverged at
output token 14, with the close-logit ordering reversed. A second temporary NVFP4 A16 GDN input
projection schedule alignment (same sliced-K schedule for three and nine columns) also left that
token mismatch unchanged. With both alignments, layer 1's input, control values, and z projection
matched bitwise, while q/k/v differed in 258/2,048, 494/2,048, and 1,317/6,144 BF16 values. The
first remaining difference is therefore at the layer-1 projected/convolved QKV boundary; the
capture does not distinguish raw QKV projection arithmetic from convolution-history/update versus
record behavior. Both route experiments and all diagnostics were removed, and the restored JP6
tree built successfully. No performance qualification was made for either experiment.

A final state capture at the same position-65 anchor compared each GDN layer's convolution source
history. The MTP verifier starts one token earlier than ordinary decode, as expected: at layer 0,
ordinary history planes 0 and 1 matched MTP planes 1 and 2 bitwise across all 10,240 channels. At
layer 1 those shifted planes matched only 10,006/10,240 and 8,096/10,240 values, showing that its
history had already diverged before the current token. The MTP record for the preceding token at
position 64 was also captured; its layer-1 projected-convolution input matched ordinary decode's
committed newest plane in 8,252/10,240 values. This localizes accumulated history drift to before
the position-65 layer-1 transition, but does not yet distinguish a layer-1 input difference at
position 64 from its projection/convolution route.

A follow-up three-row, ten-token run captured the same position-64 input token (760) at every GDN
layer. Layer-0 input and output matched bitwise between ordinary decode and MTP verification; layer-1
input already differed in 196 of 2,560 BF16 values, and its convolution source state differed in
5,420 of 30,720. Thus layer-0 GDN is not the first divergence at this earlier anchor: the mismatch
enters between layer-0 and layer-1 GDN, in the intervening transformer path. The cohort's generated
token IDs still matched exactly for this ten-token horizon. Temporary capture code was removed, and
the full restored JP6 build completed successfully.

The transformer-boundary trace showed layer 0's post-combine residual was still bitwise equal, while
layer 1 entered with 8 differing BF16 values. A follow-up capture around PLE on the same position-64
anchor confirmed its residual, normalized input, and embedding were equal before PLE. The first PLE
output difference was its batched projection (13/12,800 BF16 values), followed by the gate
(47/10,240); the convolution/norm residual differed in 72/10,240 values and normalized output in
109/10,240. The three-row, ten-token run with MTP-4 passed exact token parity and exercised decode
batch size three. This places the first observed divergence at ordinary batched PLE projection versus
MTP's per-column projection; later PLE state/convolution arithmetic also accumulates the difference.
All capture code was removed and the JP6 tree rebuilt.

A controlled route-alignment experiment then forced ordinary PLE projection to use the per-column
route and selected the same paired 256-thread Conv+Norm block for low-column decode as for MTP
verification. The three-row, 16-token real-artifact test still reproduced the same row-1 token-14
argmax mismatch (serial 2716, cohort MTP 4145), with a three-row decode batch. Thus PLE contributes
an early numerical difference but aligning those PLE routes is insufficient to resolve the longer
cohort mismatch. Both temporary route changes were reverted; the next unresolved captured boundary
is GDN layer-1 QKV projection/convolution after its input, control, and z values matched.

The Q8 GDN route was extended so long aggregate verify windows use a row-aligned fused GEMV tile,
matching ordinary decode's per-token arithmetic. A new same-input, same-state operator comparison
reproduced the prior route difference in Q/K/V and published convolution history (Z already
matched); after the change all first-token outputs and published history compare exactly. The
existing width-9, batch-3 Q8 record fixture also passes. The real 16-token cohort still fails at
row 1, token 14 (ordinary token 4145, MTP token 2716), so the operator alignment is a confirmed
correction but not a cohort-parity fix. The PLE projection/Conv+Norm alignment experiments remain
insufficient and were reverted. The next investigation should capture the remaining state and
logit differences at the failing position with the production Q8 route retained.

The three-request, 16-token failure was rerun against the committed Q8 route with CUDA Graphs off:
row 1 matched through token 13, then ordinary decode chose 4145 while MTP chose 2716 at token 14;
all three requests completed and the MTP execution exercised decode batch size three. This confirms
the remaining mismatch is reproducible with the production Q8 correction and is not limited to the
earlier temporary route experiments.

A temporary same-position trace at input token 14862 / position 71 narrowed the first observed
post-mixer difference to layer 1 MoE. Its 2,560-value BF16 hidden input matched bitwise between
ordinary batch decode (3 router columns) and MTP verify (10 router columns). Cache route planning
was aligned to the registered SM87 NVFP4 executor's one-token slices. A synchronized capture then
reproduced that exact input in ordinary and MTP execution: the router control-weight hash, route
IDs, coefficient bits, shared coefficient, and all 2,560 BF16 MoE outputs matched exactly. The
previous report of 1,192 differing outputs came from an unsynchronized device-to-host diagnostic
and is invalid. The three-request cohort still fails at row 1, token 14 (ordinary 4145, MTP 2716),
so layer 1 MoE is ruled out for the matched input; the source of the remaining state/decode
difference remains unresolved. Temporary activation/router capture code was removed. The real-
artifact loading/stream-one-layer test passed on JP6 after the cache-route change, including the
file-backed NVFP4 MoE execution and its decoded-weight oracle. The source fix is committed on the
local Orin branch; pushing that branch is still rejected by GitHub with HTTP 403 for the configured
identity.

A direct PLE capture on the same real-artifact 16-token cohort compared the ordinary row with its
matching MTP verify column at token 14862 / position 71. The nine-column PLE source history differs
in 548/92,160 BF16 values; per-plane differences from oldest to newest are
7, 38, 47, 347, 21, 29, 24, 19, and 16. The plane for position 64 differs in 47 values, exactly
matching the 47-value PLE gate difference captured at token 760 / position 64. This ties the earlier
projection/gate arithmetic difference to persisted PLE history at the failing position. The PLE
projection differs in 17/12,800 values and its gate in 92/10,240 at the later anchor; convolution
output and normalization differ in 20/10,240 and 21/10,240. A temporary MTP route that grouped
candidate projections by request row still failed the three-row, 16-token parity case at token 14;
the one-row, eight-token route passed. The candidate was reverted without performance qualification.
Next, compare PLE record generation and accepted-prefix state extraction by column to find where
the accumulated history drift first enters, then rerun the 16-token gate.

A follow-up capture tested whether routing single-column PLE projections through the same MMA path
as multi-column decode removes the record-generation difference. At token 760 / position 64 and
token 14862 / position 71, the paired ordinary and MTP PLE residual input, projection, gated value,
and newly generated record were bitwise equal. At the later anchor, accumulated source history still
differed in 347/10,240 values of one plane. The three-request, 16-token cohort continued to diverge
at output token 14. This route alignment therefore removes the sampled PLE record difference but
does not explain or fix the cohort failure; it is not retained. The capture does not establish
whether the remaining history difference comes from an earlier unsampled record or accepted-prefix
state publication. Temporary capture code and the MMA selector experiment were removed, and the
clean JP6 build completed successfully.

A stream-synchronized capture at the adjacent token 1324 / position 65 anchor refined the PLE and
layer-0 attribution. Host and device PLE lookup embeddings matched exactly between ordinary and MTP.
The PLE input residual differed in 577/10,240 BF16 values. Temporarily using per-column PLE
projection made the projected values match exactly; the PLE convolution record still differed in
347/10,240 values. Layer-0 residual and normalized inputs, and its GDN input, matched exactly.
GDN Q/K/V and Z also matched, while control projections G/Beta differed in 28/96 and 19/96 values;
the recurrent output differed in 2/6,144 and the GDN output in 17/2,560. This places the first
observed layer-0 difference at GDN control projection for this anchor.

An Orin-only experiment extended the tiled-GEMV GDN control route across 1-48 columns, covering
both ordinary decode and the supported 8-request by 6-column verify extent. At the captured
position-65 row, G/Beta, Q/K/V, Z, recurrent, normalized, and GDN output then matched bitwise. In the
three-request, 16-token cohort, the original row-1 token-14 mismatch disappeared, but row 0 newly
diverged at output token 7 (ordinary 3296, MTP 3134). Restoring batched PLE projection did not
remove that row-0 failure. The route change is not retained because the cohort gate still fails;
the first differing operator for row 0 remains to be captured. All diagnostic and route experiments
were removed, and the restored JP6 build completed successfully.

Rechecking that control-route experiment with the original batched PLE on the current source
changed the three-request, 16-token result: exact parity passed twice, once with targeted GDN
tracing and once with tracing disabled. This does not reproduce the earlier row-0 token-7 failure.
The eight-request, 16-token run still failed under the widened route: row 1 at output token 14, and
rows 4 and 7 at token 2. Its largest measured decode batch was four. The experiment therefore does
not establish the supported 1-8-request cohort gate and is not retained. Trace-enabled GDN layer-0
row-0 values at token 4145 / position 63 matched from input through output, which leaves the
eight-request divergences unattributed.

A fresh trace on the unmodified Orin route reproduced the three-request, 16-token row-1 mismatch at
output token 14 (serial 4145, MTP 2716; 13 drafts, 11 accepted, no fallback). At layer 0 and matching
row-1 token positions 58-67, the input to GDN and its Q/K/V/Z projections matched bitwise, but the
control projections differed: G differed in 27-35 of 48 FP32 values and Beta in 19-33 of 48. The
recurrent output differed in only 1-2 of 3,072 FP32 values at positions 61, 62, 64, and 65; the
BF16 GDN output differed in 3 of 2,560 values at position 61 and 17 at position 65. At position
68, the GDN input already differed substantially, so this capture does not attribute later layers
or establish that the control-projection drift alone causes the token mismatch. A three-request,
eight-token cohort on the same unmodified route passed exact token parity. Temporary tracing was
removed and the clean JP6 build completed successfully.

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

### JP6 original-model and Flash-Next baselines (2026-11-07)

The local Qwen3.8-27B groupwise-int artifact is already V3. The pinned Qwen3.6-35B-A3B V2
artifact was upgraded to a separate V3 path with `tools/upgrade_ninfer_v2_to_v3.py`; the original
remains intact. Direct SHA-256 comparison of every unchanged object payload passed for all 939
objects. The maintained Qwen chat template is intentionally replaced by the upgrader. The real
Qwen3.5-family loader passed on both artifacts with MTP enabled: Qwen3.8-27B materialized 17.54 GiB
of device weights in 3.16 s, and 35B-A3B materialized 21.94 GiB in 4.08 s. The Qwen3.8-Flash-Next
V3 artifact also passed its learned Vision-weight oracle and the public Engine benchmarks below.

The six `ninfer_bench` runs used the same 33-token prompt, no thinking, 32-token output budget,
INT8 KV, 1,024-token context, CUDA Graphs, prefix reuse off, one warm-up, and three measured
repetitions. The public Engine ran on AGX Orin / CUDA 12.6 in MAXN with `jetson_clocks`; the
reported locked clocks were CPU 2.2016 GHz, GPU 1.3005 GHz, and EMC 3.199 GHz. Values below are
medians across the three repetitions; `wall tok/s` includes prefill and request overhead.

| Artifact | Speculation | Prefill tok/s | Decode tok/s | Wall tok/s | Wall s | Draft accepted |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B groupwise-int V3 | off | 62.06 | 7.29 | 6.68 | 4.79 | — |
| Qwen3.8-27B groupwise-int V3 | MTP-3 | 61.77 | 9.36 | 8.32 | 3.85 | 18 / 36 (50.0%) |
| Qwen3.6-35B-A3B groupwise-int, migrated V3 | off | 122.97 | 36.52 | 28.60 | 1.12 | — |
| Qwen3.6-35B-A3B groupwise-int, migrated V3 | MTP-3 | 122.67 | 45.54 | 33.65 | 0.95 | 20 / 32 (62.5%) |
| Qwen3.8-Flash-Next V3 | off | 2.63 | 1.46 | 0.95 | 33.54 | — |
| Qwen3.8-Flash-Next V3 | MTP-3 | 2.58 | 1.38 | 0.91 | 35.19 | 19 / 32 (59.4%) |

The two groupwise-int targets improved on this short MTP-3 workload. Flash-Next did not: its
file-backed MoE route remains storage-bound in these measurements. These are focused baseline
points, not the long-corpus/performance-qualification matrix. The full JP6 build passed, as did
artifact framing/materialization, OpenAI/Anthropic schema, Qwen3.5 runtime-mechanism, GDN snapshot,
Qwen4Exp loading/state/Vision, and both real-model loading checks. The generic expert-parallel Op
test still reports eight NVFP4/FP8 cases for which the Orin target deliberately has no
`routed_swiglu` block-scaled kernel; the registered Flash-Next Engine uses its separate supported
complete-MoE path.

The GDN snapshot suite also found and now covers two Orin issues: NVFP4 A4 scratch must remain
disabled when the target compiles without that implementation, and Q8 fused aggregation must not
select its 16-token window kernel for a 17-token single-row input. The Q8 route now falls back to
the materialized path at that boundary. Stream validation in the CUDA-graph snapshot test now
happens before capture, which is required by CUDA 12.6's `cudaStreamGetFlags` behavior.

### Post-upstream remeasurement (2026-11-08)

After rebasing onto upstream `dev` `41da21f47`, the seven public-Engine measurements below were
repeated on AGX Orin in MAXN with `jetson_clocks` (CPU 2.2016 GHz, GPU 1.3005 GHz, EMC 3.199 GHz),
CUDA 12.6, one warmup, and three measured repetitions. The groupwise-int runs used the same
`/tmp/ninfer_jp6_benchmark_messages.json` input (707 tokens for Qwen3.8-27B, 665 for 35B-A3B),
64 output tokens, INT8 KV, 1 GiB context capacity, 512 MiB headroom, 2,048-token max context,
128-token prefill chunks, CUDA Graphs, and prefix reuse off. An additional 35B-A3B MTP-2 run used
the prior 1,024-token prefill chunk to match its earlier measured configuration. Flash-Next used the `hi` prompt
(53 input tokens), 10 output tokens, INT8 KV, 512 MiB context capacity, 4,096-token max context,
128-token prefill chunks, CUDA Graphs, and prefix reuse off. Rates and request wall times are
medians across the three measured repetitions; draft acceptance totals were identical in each
repetition. Prompt and output lengths differ across artifact families, so compare rows within each
artifact only.

| Artifact | Speculation | Input → output tokens | Chunk | Prefill tok/s | Decode tok/s | Request wall s | Draft accepted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B groupwise-int V3 | off | 707 → 64 | 128 | 181.82 | 7.285 | 12.541 | — |
| Qwen3.8-27B groupwise-int V3 | MTP-3 | 707 → 64 | 128 | 181.35 | 10.504 | 9.899 | 39 / 69 (56.5%); 1 fallback |
| Qwen3.6-35B-A3B groupwise-int V3 | off | 665 → 64 | 128 | 573.52 | 36.290 | 2.898 | — |
| Qwen3.6-35B-A3B groupwise-int V3 | MTP-3 | 665 → 64 | 128 | 570.67 | 55.287 | 2.307 | 44 / 53 (83.0%); 1 fallback |
| Qwen3.6-35B-A3B groupwise-int V3 | MTP-2 | 665 → 64 | 1024 | 1193.17 | 53.418 | 1.738 | 39 / 46 (84.8%); 1 fallback |
| Qwen3.8-Flash-Next V3 | off | 53 → 10 | 128 | 3.327 | 1.534 | 21.799 | — |
| Qwen3.8-Flash-Next V3 | MTP-4 | 53 → 10 | 128 | 3.539 | 1.238 | 22.247 | 6 / 9 (66.7%); 0 fallbacks |

The Qwen3.8-27B and 35B-A3B serial rates agree with their earlier 64-token, long-prompt runs
within 0.3%. The 35B-A3B MTP-2 run also matches the earlier same-configuration report within
0.4%, including prefill and wall time. Qwen3.8-27B MTP-3 and 35B-A3B MTP-3 improve decode throughput in this workload.
Flash-Next MTP-4 is slightly slower than serial at this short output horizon; the accepted-draft
count does not imply a speedup. The earlier 2026-11-07 table is a separate short-prompt workload.
Per-repetition reports are in `/tmp/ninfer-post-rebase-*.json`; temperature was not logged during
this measurement series.

### Flash-Next performance after QSA cohort fix (2026-11-08)

After making QSA Verify use the same batched projection association as Step, the public Engine
benchmark was repeated on the supplied Flash-Next V3 artifact. Both runs used the 53-token `hi`
input, 10-token output limit, BF16 KV, 512 MiB context capacity, 4,096-token maximum context,
128-token prefill chunks, CUDA Graphs, prefix reuse off, one warmup, and three repetitions. The
AGX Orin remained in MAXN with CPU/GPU/EMC clocks locked at 2.2016/1.3005/3.199 GHz. Values are
medians; wall time includes prefill and request overhead.

| Artifact | Speculation | Input → output tokens | Prefill tok/s | Decode tok/s | Request wall s | Draft accepted |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next V3 | off | 53 → 10 | 2.829 | 1.189 | 26.310 | — |
| Qwen3.8-Flash-Next V3 | MTP-4 | 53 → 10 | 3.542 | 1.249 | 22.170 | 6 / 9 (66.7%); 0 fallbacks |

This matched post-fix pair shows lower wall time with MTP-4 for this short workload. The serial
point is slower than the previous post-upstream serial result (21.799 s); the MTP-4 point is close
to its previous result (22.247 s). The pair is a focused remeasurement, not a stable long-horizon
Flash-Next performance qualification. Reports: `/tmp/ninfer-post-qsa-fix-flash-none.json` and
`/tmp/ninfer-post-qsa-fix-flash-mtp4.json`.

### Remaining gates

- The QSA Step/Verify numerical mismatch at layer 23 is fixed by using the same batched projection
  association for both paths. Exact parity now passes for the three-prompt, 16-token, draft-1 cohort
  with Graphs off, and for a repeated-prompt, eight-token, three-request cohort with Graphs both on
  and off. Single-request Text Graph/MTP parity remains exact through 128 output tokens. The
  separate eight-request, 16-token cohort still fails as described below, so longer cohort decoding
  remains unqualified. Continue longer Vision/video horizon qualification. Image-conditioned runs cover 8 and 32 tokens, and
  video-conditioned runs cover 4, 32, and 35 generated tokens (the last run ended on the stop token
  with a 64-token cap).
- An eight-request, 16-token cohort was repeated with `max_context=256`, a 4 GiB context reservation,
  and CUDA Graphs disabled to see whether a smaller logical context enabled wider decode batches.
  It still reached only `largest_decode_batch=4` and failed exact parity on rows 1 (first difference
  at output token 14) and 4 (token 13). The run exercised draft and acceptance work without fallback.
  This context setting did not qualify eight-row batching; retain the original test configuration.
- Repeating the eight-request, 16-token cohort with MTP draft width 1 reached
  `largest_decode_batch=8` under the original 1,024-token context and 4 GiB reservation, with Graphs
  disabled. Exact parity still failed on row 4 at output token 12; this run recorded one fallback
  step. Thus the prior batch-4 ceiling was workload timing (the one-prefill-per-cycle scheduler let
  MTP-3 rows finish before all prompts became decode-ready), while the numerical cohort mismatch
  remains. A single-request serial/MTP comparison of the exact row-4 prompt passed for 16 tokens
  with draft width 1, confirming that this mismatch requires batched execution. For the eight-row
  verify, 16 flattened control columns select GDN control MMA; the single-row verify uses 2 columns
  and tiled GEMV. Extending the tiled GEMV boundary through 48 columns still reached batch 8 but
  failed exact parity on rows 1 (token 14) and 4 (token 13), without fallback. That kernel-family
  boundary is not a sufficient fix; the current production route was restored. The discrepancy
  still requires a layer-level trace to identify its first divergent state. After restoration the
  full JP6 build and `ninfer_gdn_control_test` passed; the latter checks each selected route against
  its independent projection oracle, but its existing route-invariance case covers only T=1/T=2.
- Temporary lane/position traces of the real serial and eight-row draft-1 cohort found two
  shape-dependent numerical seams in the Flash-Next GDN path. With identical layer-0 input
  fingerprints, control G/Beta outputs differed when the serial and verify widths selected
  different projection families. GDN Q/K/V/Z also differed at the first layer before the control
  and replay state updates, locating a separate split in Q8 A16 projection/convolution routing.
  Traces use hashes, so they prove bitwise differences but do not measure their numeric size.
- The Orin control route now keeps tiled GEMV through the real verify width of 16 columns. Exact
  T=1 versus T=16 first-column equality passes, and the T=16 projection passes its FP64 oracle.
  A warm CUDA Graph Op benchmark on Orin (CUDA 12.6, SM87, batch 1, 20 repetitions) measured
  8.48, 14.85, and 21.41 microseconds at widths 1, 8, and 16 respectively. This records the
  selected short-window route's latency curve; it is not an end-to-end inference result.
  For Flash-Next Q8 GDN input projection, Orin now selects the aggregate path for all multi-row
  windows up to 16 flattened columns; this removes exclusions that made T=1,B=8 use materialized
  projection while T=2,B=8 used aggregate. The new exact B=8 decode-versus-record fixture passes,
  as does the prior B=3,T=9 fixture.
- The real-artifact eight-request, 16-token, graphs-off, draft-1 cohort still reaches batch 8 but
  now differs only on row 4, first at output token 13, with no fallback. Rows 1 and 7 match in this
  run. Earlier route settings fixed different rows, so full cohort parity remains unqualified and
  scheduling-sensitive. A four-token stage trace reached only batch 2 and showed exact row-7 GDN
  stage fingerprints there; it does not cover the failing batch width. A synchronized eight-request
  draft-1 trace matches the failing row-4 token (ID 13, absolute position 68) against serial decode
  by exact prompt-token-vector hash. A follow-up real-artifact probe recomputed layer-1 attention HC
  mix as a single-column operation from each run's actual input and weights. In both serial and MTP
  verify, the single-column output hash equals that run's ordinary batched output hash, while the
  serial and verify input hashes differ. The indexed stage trace confirms layer-0 combined state is
  bit-identical for the selected row, but the layer-1 HC-mix input differs. This is the expected PLE
  boundary: the artifact declares `ple_layer_ids: [1]`, the runtime indexes that layer directly,
  and execution consumes PLE after layer 0 and before layer 1. The discrepancy is therefore in PLE
  input/state or its projection, gate, convolution, and normalization path; it is not an unexplained
  workspace mutation and the HC-mix output route is ruled out for this token. The latest cohort
  again reached batch 7 and failed row 4 at output token 13, with all six drafts accepted and one
  fallback. A temporary route experiment used per-column PLE projection for ordinary decode as
  well as verify. For the matched row/token, the PLE embedding, residual input, incoming state,
  projection, gate, residual output, and normalized output fingerprints then matched exactly, but
  the end-to-end cohort still differed at output token 13. The PLE projection route is a real
  numerical seam, but correcting it alone does not resolve the remaining mismatch; no production
  route change was retained.
- A separate eight-request, draft-1 experiment forced per-column QSA input projection on ordinary
  decode, matching Verify while leaving PLE unchanged. The mismatch moved to row 7 at output token
  4 (the original route mismatched row 4 at token 13), so this route change alone is not a fix and
  was removed. It confirms that projection-route changes can move greedy cohort divergence; the
  combined-route experiment below identifies the projected query as the first QSA stage difference
  for the original row-4 mismatch.
- With ordinary decode and verify both using per-column PLE projection, the same row-4/token-13
  trace shows bit-identical layer-1 QSA input. The first divergence inside QSA is its projected
  query: serial decode uses the batched `linear_split` route while Verify uses the per-column route.
  The hashes already differ at projected query, before QSA preparation, block selection, sparse
  attention, or output projection. A combined PLE+QSA per-column route experiment then produced
  exact generated-token parity for all eight 14-token requests (six drafts accepted per request),
  but the public integration test failed its full-width gate: the workload only reached decode
  batch 7, not the requested batch 8. This is useful parity evidence through the observed widths,
  but does not qualify the eight-row Verify path or establish that the route change is robust; no
  production change was retained. Extending that same experiment to 24 output tokens did reach
  decode batch 8, but row 1 diverged at output token 14 (serial 4145, MTP 2716; 12 drafts, 11
  accepted, no fallback). The candidate route alignment therefore is not a full-width parity fix.
  The next trace should follow row 1 through the first differing Verify round and its target logits
  to find whether the new divergence begins in another operator or in the commit decision. The
  temporary route edits and tracing were removed; rebuilding the unmodified JP6 tree is the next
  validation step.
- A 2026-11-08 asynchronous same-stream snapshot trace corrects the earlier synchronous-boundary
  interpretation. At token 28320 / position 77, two queued device-to-host snapshots agree within
  each call at layer-4 GDN entry (ordinary `9e84eff2c131b124`, Verify `e3119eb860107278`), so the
  earlier apparent within-call mutation was a synchronization/timing artifact. Ordinary and
  Verify layer-0 block outputs are bit-identical; the first observed difference is immediately
  after PLE (`ple_residual`: `488223a1b0d5aa9d` vs `f6cda9f022751963`; `ple_normalized`:
  `01e086a57495d5f1` vs `bed4f421d847171c`). The source currently uses batched PLE projection for
  ordinary Step and per-column projection for Verify. A temporary change aligned both decode
  routes to per-column projection while keeping prefill batched, but the same 8-request, 22-token,
  draft-1 real-artifact cohort still failed exact parity (row 0 first differed at token 21,
  serial 874 vs MTP 6157; row 2 first differed at token 2). The route change was reverted. This
  identifies PLE as the earliest observed divergence but does not establish projection as its
  cause; full-width MTP parity remains unresolved and unqualified. The next trace should separate
  PLE projection, gate, convolution record, and state inputs/outputs for the first divergent row.
- A focused Op probe compared the identical first eight BF16 columns at the T=8 GEMV and T=9 MMA
  seam. The G and Beta values differed in 123/384 and 80/384 entries respectively, with maximum
  absolute differences of `3.58e-7` and `1.20e-7`. The full GDN control suite passes its independent
  projection oracle at both T=8 and T=9. The temporary probe was removed after measurement.
- The downstream replay/fold recurrence itself matches the serial state-transition contract for
  identical inputs and initial state: `ninfer_gated_delta_net_replay_record_test --case
  gdn_record/h48/w2/b8/ragged` passes exact recurrent-output comparison against repeated
  `gated_delta_net_batch_update`, and `ninfer_gdn_replay_fold_test --case
  gdn_fold/l48/h48/w16/b8/disjoint/commits_0_1_2_3_7_13_15_16` passes exact recurrent-output and
  committed-state comparisons across 48 layers. This narrows the cohort mismatch to upstream GDN
  inputs/control values or their state history rather than replay/fold transition math.
- Repeat the public mixed-media/ExpertCache breakdown under stable storage conditions. The current
  mixed-media Graph report is one diagnostic sample with NVMe timeouts; Vision numerical and
  encoder-throughput qualification covers artifact weights at up to 64 patches.
- A 2026-11-09 real-artifact route trace found the PLE gate consistently selected paired
  `320x8` for both ordinary Step and Verify at the observed decode shapes; convolution selected
  `512x2` for both. These selections do not explain the earlier Step/Verify arithmetic difference
  for this cohort. With the committed runtime and CUDA Graphs off, the three-request, 16-token,
  draft-1 cohort passed exact token parity three consecutive times, each reaching decode batch 3;
  the prior row-1 token-14 mismatch was not reproduced. The first traced pass overlapped a kernel
  NVMe timeout, so it does not qualify storage reliability. The eight-request full-width parity
  gate remains open. The route logger was removed and the unmodified JP6 build and PLE Op suite
  passed after cleanup.
- A follow-up eight-request, 24-token, draft-1, graphs-off run reached decode batch 8 and failed
  exact parity on rows 1 (first difference at output token 14), 3 (token 18), and 7 (token 4).
  The route trace showed ordinary Step at `T=1,B=8` and Verify at `T=2,B=8` both using the same
  PLE convolution `norm-512x2-conv` and gate `320x8` routes. A temporary selector experiment
  forced `norm-512x2-conv` for all `T=2,B=2..8` verification batches, aligning them with Step;
  the same rows and first-difference positions still failed. Thus the earlier `T=2,B=8`
  `norm-256x2-conv` selection was not sufficient to explain the full-width mismatch. The
  experiment and route logger were removed; this candidate did not establish a production fix.
- A temporary BF16 target-logit probe was built into the JP6 real-artifact test path. In the serial
  row-1 decode trace, the step producing token `4145` ranked it at `23.25` and token `2716` at
  `23.0`, a narrow `0.25` margin. The probe captured 20 serial row-1 decisions and 16 MTP Verify
  columns (eight rounds), but the MTP run hit its 600-second timeout while a file-backed artifact
  `pread64` remained blocked and NVMe timeouts recurred. The captured MTP columns could not be
  aligned conclusively to the serial near-tie position, so this does not establish whether target
  logits or acceptance first diverge. The temporary probe was removed; a repeat needs a completed
  real-artifact run with row/position inputs attached to each logit snapshot.
- Distinguish SSD/controller cache state for physical cold/warm storage measurements; `O_DIRECT`
  isolates requests from the host page cache but does not flush or measure device caches.
- Storage diagnosis: the 4 TB Seagate FireCuda 530 ZP4000GM30023 (`SU6SM001`, active slot 1)
  reports no critical warning,
  100% spare, 0 media errors, and 44 C after the latest run. PCIe is 16 GT/s ×4 with ASPM disabled.
  A temporary APST-disable run still logged NVMe I/O timeouts; APST and its original PS3/PS4
  transition table have been restored. In that run the 64-token text parity case passed, then one
  three-request cohort submission expired while waiting for admission after an NVMe timeout. APST
  alone therefore does not explain the storage stalls, and the cohort result remains qualified by
  its earlier successful run rather than this repeat. A later eight-row Flash-Next cohort read again
  entered uninterruptible block I/O while the kernel logged repeated NVMe request timeouts; that
  run was terminated without a parity result. Read-only follow-up showed 0% drive wear, 47 C, 100%
  spare, 0 media errors, and no critical warning. The endpoint and root port report 16 GT/s ×4,
  ASPM disabled, and clear Device Status/AER error bits. APST is currently enabled with the
  controller's 100 ms transitions to PS3 and 2 s transition to PS4. These checks show that the
  observed stalls are not accompanied by SMART media-health or current PCIe error indicators; the
  timeout source is still unidentified. An experimental 512 KiB maximum for synchronous
  file-backed expert reads did not prevent the repeated-prompt cohort from blocking in `pread64`
  on a 512 KiB request; that run was terminated without a parity result and the generic reader
  change was removed. Limiting individual request size is therefore insufficient evidence for a
  fix. A real-artifact PLE-only depth comparison at batch 8 × 4 positions read 1.875–1.906 MiB
  in 3.35–4.02 ms at depth 128, versus 5.93–6.38 ms at depth 16 (three samples per setting).
  Lowering the PLE depth globally would regress this path and has not been adopted. The comparison
  does not exercise simultaneous expert-cache reads; the next storage change must account for
  both the PLE direct-read path and its overlap with expert-cache I/O before selecting a scheduling
  mitigation.
  A sustained same-artifact overlap probe then ran the PLE reader at batch 8 × 4 positions for
  5,000 windows with `cache_mib=0` while `fio` issued direct random reads at QD1. Both earlier
  `fio` workloads targeted the artifact entry file (`qwen3_8_flash_next.ninfer`), while the PLE
  table backing uses continuation files; their no-error results were not a same-volume check. The
  512 KiB
  workload read 11.0 GiB in 10 s at 1.10 GiB/s; all PLE windows completed, but one took 15.76 s
  (median 1.917 ms). The 128 KiB workload read 4.45 GiB in 10 s at 455 MiB/s; all windows
  completed, with one 7.51 s outlier (median 1.914 ms). Neither QD1 run logged a new kernel NVMe
  timeout, despite these PLE stalls. Repeating the 512 KiB overlap at PLE depth 16 timed out in
  the PLE readiness wait after window 1,479 and logged an NVMe timeout; `fio` completed without
  errors. Thus lower PLE depth is not a reliable mitigation, and smaller QD1 reads did not remove
  the long tail. These runs show PLE stalls while the same SSD handles concurrent direct reads,
  but did not test contention on the same volume or reproduce the expert-cache access pattern. A
  follow-up PLE-only run (same batch/window/depth, no `fio`, cache disabled)
  completed windows 0–18 at 3.42–4.01 ms, then hit the 30 s PLE readiness timeout on the next
  window. The kernel logged another NVMe timeout at 23:14:12 (QID 1); SMART then reported 38 C,
  100% spare, 0% wear, and 0 media errors. The external concurrent reader is therefore not
  required to trigger the failure. The remaining evidence points to intermittent direct-read/device
  stalls, with aggregate I/O still a potential aggravating factor rather than an established cause.
  Added PLE timeout telemetry located a later 30 s stall on backing file 1, byte offset
  11,561,353,216; mapping the PLE shard order resolves this to
  `qwen3_8_flash_next.ninfer.part-0003`, object `weight/000222`, shard 124, page 6,444 (row
  290,001, head 15, batch row 7, token column 3 in sample 632). Replaying that exact input later
  completed in 2.38 ms. A 10 s registered-file/fixed-buffer `io_uring` QD128 random-read run on
  part-0003 completed 4.67 GiB at 478 MiB/s (mean completion 1.04 ms, max 7.76 ms); repeatedly
  reading the exact 4 KiB offset for 10 s also completed without FIO errors (20.1k IOPS, mean
  latency 6.37 ms, max 12.65 ms). Neither same-volume FIO run added a kernel timeout. The stalled
  page is therefore readable and the failure is intermittent; the custom PLE workload still
  differs from these FIO access patterns, and its failure mechanism remains open. The current host
  drive identifies as a Seagate FireCuda 530 (SU6SM001), PCIe Gen4 x4; PCIe AER reports no errors.
  SMART still reports zero media errors. Its nonzero NVMe error-log entries are `INVALID_FIELD`
  admin-command statuses (the remaining returned entries are successful), not a logged media or
  transport failure; they do not explain the kernel’s timed-out I/O requests.
  A controlled APST experiment toggled autonomous power-state transitions only for each run and
  restored the original enabled state afterward. With APST disabled, one matched 5,000-window run
  completed in 10.98 s (median 1.97 ms, p99 3.55 ms, max 4.21 ms); its repeat timed out at sample
  163 with one in-flight page at continuation-file index 1, offset 10,108,174,336, and a new kernel
  timeout. With APST enabled, a matched run timed out at sample 3,067 at file index 0, offset
  17,103,908,864, again with one page aged 29,998 ms and a new kernel timeout. The APST-off
  success is not reproducible evidence of a fix: both settings produced stalls, at different offsets.
  APST is therefore not sufficient to explain or prevent the storage failure.
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
