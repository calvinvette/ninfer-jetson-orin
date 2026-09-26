# NInfer Jetson Orin Experiment Ledger

Status assessed: 2026-09-26. All performance results below use the native SM87
CUDA 12.6 build on Jetson AGX Orin. Fixed-clock entries use MAXN plus
`jetson_clocks`: 1,300.5 MHz GPU, 2,201.6 MHz CPU, and EMC override.

| Test | State | Route / shape | Result | Criterion / outcome |
| --- | --- | --- | --- | --- |
| Qwen3.6-35B-A3B current v3 artifact | ❌ Incompatible | Publisher revision `ee449580`; 22,790,484,480-byte v3 `.ninfer` | Rejected at reader framing | This port implements only v1/v2 framing; v3 uses an expanded header and is not safely interchangeable. |
| Qwen3.6-35B-A3B pinned v2 artifact | ✅ Complete | Publisher revision `3c739ac9`; SHA-256 `1fb9…325d2`; 21.22 GiB | Loads as `qwen3_6_35b_a3b/groupwise-int` | Exact compatible artifact for this port; 21.04 GiB of weights resident at the short control. |
| 35B-A3B short BF16 MTP matrix | ✅ Complete | `pp512+tg64`, off/2/3/4, 3 reps | 37.50 / **43.63** / 34.34 / 29.41 decode tok/s | MTP-2 wins; acceptance n/a / 56.67% / 34.41% / 26.02%. |
| 35B-A3B short INT8 MTP matrix | ✅ Complete | `pp512+tg64`, off/2/3/4, 3 reps | 37.45 / **42.16** / 36.50 / 32.31 decode tok/s | MTP-2 wins; acceptance n/a / 53.23% / 38.64% / 30.97%. |
| 35B-A3B long BF16 MTP matrix | ✅ Complete | `pp2048+tg128`, off/2/3/4, 3 reps | 36.68 / 56.59 / 60.82 / **62.91** decode tok/s | Acceptance n/a / 94.32% / 89.42% / 87.61%; draft-4 wins. |
| 35B-A3B long INT8 MTP matrix | ✅ Complete | `pp2048+tg128`, off/2/3/4, 3 reps | 36.83 / 55.60 / 61.11 / **62.31** decode tok/s | Acceptance n/a / 91.11% / 89.42% / 87.61%; draft-4 wins. |
| 35B-A3B BF16 capacity gates | ✅ Complete | eager `pp8192/16384/32768+tg1`, 1.2 GiB guard | 1,237.42 / 1,221.02 / 1,132.53 prefill tok/s | KV payload 0.16 / 0.31 / 0.63 GiB; minimum host availability 7.10 / 6.93 / 6.59 GiB. |
| 35B-A3B INT8 capacity gates | ✅ Complete | eager `pp8192/16384/32768+tg1`, 1.2 GiB guard | 1,204.29 / 1,155.05 / 1,028.17 prefill tok/s | KV payload 0.08 / 0.16 / 0.32 GiB; minimum host availability 7.09 / 7.00 / 6.83 GiB. |
| Fixed-clock BF16 MTP matrix | ✅ Complete | `pp512+tg64`, MTP off/2/3/4 | 7.68 / 10.60 / 10.98 / 9.39 decode tok/s | Draft-3 wins; acceptance 58.62% / 47.44% / 35.92% for draft-2/3/4. |
| Fixed-clock INT8 MTP matrix | ✅ Complete | `pp512+tg64`, MTP off/2/3/4 | 7.70 / 10.26 / 11.00 / 9.05 decode tok/s | Draft-3 wins; acceptance 54.10% / 47.44% / 33.64%. |
| Long BF16 MTP control | ✅ Complete | `pp2048+tg128`, off vs draft-3 | 7.62 vs 17.21 decode tok/s | Draft-3 acceptance 93.07%, no fallback steps. |
| Long INT8 MTP control | ✅ Complete | `pp2048+tg128`, off vs draft-3 | 7.63 vs 17.26 decode tok/s | Draft-3 acceptance 93.07%, no fallback steps. |
| 32K INT8 capacity gate | ✅ Complete | eager 32,768-token prefill plus one generated token | 209.03 prefill tok/s; 1.03 GiB KV | Completed with retained startup headroom. |
| 32K BF16 capacity gate | ✅ Complete | eager 32,768-token prefill plus one generated token | 211.50 prefill tok/s; 2.00 GiB KV | Completed with retained startup headroom. |
| Guarded 40,960 INT8 capacity gate | ❌ Pressure limited | eager `pp40960+tg1`, 1.2 GiB host-memory floor | stopped at 1.063 GiB `MemAvailable` | The retry reached benchmark dispatch but was terminated by the guard before a valid result. |
| llama.cpp reference | ✅ Complete | Qwen3.8-27B Q4_K_XL, CUDA, FP16 KV, `pp512+tg64` | 238.15 ± 4.21 prefill; 7.78 ± 0.06 decode tok/s | Matched model family only; GGUF quantization/KV differ from NInfer. |
| llama.cpp Q4_K_M reference | ✅ Complete | Qwen3.8-27B UD-Q4_K_M, CUDA, FP16 KV, `pp512+tg64` | 252.06 prefill; 8.434 decode tok/s | Full offload/FlashAttention; matched-family conventional-Q4 reference, not groupwise-int or INT8-KV equivalent. |
| Reconstructed initial-SM87 route | ✅ Complete | historical R32C64S4, INT8 KV/draft-3, fixed-clock `pp512+tg64`, no prefix reuse | 235.01 prefill; 10.976 decode tok/s | Exact historical schedule rebuilt, focused test passed, and end-to-end comparison completed. |
| Q4 SwiGLU C128 control | ✅ Complete | T=1024 cold-cache public Op | 28.40–28.53 ms | Selected large-prefill schedule. |
| Q4 SwiGLU C64 tile | ❌ Rejected | T=1024 cold-cache public Op | 42.44 ms | Oracle passed; ~49% slower than C128. Removed. |
| Q4 SwiGLU C96 tile | ❌ Rejected | T=1024 cold-cache public Op | 38.21 ms | Oracle passed; ~34% slower than C128. Removed. |
| Q4 SwiGLU C128 `ca` loads | ❌ Rejected | T=1024 public Op | 29.435 ms | Oracle passed; slower than the 28.40–28.53 ms `cg` control. Removed. |
| Q5 GDN-output C128 control | ✅ Complete | N=6144, K=5120, T=1024 public Linear | 5.008–5.019 ms | Selected large-prefill schedule. |
| Q5 GDN-output C64 tile | ❌ Rejected | N=6144, K=5120, T=1024 public Linear | 7.271 ms | Oracle passed; ~45% slower than C128. Removed. |
| Q5 GDN-output C128 `ca` loads | ❌ Rejected | N=6144, K=5120, T=1024 public Linear | 6.075 ms | Oracle passed; ~21% slower than the `cg` control. Removed. |
| Q5 GDN-output C128 scalar scales | ❌ Rejected | N=6144, K=5120, T=1024 public Linear | 5.202 ms | Oracle passed; 3.9% slower than Pair32 scales. Removed. |
| Q5 GDN-output C128 ping-pong fragments | ❌ Rejected | N=6144, K=5120, T=1024 public Linear | 5.041 ms | Oracle passed; did not beat the 5.038 ms serial control. Removed. |
| Q4/Q5 GDN-input R64C128 control | ✅ Complete | mixed Q4/Q5 GDN input, T=1024, cold public Op | 14.494 ms | Exact owner of the 15.3% grouped-MMA trace contributor. |
| Q4/Q5 GDN-input R64C64 tile | ❌ Rejected | mixed Q4/Q5 GDN input, T=1024, cold public Op | 17.394 ms | Oracle passed; 20.0% slower than R64C128. Removed. |
| Q4/Q5 GDN-input R32C128 tile | ❌ Rejected | mixed Q4/Q5 GDN input, T=1024, cold public Op | 17.345 ms | Oracle passed; 19.7% slower than R64C128. Removed. |
| Q4/Q5 attention-input R32C64S4 control | ✅ Complete | mixed Q4/Q5 attention input, T=1024, cold public Op | 21.648 ms | Trace owner for the paired Q4 (4.0%) and Q5 (3.6%) grouped-MMA kernels. |
| Q4/Q5 attention-input R32C128S2 | ✅ Complete | mixed Q4/Q5 attention input, T=1024, cold public Op | 19.448 ms | Oracle passed; 10.1% faster than the original control; later superseded. |
| Q4/Q5 attention-input R16C128S2 | ❌ Rejected | mixed Q4/Q5 attention input, T=1024, cold public Op | 20.867 ms | Oracle passed; 7.3% slower than R32C128S2. Removed. |
| Q4/Q5 attention-input R32C64S3 | ❌ Rejected | mixed Q4/Q5 attention input, T=1024, cold public Op | 20.355 ms | Oracle passed; 4.7% slower than R32C128S2. Removed. |
| Q4/Q5 attention-input R64C64S3 | ✅ Complete | mixed Q4/Q5 attention input, T=1024, cold public Op | 16.236 ms | Oracle passed; later superseded by R64C128S2. |
| Q4/Q5 attention-input R64C128S2 | ✅ Selected | mixed Q4/Q5 attention input, T=1024, cold public Op | 13.570 ms | Oracle passed; 37.3% faster than original R32C64S4 control. |
| Q4/Q5 attention-input R128C64S3 | ❌ Not admitted | mixed Q4/Q5 attention input, T=1024 | No binary | Compile-time shared-memory check rejected its >48 KiB staging footprint. |
| Q4/Q5 attention-input R128C64S2 | ❌ Rejected | mixed Q4/Q5 attention input, T=1024, cold public Op | 15.592 ms | Oracle passed; 14.9% slower than R64C128S2. Removed. |
| Q4/Q5 attention-input R64C128S2 `ca` loads | ❌ Rejected | mixed Q4/Q5 attention input, T=1024, cold public Op | 13.998 ms | Oracle passed; 3.2% slower than the `cg` control. Removed. |
| Q4/Q5 attention-input R64C128S2 individual scales | ❌ Rejected | mixed Q4/Q5 attention input, T=1024, cold public Op | 16.778 ms | Oracle passed; 23.6% slower than paired-scale staging. Removed. |
| End-to-end validation of Q5 candidates | ✅ Not required | fixed-clock INT8 draft-3 `pp2048+tg128` | No Q5 winner | Every qualified Q5 variant lost to the selected serial/Pair32/C128 route. |
| End-to-end validation of attention-input winner | ⚠️ Pressure limited | fixed-clock INT8 `pp2048+tg128` | No new result | Model setup crosses the retained 1.2 GiB host-memory floor before prefill. |
| Map Q4/Q5 grouped-rowsplit trace | ✅ Complete | 15.3% of fixed-clock draft-3 GPU kernel time | GDN input R64C128 mixed-MMA | Exact `5120 -> {4096,6144,6144}` public projection at T=1024. |
| Long BF16 MTP-window matrix | ✅ Complete | `pp2048+tg128`, draft-2 and draft-4 | 13.95 / 17.86 decode tok/s | Acceptance 94.32% / 91.74%; draft-4 is the BF16 winner at this workload. |
| Long INT8 MTP-window matrix | ⚠️ Pressure limited | `pp2048+tg128`, draft-2 and draft-4 | No additional result | The 1.2 GiB guarded draft-2 retry stopped during setup at 0.574 GiB `MemAvailable`; draft-4 was not started because setup never reached the draft window. |
| Safe context expansion | ⚠️ Pressure limited | INT8 capacity above 32K | No additional point | The 1.2 GiB guarded 40,960-token retry reached dispatch but stopped at 1.063 GiB `MemAvailable` before a valid result. |
| Jetson allocation-class trial | ✅ Complete | memory-pool class vs explicit `cudaMalloc` | Explicit remains selected | The new opt-in stream-ordered class is qualified below; no end-to-end evidence supports replacing the explicit control. |
| Stream-ordered allocation class | ✅ Qualified, not selected | `DeviceBuffer`/`DeviceArena`, `cudaMallocAsync`/`cudaFreeAsync` | Arena transfer and suballocation test passes | Orin supports the candidate; explicit allocation remains the Engine default because no end-to-end gain has been demonstrated. |

## Completed tests

### Qwen3.6-35B-A3B MoE matrix (2026-09-26)

The upstream current release is a verified 22,790,484,480-byte v3 artifact,
but its framing is newer than this port's v1/v2 reader. It was retained at
`~/models/qwen3_6_35b_a3b/` and correctly rejected before model loading. The
matrix therefore pins the publisher's v2 release `3c739ac9`, stored at
`~/models/qwen3_6_35b_a3b_v2/qwen3_6_35b_a3b.ninfer`; its SHA-256 is
`1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2`.

All twenty throughput points use MAXN plus `jetson_clocks`, CUDA 12.6,
SM87, CUDA Graph decode, no prefix reuse, a 2,048-token short capacity or
4,096-token long capacity, one warm-up, and three measured repetitions. The
six capacity gates use eager execution, one repetition, and the 1.2 GiB
host-memory guard. Raw JSON and guard telemetry are under
`profiles/bench/jetson_orin_35b_a3b/`. At the short
workload, MTP-2 is the best configuration: 1,178.62 ± 0.13 prefill tok/s and
43.63 ± 0.01 decode tok/s with BF16 KV, or 1,170.97 ± 0.19 / 42.16 ± 0.01
with INT8 KV. At the long workload, draft-4 wins: BF16 reaches
1,359.73 ± 0.09 prefill and 62.91 ± 0.01 decode tok/s; INT8 reaches
1,344.74 ± 0.11 and 62.31 ± 0.01. The MTP-off long controls are 36.68 BF16
and 36.83 INT8 decode tok/s. All 8K, 16K, and 32K BF16/INT8 capacity gates
completed without host-pressure termination. These are MTP results only; the
MoE artifact's separate text-only DFlash backend has not been included in this
same MTP matrix.

The fixed-clock matrices establish draft-3 as the current best MTP window for
both qualified KV formats. At 64 output tokens it reaches roughly 11 decode
tok/s; at the longer 128-token control its high acceptance rate raises decode
throughput to roughly 17.2 tok/s. INT8 halves the reserved KV payload at the
short control without a measured decode penalty.

Both 32K capacity gates completed. The guarded 40K INT8 attempt deliberately
stopped before host pressure could destabilize the board; its retained telemetry
is evidence of a current practical pressure limit, not an inference result.

The Q4 and Q5 tile experiments all retained numerical correctness but lost
substantially to the existing C128 large-prefill schedules. Their project-owned
paths were removed rather than retained as dormant alternatives.

The next trace contributor was the mixed Q4/Q5 GDN input projection, not a
standalone Q4 linear route. Its selected R64C128 grouped-MMA schedule measures
14.494 ms at the traced 1024-token geometry. Both feasible tile changes passed
the independent Op oracle but lost by about 20%, so the selected implementation
remains unchanged.

The following traced pair belongs to the Q4/Q5 attention input projection. Its
final selected R64C128S2 route measures 13.570 ms at T=1024, versus 21.648 ms
for the former R32C64S4 route. The 37.3% improvement passed the public Op's
numerical test and replaced the superseded R32C128S2 schedule with matching
schedule diagnostics. The stream-ordered allocation class also passed the core
arena gate on Orin; explicit allocation remains selected without an end-to-end
advantage or a CUDA-Graph-stability reason to change the Engine policy.

## Final experiment notes

The Q5 `ca` experiment isolated cache policy while preserving its mathematical
route and CTA geometry. It passed the operator oracle but lost to the streaming
load control, so the experiment was removed without an end-to-end run.

The long INT8 draft-2 retry was repeated after lowering the active guard to
1.2 GiB. It still stopped before prefill, at 0.574 GiB during model setup after
10.26 seconds. The existing INT8 draft-3 control remains valid; no new draft-2
or draft-4 result was produced. The sub-floor sample shows that 250 ms polling
cannot keep a rapidly consuming setup exactly at the configured threshold.

After external services were reduced, the 1.2 GiB guarded selected-route retries
were repeated with 24 GiB idle `MemAvailable`. Long INT8 draft-2 still stopped
during setup at 1.167 GiB after 10.02 seconds, and the 40,960-token gate reached
dispatch but stopped at 1.075 GiB after 9.76 seconds. Neither produced a valid
benchmark result. The reconstructed initial-SM87 route differs only after the
same shared model setup, so it was not rebuilt merely to repeat that setup abort.

Phase 7 therefore qualifies 32,768 prompt tokens plus one generated token as
the supported long-context point for both BF16 and INT8 KV on this host. The
next tested INT8 point, 40,960 prompt tokens, is pressure-limited during setup
under the retained 1.2 GiB host-memory floor. It is not a failed inference run or
a measured maximum-context value, and the floor must not be lowered to extend
the result.

## Phase 8 comparison scope

| Endpoint | PP tok/s | TG tok/s | TTFT | Memory reservation | Max context | VDD_GPU_SOC | PP tok/J | TG tok/J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NInfer initial SM87 R32C64S4 | 235.01 | 10.976 | 2.179 s | 16.67 GiB weights + 0.44 GiB runtime | 32,768 qualified | 40.18 W | 5.849 | 0.273 |
| NInfer tuned SM87 R64C128S2 | 242.28 | 10.981 | 2.113 s | 16.67 GiB weights + 0.44 GiB runtime | 32,768 qualified | 41.38 W | 5.855 | 0.265 |
| llama.cpp UD-Q4_K_M | 252.06 | 8.434 | 2.031 s prefill proxy | 15.32 GiB GGUF file; runtime allocation not exposed | 2,048 benchmark capacity | 41.59 W | 6.061 | 0.203 |

All NInfer rows use fixed MAXN/`jetson_clocks`, CUDA Graph decode, INT8 group-64 KV, MTP draft-3,
full 27B groupwise-int artifact, three measured repetitions after one warmup, and
`--no-prefix-reuse`. The cache switch removes the otherwise unused 8 GiB host-KV reservation; it
does not change inference math. TTFT is prepare plus prefill time. Power is the mean
`VDD_GPU_SOC` rail reading during the measured window; tokens/J is throughput divided by that rail
power, not an estimate of whole-board input efficiency. llama.cpp is the user-approved
matched-family conventional-Q4 reference with full CUDA offload, FlashAttention, and FP16 KV; it
does not support NInfer's MTP/KV/weight-format equivalence claim.

The initial-SM87 endpoint was reconstructed from the exact historical R32C64S4
schedule and passes `ninfer_attn_input_proj_test`; the selected R64C128S2 source
was restored after comparison. The standalone benchmark's `--no-prefix-reuse`
mode removes only unused context-cache storage, allowing both endpoints to
complete safely under the retained 1.2 GiB guard.
