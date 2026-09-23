# NInfer Jetson Orin Experiment Ledger

Status assessed: 2026-09-23. All performance results below use the native SM87
CUDA 12.6 build on Jetson AGX Orin. Fixed-clock entries use MAXN plus
`jetson_clocks`: 1,300.5 MHz GPU, 2,201.6 MHz CPU, and EMC override.

| Test | State | Route / shape | Result | Criterion / outcome |
| --- | --- | --- | --- | --- |
| Fixed-clock BF16 MTP matrix | ✅ Complete | `pp512+tg64`, MTP off/2/3/4 | 7.68 / 10.60 / 10.98 / 9.39 decode tok/s | Draft-3 wins; acceptance 58.62% / 47.44% / 35.92% for draft-2/3/4. |
| Fixed-clock INT8 MTP matrix | ✅ Complete | `pp512+tg64`, MTP off/2/3/4 | 7.70 / 10.26 / 11.00 / 9.05 decode tok/s | Draft-3 wins; acceptance 54.10% / 47.44% / 33.64%. |
| Long BF16 MTP control | ✅ Complete | `pp2048+tg128`, off vs draft-3 | 7.62 vs 17.21 decode tok/s | Draft-3 acceptance 93.07%, no fallback steps. |
| Long INT8 MTP control | ✅ Complete | `pp2048+tg128`, off vs draft-3 | 7.63 vs 17.26 decode tok/s | Draft-3 acceptance 93.07%, no fallback steps. |
| 32K INT8 capacity gate | ✅ Complete | eager 32,768-token prefill plus one generated token | 209.03 prefill tok/s; 1.03 GiB KV | Completed with retained startup headroom. |
| 32K BF16 capacity gate | ✅ Complete | eager 32,768-token prefill plus one generated token | 211.50 prefill tok/s; 2.00 GiB KV | Completed with retained startup headroom. |
| Guarded 40,960 INT8 capacity gate | ❌ Pressure limited | eager `pp40960+tg1`, 2 GiB host-memory floor | stopped at 1.49 GiB `MemAvailable` | No prefill occurred; not a capacity pass or failure classification. |
| llama.cpp reference | ✅ Complete | Qwen3.8-27B Q4_K_XL, CUDA, FP16 KV, `pp512+tg64` | 238.15 ± 4.21 prefill; 7.78 ± 0.06 decode tok/s | Matched model family only; GGUF quantization/KV differ from NInfer. |
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
| End-to-end validation of attention-input winner | ⚠️ Pressure limited | fixed-clock INT8 `pp2048+tg128` | No new result | Model setup crosses the retained 2 GiB host-memory floor before prefill. |
| Map Q4/Q5 grouped-rowsplit trace | ✅ Complete | 15.3% of fixed-clock draft-3 GPU kernel time | GDN input R64C128 mixed-MMA | Exact `5120 -> {4096,6144,6144}` public projection at T=1024. |
| Long BF16 MTP-window matrix | ✅ Complete | `pp2048+tg128`, draft-2 and draft-4 | 13.95 / 17.86 decode tok/s | Acceptance 94.32% / 91.74%; draft-4 is the BF16 winner at this workload. |
| Long INT8 MTP-window matrix | ⚠️ Pressure limited | `pp2048+tg128`, draft-2 and draft-4 | No additional result | A guarded draft-2 retry stopped during model setup at 1.67 GiB `MemAvailable`; draft-4 was not started. |
| Safe context expansion | ⚠️ Pressure limited | INT8 capacity above 32K | No additional point | The guarded 40,960-token point stopped before prefill at 1.49 GiB `MemAvailable`; do not lower the 2 GiB safety floor. |
| Jetson allocation-class trial | ✅ Complete | memory-pool class vs explicit `cudaMalloc` | Explicit remains selected | The new opt-in stream-ordered class is qualified below; no end-to-end evidence supports replacing the explicit control. |
| Stream-ordered allocation class | ✅ Qualified, not selected | `DeviceBuffer`/`DeviceArena`, `cudaMallocAsync`/`cudaFreeAsync` | Arena transfer and suballocation test passes | Orin supports the candidate; explicit allocation remains the Engine default because no end-to-end gain has been demonstrated. |

## Completed tests

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
for the former R32C64S4 route. The 25.0% improvement passed the public Op's
numerical test and replaced the superseded R32C128S2 schedule with matching
schedule diagnostics. The stream-ordered allocation class also passed the core
arena gate on Orin; explicit allocation remains selected without an end-to-end
advantage or a CUDA-Graph-stability reason to change the Engine policy.

## Final experiment notes

The Q5 `ca` experiment isolated cache policy while preserving its mathematical
route and CTA geometry. It passed the operator oracle but lost to the streaming
load control, so the experiment was removed without an end-to-end run.

The guarded long INT8 draft-2 retry similarly stopped before prefill: its 2 GiB
host-memory floor was crossed at 1.67 GiB during model setup, and memory
recovered immediately after termination. The existing INT8 draft-3 control
remains valid; no new draft-2 or draft-4 result was produced.

Phase 7 therefore qualifies 32,768 prompt tokens plus one generated token as
the supported long-context point for both BF16 and INT8 KV on this host. The
next tested INT8 point, 40,960 prompt tokens, is pressure-limited during setup
under the retained 2 GiB host-memory floor. It is not a failed inference run or
a measured maximum-context value, and the floor must not be lowered to extend
the result.
