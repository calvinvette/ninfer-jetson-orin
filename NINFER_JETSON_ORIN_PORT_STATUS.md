# NInfer Jetson Orin Port Status

Status assessed: 2026-09-21.

Phases 0 through 2 are complete, and Phase 3 has been completed to the currently
qualified operator and real-model test scope. Phase 4 has started with initial
Orin performance samples; controlled baseline measurement and broader Phase 3
coverage remain open. The governing sequence and detailed inventory remain in
[the port plan](NINFER_JETSON_ORIN_PORT_PLAN.md).

## Phase status

| Phase | Current state | Remaining acceptance work |
| --- | --- | --- |
| 0 — Baseline and inventory | Source inventory complete; inherited SM86 behavior documented. | No fresh SM86 runtime baseline was established. Published validation is the baseline evidence currently available. |
| 1 — CUDA 12.6 compatibility | E2M1 decode backported and exactly qualified; complete CUDA 12.6/SM86 build passed. | Commit/document the floor change, preserve newer-toolkit support, and run the focused SM87-native checks. |
| 2 — Native aarch64 build | Native dependency setup, full aarch64 compilation, application help checks and focused runtime tests passed. | Keep the local dependency runtime path documented; no CPU portability fix has been needed. |
| 3 — SM87 correctness | Explicit architecture 87 configuration is enabled. The full native SM87 build passed 480/480 compile/link steps; the native device/runtime, CUDA Graph, E2M1 codec, 30 core scheduling/state tests, 21 supported operator/projection tests, real Qwen3.8-27B prefix integration, BF16 causal scoring, INT8-KV MTP generation and short CLI/MTP generation tests pass on Orin. A repeated 64-token CLI decode also reproduced identical output and MTP counters across two fresh processes. | The original FP8 causal-score fixture now skips cleanly on SM87 because that route has no supported implementation; BF16 scoring is the qualified Orin path. |
| 4 — Orin performance baseline | Initial text samples recorded with BF16 KV: MTP off 7.71, draft-2 12.86, draft-3 16.06 and draft-4 13.67 decode tok/s. A repeated 64-token draft-3 sample measured 12.75 tok/s on both runs with 59.70% acceptance and no fallbacks. Under MAXN, a synchronized 64-token draft-3 sample measured 12.71 decode tok/s, 36.82 prefill tok/s and 11.41 overall tok/s; tegrastats observed up to 99% GR3D utilization and 39.8 W VDD_GPU_SOC instantaneous power. Draft-3 remains the current candidate. | GPU clocks were not lockable from the unprivileged session; repeat with fixed clocks and a larger workload before publishing a final baseline. |
| 5 — SM87 schedule tuning | Not started. | Profile and tune only after correctness and baseline measurements. |
| 6 — Memory experiments | Not started. | Compare selected allocation classes against the existing device-allocation control. |
| 7 — Capacity/context tuning | Not started. | Establish system-memory headroom and qualified BF16/INT8 KV context limits. |
| 8 — Final qualification | Not started. | Compare llama.cpp, initial SM87 and tuned SM87 with controlled workloads and energy measurements. |

## Evidence retained from the interrupted work

The plan's execution-status section records the initial investigation. Those
historical observations are retained below for provenance; later milestones
supersede the initial “no full build” state.

- Host/toolchain: Linux aarch64, GCC 11.4, CMake 4.0.2 and explicitly selected
  `/usr/local/cuda-12.6/bin/nvcc` (CUDA 12.6.77).
- Initial configuration in `build/port-cuda126-sm86`, with architecture 86 and
  apps, tests and benchmarks enabled, passed compiler detection but stopped at
  the project's CUDA 12.8 minimum. This was the pre-backport result; the later
  CUDA 12.6 build completed successfully.
- A direct CUDA 12.6/SM86 compilation probe of
  `src/ops/linear/nvfp4/nvfp4_gemv.cu` failed on missing `cuda_fp4.h`.
  The supported A16 weight-decode route uses that header; disabling NVFP4
  weight loading would regress the existing behavior.
- Installed FFmpeg pkg-config versions were 58.76.100 / 58.134.100 /
  56.70.100 / 5.9.100, below the required 60 / 60 / 58 / 7 major versions.
  `libcurl.pc` was absent; the project requires libcurl >=7.85.
- The initial sandboxed `nvidia-smi` attempt failed with NvRm initialization/
  driver-access errors. An unrestricted check later succeeded and identified
  Orin; the sandbox error did not represent host GPU availability.
- The inherited README reports Windows v0.6.0 validation and the Linux
  v0.6.1 CUDA 13.1 compile/link gate (245 steps and application help).
  Linux real-artifact generation and performance remain unqualified there.

## Checkout verification at resumption (before implementation)

- Before this update, tracked files had no local changes; the port plan was
  untracked. HEAD was `75d94eab` (upstream-catchup merge), with no subsequent
  port implementation commit in the current branch.
- At the initial checkout, `CMakeLists.txt` defaulted to SM86, accepted only
  86/89 and required CUDA >=12.8. The existing `NINFER_SM8X_COMPAT` selection
  remains unchanged.
- `src/ops/linear/nvfp4/nvfp4_codec.cuh` still includes `cuda_fp4.h` and
  decodes through `__nv_fp4x2_e2m1`; the proposed backport is absent.
- The retained `build/port-cuda126-sm86/CMakeCache.txt` confirms the explicit
  CUDA 12.6 compiler, architecture 86 and enabled apps/tests/benchmarks.
  The directory contains the cache and CMake metadata, but no generated
  Ninja build file.

These checks corroborated the saved stopping point. Subsequent work is recorded below.

## Resume point

1. Completed: replace the CUDA FP4-header dependency with exact E2M1 decoding,
   check all 256 packed bytes and isolate unsupported A4 quantization from SM8x.
2. Resolve the native FFmpeg/libcurl build prerequisites without silently
   reducing the dependency contract. Establish the complete CUDA 12.6/SM86
   build using the configuration recorded in the plan and
   `cmake --build build/port-cuda126-sm86 -j`; qualify the toolkit-floor
   change with compatibility evidence.
3. Run available SM86 correctness/model checks, explicitly retaining any
   hardware-dependent validation gap. Finish the native-build gate before
   introducing SM87 execution changes.
4. For SM87, review cooperative-grid residency in particular: the inventory
   identifies an 82-SM minimum/fallback and fixed occupancy/shared-memory
   assumptions in the GDN planner. Complete numerical and real-model gates
   before performance tuning.

Keep explicit device allocation as the initial execution model. Memory-policy
experiments and schedule optimization remain later, separately verified phases.

## Resumed Phase 3 milestone — native SM87 compilation path

- CMake now accepts `CMAKE_CUDA_ARCHITECTURES=87` alongside the existing 86 and
  89 targets. The existing `NINFER_SM8X_COMPAT` capability path is enabled for
  SM87 as well, preserving the explicit rejection of Blackwell-only FP8/FP4
  tensor-core routes while retaining A16 decode and supported Ampere kernels.
- CUDA 12.6 native SM87 configuration completed on Orin's aarch64 environment.
  The focused native build produced device, CUDA Graph and NVFP4 codec tests.
- `ctest -R '^ninfer_(nvfp4_codec|device|decode_graph)_test$'` passed 3/3 on
  Orin. The codec comparison covers all 256 packed E2M1 bytes bit-for-bit,
  including signed zero. This is native SM87 evidence; it does not qualify the
  complete model schedule.
- The full SM87 project build passed. GDN cooperative-residency qualification is
  currently limited to the tested operator shapes. Existing planner catalogs
  were measured for 82–84-SM desktop GPUs. The
  planner now treats those boundaries as candidates and falls back to unsplit
  MMA when Orin's 16-SM resource budget cannot hold a cooperative grid.
- After that change, `ninfer_gdn_gating_proj_test` passed on native SM87,
  covering 27B/35B route boundaries, norm/control paths, independent numerical
  oracles and workspace contracts. This qualifies the planner fallback and
  operator behavior for its tested shapes; it is not yet model execution.
- The pinned Qwen3.8-27B groupwise artifact checksum passed exactly. The first
  model attempt happened during build/download pressure and saw only
  6,837,530,624 free bytes; after pressure cleared, the same artifact loaded and
  `ninfer_qwen3_6_27b_prefix_real_test` passed in 31.79 seconds with native SM87.
  That test exercises model admission/materialization, frontend, full prefill,
  state/checkpoint/prefix behavior and the configured MTP startup features. It
  is not a throughput measurement.
- The runtime capability gate now accepts SM87. The artifact's required weight
  allocation is approximately 17.9 GB; the Orin reported 26.25 GB free after
  the test and returned to that level after teardown.
- A short CLI smoke test with the pinned artifact, BF16 KV, greedy sampling and
  MTP draft window 2 generated `2 + 2 = **4**`. It completed 3 MTP rounds,
  drafted 5 tokens, accepted 4, and reported 80% acceptance (2.33 accepted
  tokens/round). This is a functional smoke result, not a performance baseline.
- Initial phase-4 sample used the same 23-token prompt, 32 generated tokens,
  greedy/no-thinking sampling, BF16 KV, explicit 2048-token capacity and native
  SM87. MTP off measured 37.34 prefill tok/s and 7.71 decode tok/s; MTP draft-2
  measured 36.24 prefill tok/s and 12.86 decode tok/s with 82.61% acceptance
  (2.58 accepted tokens/round). Weight materialization was 8.9–9.1 seconds and
  used 15.92–16.67 GiB. These are one-device smoke measurements, not a published
  performance claim; power mode and clocks were not fixed.
- The same sample at draft-3 measured 36.01 prefill tok/s, 16.06 decode tok/s,
  12.46 overall tok/s and 88.00% acceptance (3.44 accepted tokens/round), with
  no fallback steps. Draft-4 measured 35.62 prefill tok/s, 13.67 decode tok/s,
  10.98 overall tok/s and 68.75% acceptance, with one fallback step. The
  observed draft-3 advantage is a workload sample, not enough evidence to change
  a product default.
- A repeated fresh-process CLI run used the pinned artifact, the same 24-token
  prompt, 64 generated tokens, greedy/no-thinking sampling, BF16 KV and MTP
  draft-3. Both runs produced identical text, 23 MTP rounds, 67 drafted tokens,
  40 accepted tokens, 59.70% acceptance, no fallbacks and 12.75 decode tok/s.
  This supports repeated decode stability for this route; it is not a controlled
  power or clock-normalized performance baseline.
- The native SM87 operator and schedule sweep passed 51/51 selected tests: 30
  core scheduling/state/attention/MTP tests and 21 supported quantized linear,
  projection and GDN replay tests. The sweep excludes routes explicitly gated
  out on SM87, including FP8 linear/attention and A4 NVFP4 quantization.
- A real-model CLI smoke with the pinned artifact, greedy/no-thinking sampling,
  MTP draft-2 and INT8 group64 KV generated `2 + 2 = **4**`; it completed three
  rounds with 83.33% acceptance (2.67 accepted tokens/round). This qualifies the
  currently supported INT8 KV route on SM87.
- The controlled Phase 4 sample ran under the reported `MAXN` power mode with
  the existing dynamic clock policy. It used the pinned artifact, a 24-token
  prompt, 64 generated tokens, BF16 KV and MTP draft-3. The run measured 36.82
  prefill tok/s, 12.71 decode tok/s and 11.41 overall tok/s, with 59.70%
  acceptance and no fallback steps. `tegrastats` observed 99% GR3D utilization;
  GPU clock locking was unavailable without elevated Jetson privileges, so this
  is a controlled-power-mode sample rather than a final clock-normalized result.
- An Nsight Systems CUDA trace of the short BF16/MTP route identified the
  quantized Q4/Q5 GEMM families as the dominant GPU kernel-time contributors,
  with GDN recurrent kernels and attention below them. This is Phase 5 triage
  evidence only; no SM87 schedule has been changed from the qualified
  implementation yet.
- The causal-score real test was attempted with the pinned artifact and reached
  the runtime correctly, but its fixture requests FP8 E4M3 KV. The runtime
  rejects that storage on SM87 because the FP8 causal-attention implementation
  is limited to newer architectures; the fixture now reports a clean skip on
  Orin. A companion BF16 fixture runs the same 1,537-token overlap and repeated
  window checks; it passed on native SM87 in 29.70 seconds with finite scores,
  overlap error within the 0.25 tolerance and no state/KV contamination.

## Artifact acquisition milestone

- Downloaded the documented Qwen3.8-27B groupwise NInfer artifact to
  `/home/calvin/models/qwen3_8_27b.ninfer`. The completed file is
  20,437,521,664 bytes. Its expected SHA-256 is recorded in
  `model-cards/Qwen3.8-27B-NInfer/SHA256SUMS`; checksum verification is running
  before model loading.
- The source was selected by the exact documented model identity,
  `neroued/Qwen3.8-27B-NInfer/qwen3_8_27b.ninfer`, rather than by filename or
  modification time. The current `main` artifact is 20,437,521,664 bytes with
  SHA-256 `81f924d440c27261d820c19a9f8d45794c5aee410f8a68bd358133fa8c0375da`,
  while this checkout's pinned v1 manifest expects 18,210,531,328 bytes and
  SHA-256 `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.
  The current main artifact is container v3 and is not accepted as the pinned
  runtime artifact. The matching historical revision
  `3526913004b1cf552cb57b88d6a5c6f5e4a89a70` is now downloading into
  `/home/calvin/models/qwen3_8_27b_v1`; it will be checksum-verified before use.
  The GGUF files under `~/models` remain reference material for llama.cpp only
  and are not passed to NInfer.

## Resumed Phase 1 milestone — exact E2M1 decode

- Replaced the CUDA FP4 type conversion with direct exact FP32 bit construction.
  A16 NVFP4 weight support is preserved; A4-only quantization helpers are excluded
  under the existing SM8x capability definition. No architecture or schedule changed.
- Added `ninfer_nvfp4_codec_test`, which executes the production device decoder
  for every packed byte and compares both FP32 outputs bit-for-bit with the
  existing independent test-owned E2M1 magnitude-table oracle. Signed zeros and
  nibble ordering are included.
- CUDA 12.6.77: standalone test compiled with C++20, `-arch=sm_86` and
  `NINFER_SM8X_COMPAT=1`; all 256 bytes passed on Orin. The previously failing
  `nvfp4_gemv.cu` also compiled successfully with the same profile.
- CUDA 12.9: the standalone decoder test compiled successfully. This is focused
  newer-toolkit compile evidence, not yet a full-project compatibility gate.
- Unrestricted `nvidia-smi` succeeds and identifies Orin, driver 540.4.0, CUDA
  12.6. Earlier GPU-access errors were caused by sandbox device isolation. GPU
  tests require execution outside that sandbox. Running SM86 code on Orin is
  evidence for this codec only, not SM86 hardware regression or SM87 qualification.
- Local FFmpeg 6.1.2 and curl 8.10.1 dependency builds completed under
  `build/jetson-deps`; system packages remain unchanged. The available Ubuntu
  packages (FFmpeg 4.4 and curl 7.81) do not meet the current project minimums.
- The committed toolkit floor remains 12.8. A candidate reduction to 12.6 is
  being exercised in the working tree and will be committed only with successful
  complete-build evidence.

## Native build prerequisites milestone

- Added [native Jetson development instructions](docs/jetson-orin.md), including
  explicit CUDA selection and an isolated dependency prefix. The desktop Docker
  image and system JetPack libraries are unchanged.
- CMake successfully configured the complete project with apps, tests and
  benchmarks enabled: Linux aarch64, GCC 11.4, CUDA 12.6.77, architecture 86,
  Python 3.11.12, FFmpeg 6.1.2 and curl 8.10.1.
- `cmake --build build/port-cuda126-sm86 -j` completed all 614 compile/link
  steps. CLI, server and perplexity `--help` checks passed. Device, CUDA Graph,
  arena, public API, wide-math, artifact-reader, and media-decode tests passed;
  media-decode required `LD_LIBRARY_PATH` to include the local dependency prefix.
- The real-shape `ninfer_linear_nvfp4_a16_test` was stopped after 11 minutes of
  PTX-JIT activity with zero GPU utilization. An SM86 binary is not a valid
  native Orin qualification: with `CUDA_DISABLE_PTX_JIT=1` it fails with “PTX
  JIT compilation was disabled”. Native SM87 compilation is required for the
  next GPU qualification step.
- The user clarified that `~/.ninfer` is a placeholder configuration file, not
  model storage. The actual model root is `~/models` (`/home/calvin/models`).
  That directory contains downloaded GGUF files, including:
  `qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17,559,178,144 bytes),
  `qwen3.8-27b-iq3s/Qwen3.8-27B-UD-IQ3_S.gguf` (12,040,883,104 bytes),
  `qwen3.8-27b/MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1,369,590,656 bytes), and
  `Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`
  (18,536,192,288 bytes). No `.ninfer` artifact is present under `~/models`.
  These GGUF files are not interchangeable with NInfer's registered `.ninfer`
  artifacts. A compatible Qwen3.8-27B groupwise artifact is now being downloaded
  to `/home/calvin/models/qwen3_8_27b.ninfer` from the documented
  `neroued/Qwen3.8-27B-NInfer` repository. Real-model gates remain pending the
  completed checksum and native SM87 build.
