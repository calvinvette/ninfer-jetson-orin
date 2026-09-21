# NInfer Jetson Orin Port Status

Status assessed: 2026-09-21.

Work has resumed in Phase 1. The CUDA 12.6 E2M1 decoder backport is implemented
and exhaustively checked on the Orin GPU using SM86 code. **The complete CUDA
12.6 build and explicit SM87 execution target remain unqualified.** The governing sequence
and detailed inventory remain in [the port plan](NINFER_JETSON_ORIN_PORT_PLAN.md).

## Phase status

| Phase | Current state | Remaining acceptance work |
| --- | --- | --- |
| 0 — Baseline and inventory | Source inventory complete; inherited SM86 behavior documented. | No fresh SM86 runtime baseline was established. Published validation is the baseline evidence currently available. |
| 1 — CUDA 12.6 compatibility | E2M1 decode backported and exactly qualified; complete CUDA 12.6/SM86 build passed. | Commit/document the floor change, preserve newer-toolkit support, and run the focused SM87-native checks. |
| 2 — Native aarch64 build | Native dependency setup, full aarch64 compilation, application help checks and focused runtime tests passed. | Keep the local dependency runtime path documented; no CPU portability fix has been needed. |
| 3 — SM87 correctness | Not started. Architecture 87 is still rejected. Orin reports SM87, 16 SMs, 167,936 bytes shared memory/SM and 65,536 registers/SM. | Explicit architecture support, capability/residency review, operator qualification and compatible `.ninfer` model gates. |
| 4 — Orin performance baseline | Not started. | Measure the qualified explicit-device-memory implementation, including MTP off/2/3/4 and power/clock context. |
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
  artifacts, so real NInfer model loading, generation, MTP and performance gates
  remain pending a compatible `.ninfer` file. No model download or conversion was
  initiated.
