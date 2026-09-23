# NInfer on Jetson AGX Orin

This repository is the Jetson port of NInfer: a native Arm64/aarch64 C++20/CUDA inference
engine for Qwen3-family `.ninfer` artifacts on NVIDIA Jetson AGX Orin. The port exists because
the upstream [`ninfer-3090`](https://github.com/Don-Chad/ninfer-3090) project targets a discrete
RTX 3090 and its SM86 execution and memory assumptions do not describe Orin.

The supported platform is Ubuntu on Jetson AGX Orin:

| Platform | CUDA | GPU target | Status |
|---|---:|---:|---|
| JetPack 6.2 | 12.6 | SM87 (`CMAKE_CUDA_ARCHITECTURES=87`) | Native qualification platform |
| JetPack 7.1 | 13.1 | SM87 (`CMAKE_CUDA_ARCHITECTURES=87`) | Supported build target; repeat native qualification on the installed image |

The code is deliberately native aarch64, and this README documents only the Jetson Ubuntu path.

## Why this port is different

The upstream RTX 3090 implementation is compiled and tuned for **SM86** (GA102), a discrete GPU
with dedicated device memory. Orin is **SM87** (GA10B), with 16 SMs and physically unified LPDDR
shared by the CPU and GPU. SM86 and SM87 are related Ampere architectures, but they are different
execution targets: launch residency, shared-memory/register budgets, device properties, and
supported tensor-core routes must be qualified on the actual GPU.

Physical memory being shared does not make every CUDA allocation a Unified Memory allocation.
This port keeps the existing explicit `cudaMalloc` device-allocation model as its correctness and
performance control. `cudaMallocManaged`, mapped host memory, and other allocation strategies are
separate experiments; they are not enabled globally merely because Orin has unified physical
memory. CUDA Graph address stability, page residency, synchronization, and CPU access behavior
still depend on the allocation type.

The port therefore makes the following boundaries explicit:

- CUDA 12.6 is the minimum tested toolkit, while CUDA 13.1 remains a supported newer toolkit.
- SM87 is an explicit architecture target; it is not silently treated as SM86.
- BF16 and INT8 KV-cache routes are qualified on Orin. FP8 E4M3 causal attention is unavailable
  on SM87 and reports a clean unsupported-route skip.
- NVFP4 weight decoding remains available through the A16 dequantizing route. A4 NVFP4 and other
  Blackwell-only tensor-core routes are not emulated on Orin.
- Cooperative schedules use actual Orin residency information and fall back to a legal unsplit
  route when a desktop-sized cooperative grid cannot fit its 16-SM budget.

## Current qualification state

Phases 0–2 of the [port plan](NINFER_JETSON_ORIN_PORT_PLAN.md) are complete. Phase 3 SM87
correctness is qualified for the tested operator and real-model scope, including native build,
CUDA Graphs, NVFP4 codec, Q4/Q5/W8/BF16 routes, GDN paths, real-model prefix/state behavior,
BF16 causal scoring, INT8-KV MTP generation, and repeated decode stability.

Phase 4 SM87 experiments have completed. Fixed `MAXN`/`jetson_clocks` measurements on the pinned
Qwen3.8-27B groupwise artifact establish MTP as the decisive decode optimization: draft-3 is best
at the short workload for both qualified KV formats, while BF16 draft-4 is best at the long
workload. INT8 KV halves the reserved KV payload at the qualified 32K capacity point without a
measured short-workload decode penalty.

| Best configuration | Workload | PP tok/s | TG tok/s | MTP acceptance |
|---|---|---:|---:|---:|
| INT8 KV, draft-3 | `pp512+tg64` | 235.32 | **11.00** | 47.44% |
| BF16 KV, draft-3 | `pp512+tg64` | 235.76 | 10.98 | 47.44% |
| BF16 KV, draft-4 | `pp2048+tg128` | 238.25 | **17.86** | 91.74% |
| INT8 KV, draft-3 | `pp2048+tg128` | 237.41 | 17.26 | 93.07% |

SM87 kernel work also replaced the Q4/Q5 attention-input large-prefill route with R64C128S2.
At T=1024, its public-op median is 13.570 ms, 37.3% faster than the original R32C64S4 route;
the associated numerical test passes.

The full result matrix—including rejected schedules, capacity guard outcomes, commands, and
hardware context—is in the [SM87 experiment ledger](NINFER_JETSON_ORIN_PORT_EXPERIMENTS.md).
See the [live port status](NINFER_JETSON_ORIN_PORT_STATUS.md) for current port provenance and
limitations.

## Build on Orin

Install the JetPack-provided CUDA, compiler, CMake, Ninja, pkg-config, and Python 3.11 packages.
The project also requires FFmpeg >=6 and libcurl >=7.85. On the qualification host, those were
built into the isolated `build/jetson-deps/install` prefix; do not replace JetPack system
libraries globally.

Select the toolkit explicitly and configure an SM87 build:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export PKG_CONFIG_PATH="$PWD/build/jetson-deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PWD/build/jetson-deps/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cmake -S . -B build/port-cuda126-sm87 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DNINFER_BUILD_APPS=ON \
  -DNINFER_BUILD_BENCHMARKS=ON \
  -DBUILD_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/jetson-deps/install"

cmake --build build/port-cuda126-sm87 -j
ctest --test-dir build/port-cuda126-sm87 --output-on-failure
```

For JetPack 7.1, select `/usr/local/cuda-13.1/bin/nvcc` and use a separate build directory, for
example `build/port-cuda131-sm87`. Keep the architecture at 87; CUDA version selection and SM87
qualification are separate checks.

## Tested artifact and CLI

The native real-model gates use the checksum-matched historical v1 artifact:

```text
/home/calvin/models/qwen3_8_27b_v1/qwen3_8_27b.ninfer
SHA-256: eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e
```

The GGUF files under `/home/calvin/models` are reference material for other runtimes and are not
interchangeable with NInfer's registered `.ninfer` product artifact. The `~/.ninfer` path on the
qualification host is a placeholder configuration file, not a model directory.

Run a short BF16-KV MTP generation:

```bash
build/port-cuda126-sm87/apps/ninfer \
  /home/calvin/models/qwen3_8_27b_v1/qwen3_8_27b.ninfer \
  --prompt 'What is 2+2?' --max-new 16 --no-thinking --greedy \
  --kv-dtype bf16 --spec mtp --draft-tokens 2 --lm-head-draft
```

Use `--kv-dtype int8` for the qualified INT8 group-64 KV route. Keep explicit context capacity
while measuring; `--max-context` and `--kv-capacity` control the reservation independently.

## Product surface

NInfer exposes one public Engine route for the registered `.ninfer` artifacts. The native CLI and
serving applications share the same model admission, frontend, state transactions, paged KV,
CUDA Graph, MTP, prefix-reuse, and output semantics. OpenAI Chat Completions, OpenAI Responses,
and Anthropic Messages serving are available through the native Arm64 build.

The current product is one resident model instance on one Orin GPU with bounded one-to-eight
request concurrency. Large-scale preemptive continuous batching and unqualified additional
checkpoint identities remain outside this port's validated scope.

For the repeatable benchmark command and Jetson dependency workflow, see
[Native Jetson development](docs/jetson-orin.md). For CLI options and serving behavior, see the
[documentation map](docs/README.md).
