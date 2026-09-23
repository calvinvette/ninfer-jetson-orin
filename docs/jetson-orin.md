# Native Jetson Orin development

The Jetson port is in progress. See the
[port status](../NINFER_JETSON_ORIN_PORT_STATUS.md) for completed gates and
remaining limitations. A successful native build is not a claim of qualified
SM87 inference or performance.

Use the native Linux/aarch64 environment and the CUDA installation supplied with
JetPack. The existing desktop CUDA Docker image is not the Jetson build path.
Keep explicit CUDA device allocation during correctness qualification.

## Prerequisites

- CMake >=3.28, Ninja, pkg-config and a C++20 host compiler.
- CUDA 12.6 selected explicitly, rather than through an ambiguous CUDA symlink.
- FFmpeg: libavformat >=60, libavcodec >=60, libavutil >=58, libswscale >=7.
- libcurl >=7.85, with TLS support for HTTPS media acquisition.
- Python 3.11 for maintainer tests/tools.

Prefer system development packages when their versions satisfy these requirements.
The Ubuntu 22.04 package candidates on the porting host do not: FFmpeg 4.4 and
curl 7.81 are too old. A local dependency prefix avoids replacing JetPack's system
libraries. The following commands use already-downloaded, explicitly versioned
source archives; they do not acquire model artifacts.

```bash
mkdir -p build/jetson-deps/src
tar -xf /path/to/ffmpeg-6.1.2.tar.xz -C build/jetson-deps/src
tar -xf /path/to/curl-8.10.1.tar.xz -C build/jetson-deps/src
NINFER_DEPS="$PWD/build/jetson-deps/install"

(
  cd build/jetson-deps/src/ffmpeg-6.1.2
  ./configure --prefix="$NINFER_DEPS" \
    --disable-programs --disable-doc --disable-debug \
    --enable-shared --disable-static
  make -j
  make install
)

cmake -S build/jetson-deps/src/curl-8.10.1 \
  -B build/jetson-deps/curl-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$NINFER_DEPS" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF \
  -DCURL_USE_OPENSSL=ON
cmake --build build/jetson-deps/curl-build -j
cmake --install build/jetson-deps/curl-build
```

The curl build requires OpenSSL development headers/libraries. The FFmpeg build
detects available optional system dependencies; the porting host provides zlib.

## CUDA 12.6 compatibility gate

During the toolkit backport, keep architecture 86 to separate toolkit changes
from SM87 enablement. This command deliberately does not select SM87:

```bash
export PKG_CONFIG_PATH="$PWD/build/jetson-deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$PWD/build/jetson-deps/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S . -B build/port-cuda126-sm86 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPython3_EXECUTABLE=/usr/bin/python3.11 \
  -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build/port-cuda126-sm86 -j
ctest --test-dir build/port-cuda126-sm86 \
  -R '^ninfer_nvfp4_codec_test$' --output-on-failure
```

GPU tests must have access to the host's NVIDIA device nodes. An isolated sandbox
can report a driver initialization failure even when CUDA works on the host.
An SM86-compiled codec test running on Orin verifies that codec; it does not
replace SM86 hardware regression tests or the full SM87 qualification gates.

Keep the local prefix in `LD_LIBRARY_PATH` when running applications or tests:
FFmpeg's transitive shared-library dependencies also reside in that prefix.

## Repeatable Orin benchmark

Build the product benchmark with `-DNINFER_BUILD_BENCHMARKS=ON`, then run a fixed
`pp512+tg64` matrix against the pinned artifact:

```bash
NINFER=/home/calvin/models/qwen3_8_27b_v1/qwen3_8_27b.ninfer
LD_LIBRARY_PATH="$PWD/build/jetson-deps/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build/port-cuda126-sm87/bench/ninfer_bench \
  --weights "$NINFER" --prompt-gen 512,64 --repetitions 3 --warmup 1 \
  --max-ctx 2048 --kv-dtype bf16 --mtp-draft-tokens 3 --lm-head-draft
```

Record `nvpmodel -q` and `tegrastats` output with the result. `jetson_clocks`
requires elevated Jetson privileges; when those are unavailable, report the
power mode and dynamic-clock limitation instead of calling the result
clock-normalized.

The supported long-context point is 32,768 prompt tokens plus one generated token
for either BF16 or INT8 KV. The eager gates measure 211.50 BF16 and 209.03 INT8
prefill tok/s, with 2.00 GiB and 1.03 GiB KV payloads respectively. For contexts
above that point, retain host-memory telemetry and stop before Linux memory
pressure can destabilize the board. This wrapper writes a sample stream from
`/proc/meminfo` and terminates the child if `MemAvailable` falls below its 2 GiB
default floor:

```bash
python3 tools/bench/run_with_host_pressure.py \
  --output /tmp/pp40960-host-pressure.json -- \
  build/port-cuda126-sm87/bench/ninfer_bench \
  --weights "$NINFER" --prompt-gen 40960,1 --repetitions 1 --warmup 0 \
  --max-ctx 40961 --kv-dtype int8 --no-cuda-graph
```

Do not classify an aborted run as a capacity result. Preserve the JSON report,
including its minimum `mem_available_bytes` sample, when reporting a pass or a
pressure-limited result.

## Allocation classes

The Engine default remains explicit `cudaMalloc`. The core also exposes an opt-in
stream-ordered `cudaMallocAsync`/`cudaFreeAsync` class for `DeviceBuffer` and
`DeviceArena`; its Orin allocation, transfer, suballocation, and destruction gate
passes. It is not selected for Engine allocations because an end-to-end benefit and
CUDA-Graph-stability advantage have not been demonstrated. Do not substitute managed
or mapped allocations globally based only on unified physical LPDDR.
