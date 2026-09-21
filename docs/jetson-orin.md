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
