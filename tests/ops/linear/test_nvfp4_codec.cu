#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/quantized_weight.h"

#include <array>
#include <cstdio>

namespace {

__global__ void decode_all(float2* output) {
    const unsigned byte = threadIdx.x;
    output[byte] = ninfer::ops::detail::decode_nvfp4_e2m1x2(byte);
}

bool check(cudaError_t result) {
    if (result == cudaSuccess) { return true; }
    std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
    return false;
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::puts("SKIP: no usable CUDA device");
        return 77;
    }
    float2* device = nullptr;
    if (!check(cudaMalloc(&device, 256 * sizeof(float2)))) { return 1; }
    decode_all<<<1, 256>>>(device);
    std::array<float2, 256> actual{};
    const bool launched = check(cudaGetLastError());
    const bool copied = check(cudaMemcpy(actual.data(), device, sizeof(actual), cudaMemcpyDeviceToHost));
    const bool freed = check(cudaFree(device));
    if (!launched || !copied || !freed) { return 1; }

    // The independent exact oracle is the test-owned E2M1 magnitude table.
    // Compare FP32 bits, including both signed zeros and nibble ordering.
    using namespace ninfer::test::quantized_weight::detail;
    for (unsigned byte = 0; byte < 256; ++byte) {
        const float lo = static_cast<float>(decode_e2m1(byte & 15U));
        const float hi = static_cast<float>(decode_e2m1(byte >> 4));
        if (float_bits(actual[byte].x) != float_bits(lo) ||
            float_bits(actual[byte].y) != float_bits(hi)) {
            std::fprintf(stderr, "FAIL: packed E2M1 byte %u\n", byte);
            return 1;
        }
    }
    std::puts("OK: all 256 packed E2M1 bytes decode exactly, including signed zero");
    return 0;
}
