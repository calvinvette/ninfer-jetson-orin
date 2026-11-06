#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/fp8/compute.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8 {
namespace {

[[noreturn]] void require_decode_route() {
    throw std::invalid_argument(
        "SM87 block-FP8 MoE supports the qualified vector decode route only");
}

} // namespace

void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&,
                  CudaExecutionView, bool) {
    require_decode_route();
}

void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    require_decode_route();
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, const Plan& plan, CudaExecutionView execution) {
    if (plan.format != QType::FP8_E4M3FN_BLOCK128X128_BF16 || plan.intermediate != kIntermediate)
        throw std::invalid_argument("SM87 block-FP8 vector route has unsupported geometry");
    const auto stream = execution.stream();
    const auto* ids = static_cast<const int*>(routes.ids.data);
    const auto* mask = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<<<dim3(kIntermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        storage.vector_activation);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<4><<<dim3(kHidden / 4, x.ne[1]), kTopK * 32, 0, stream>>>(
            storage.vector_activation, experts.bank.down, ids, coefficients, subset.kind, mask,
            output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (shared) {
        down(FloatContribution{storage.routed_sum});
    } else {
        with_output(destination, mode, down);
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8
