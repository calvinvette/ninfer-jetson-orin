#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/compute.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4 {
namespace {

[[noreturn]] void require_decode_route() {
    throw std::invalid_argument("SM87 NVFP4 MoE supports the qualified vector decode route only");
}

template <int Intermediate>
void launch_vector_impl(const Tensor& x, const CudaSwiGLUExperts& experts,
                        const ExpertRoutes& routes, ExpertSubset subset, bool shared,
                        MoeEpilogue mode, Tensor& destination, const Storage& storage,
                        CudaExecutionView execution) {
    const auto stream = execution.stream();
    const auto* ids = static_cast<const int*>(routes.ids.data);
    const auto* mask = static_cast<const int*>(subset.mask.data);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    vector_gate<Intermediate><<<dim3(Intermediate / 4, x.ne[1] * kTopK), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), experts.bank.gate_up, ids, subset.kind, mask,
        storage.vector_activation);
    CUDA_CHECK(cudaGetLastError());
    const auto down = [&](auto output) {
        vector_down<Intermediate, 8, 2><<<dim3(kHidden / 8, x.ne[1]), 2 * 32, 0, stream>>>(
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

} // namespace

void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&, CudaExecutionView, bool) {
    require_decode_route();
}

void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    require_decode_route();
}

void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, const Plan& plan, CudaExecutionView execution) {
    if (plan.intermediate == 320)
        launch_vector_impl<320>(x, experts, routes, subset, shared, mode, destination, storage,
                                execution);
    else
        launch_vector_impl<640>(x, experts, routes, subset, shared, mode, destination, storage,
                                execution);
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::nvfp4
