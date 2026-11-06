#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("block-scaled NVFP4/FP8 MoE has no SM87 implementation");
}
} // namespace

namespace nvfp4 {
void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&, CudaExecutionView, bool) {
    unsupported();
}
void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    unsupported();
}
void launch_vector(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, ExpertSubset,
                   bool, MoeEpilogue, Tensor&, const Storage&, const Plan&, CudaExecutionView) {
    unsupported();
}
} // namespace nvfp4

namespace fp8 {
void launch_input(const Tensor&, const CudaSwiGLUExperts&, const Storage&, CudaExecutionView, bool) {
    unsupported();
}
void launch_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                    const Storage&, CudaExecutionView) {
    unsupported();
}
void launch_vector(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, ExpertSubset,
                   bool, MoeEpilogue, Tensor&, const Storage&, const Plan&, CudaExecutionView) {
    unsupported();
}
} // namespace fp8
} // namespace ninfer::ops::detail::moe_cuda::block_scaled
