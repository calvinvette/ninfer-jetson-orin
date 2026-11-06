#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_quantize(const Tensor&, const Weight&, Nvfp4A4Workspace,
                              Nvfp4ScaleLayout, cudaStream_t) {
    throw std::invalid_argument("NVFP4 A4 quantize: no SM87 implementation");
}
} // namespace ninfer::ops::detail
