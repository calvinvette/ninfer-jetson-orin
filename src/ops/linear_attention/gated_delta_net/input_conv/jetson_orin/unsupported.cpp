#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/launch.h"

namespace ninfer::ops::detail {
std::size_t fp8_gdn_input_partial_capacity_bytes(std::int32_t) {
    throw std::invalid_argument("FP8 GDN A8: no SM87 implementation");
}

void launch_fp8_gdn_input_a8(const Tensor&, const Weight&, Tensor&, Tensor&, Fp8A8Workspace,
                             cudaStream_t) {
    throw std::invalid_argument("FP8 GDN A8: no SM87 implementation");
}

void launch_nvfp4_gdn_input_a4(const Tensor&, const Weight&, Tensor&, Tensor&,
                               Nvfp4A4Workspace, cudaStream_t) {
    throw std::invalid_argument("NVFP4 GDN A4: no SM87 implementation");
}
} // namespace ninfer::ops::detail
