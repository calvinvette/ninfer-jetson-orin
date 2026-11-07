#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail::kda_input {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("MXFP8 KDA input projection: no SM87 implementation");
}
} // namespace

std::size_t mxfp8_workspace_capacity(int, int, int, bool, LinearPolicy) {
    unsupported();
}

Mxfp8Workspace mxfp8_execution_workspace(int, int, bool, LinearPolicy) {
    unsupported();
}

void launch_mxfp8(const Tensor&, const Weight&, LinearPolicy, const Arguments&, void*,
                  cudaStream_t) {
    unsupported();
}
} // namespace ninfer::ops::detail::kda_input
