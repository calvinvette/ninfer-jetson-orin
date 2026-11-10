#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
using Output = LinearBf16SegmentedOutput<8192, 4096>;
using Routes = Q8A16OutputLaunch<Output>;

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(out[0].data),
                        static_cast<__nv_bfloat16*>(out[1].data)};
    const Routes routes{q8_linear_operands(x, weights.first), output, stream};
    const int t = x.ne[1];
    if (t == 1) return routes.template gemv<Q8A16GemvSchedule<8, 1, 2, 2048>>();
    if (t <= 2) return routes.template gemv<Q8A16GemvTileSchedule<2, 8, 1, 2, 2048>>();
    if (t <= 4) return routes.template gemv<Q8A16GemvTileSchedule<4, 8, 1, 2, 2048>>();
    if (t <= 8) return routes.template gemv<Q8A16GemvTileSchedule<8, 8, 1, 2, 2048>>();
    if (t <= 16) return routes.template gemv<Q8A16GemvTileSchedule<16, 8, 1, 2, 2048>>();
    if (t <= 32) return routes.template gemv<Q8A16GemvTileSchedule<16, 8, 1, 2, 2048>>();

    if (t <= 48)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<48, 4, 2, 1, 3, 2048>>();
    if (t <= 64)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<64, 4, 2, 1, 2, 2048>>();
    if (t <= 96)
        return routes.template grouped<Q8A16GroupedSlicedKMmaSchedule<96, 2, 4, 1, 3, 2048>>();

    routes.template mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>();
}
} // namespace

const SplitShape kQ8SplitN12288K2048{12288, 2048, kSplitRows<8192, 4096>, launch_split};
} // namespace ninfer::ops::detail::linear_split
