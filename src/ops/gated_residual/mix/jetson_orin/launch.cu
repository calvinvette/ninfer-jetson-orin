#include "ops/gated_residual/mix/launch.h"
#include "ops/common/host/launch.h"
#include "ops/gated_residual/mix/kernel/output.cuh"
#include "ops/gated_residual/mix/kernel/up_mma_mix.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/linear/kernels/bf16/bf16_split_k_launch.cuh"

#include <type_traits>
#include <stdexcept>

namespace ninfer::ops::detail::hc::mix {
namespace {
template <Kind Form, class Schedule>
using DownRows = std::conditional_t<Form == Kind::Block, Bf16RowTailSchedule<Schedule>, Schedule>;

template <Kind Form, class Schedule>
using DownSchedule = DownRows<Form, Bf16ScheduleInstance<Schedule, kResidualWidth>>;

template <int Rows, int Tokens, int K, int MinBlocks>
using SplitSchedule =
    Bf16A16MmaSchedule<Rows, Tokens, K, 16, 8, 3, MinBlocks, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>;

using DownGemv = Bf16ScheduleInstance<
    Bf16A16GemvSchedule<2, 1, 1, 16, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                        Bf16PhaseOrder::Sequential, 1, 1, 4, 2>,
    kResidualWidth>;
using DownSimt = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<8, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::DirectStream,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    kResidualWidth>;

template <Kind Form>
void down(Operands o, Down schedule, Scratch scratch, cudaStream_t stream) {
    const Bf16A16Operands p{static_cast<const __nv_bfloat16*>(o.normalized),
                            static_cast<const __nv_bfloat16*>(o.down), down_rows(Form),
                            kResidualWidth, o.columns};
    const DownOutput<Form> output{static_cast<__nv_bfloat16*>(scratch.activation), o.injection};
    const LinearIdentityEpilogue identity;
    const Bf16SplitKWorkspace partials{static_cast<float*>(scratch.partials.data),
                                       scratch.partials.bytes};
    switch (schedule) {
    case Down::Gemv:
        launch_bf16_a16_gemv<DownGemv>(p, output, identity, stream);
        break;
    case Down::Simt:
        launch_bf16_a16_simt<DownSimt>(p, output, identity, stream);
        break;
    case Down::Split8x16:
        launch_bf16_a16_split_k_mma<DownSchedule<Form, SplitSchedule<16, 16, 128, 2>>, 8>(
            p, output, identity, partials, stream);
        break;
    case Down::Split8x32:
        launch_bf16_a16_split_k_mma<DownSchedule<Form, SplitSchedule<32, 32, 64, 1>>, 8>(
            p, output, identity, partials, stream);
        break;
    default:
        throw std::logic_error("hc_mix: unsupported SM87 down route");
    }
}

template <int Features, int Tokens>
void folded_up(Operands o, Scratch scratch, cudaStream_t stream) {
    using S               = FoldedUpSchedule<Features, Tokens>;
    constexpr auto kernel = up_mma_mix_kernel<Features, Tokens>;
    const int bytes       = common::prepare_shared<S::kSharedBytes, kernel, true>();
    common::for_each_token_slice(o.columns, Tokens, [&](int offset, int count) {
        const unsigned blocks = (kWidth / Features) * common::div_up(count, Tokens);
        kernel<<<blocks, S::kThreads, bytes, stream>>>(
            o, static_cast<const __nv_bfloat16*>(scratch.activation), offset);
        CUDA_CHECK(cudaGetLastError());
    });
}


} // namespace

void launch(Kind kind, Operands o, Plan p, Scratch scratch, CudaExecutionView execution) {
    if (kind == Kind::Block)
        down<Kind::Block>(o, p.down, scratch, execution.stream());
    else
        down<Kind::Final>(o, p.down, scratch, execution.stream());
    switch (p.up) {
    case Up::Folded16x16:
        folded_up<16, 16>(o, scratch, execution.stream());
        break;
    case Up::Folded32x32:
        folded_up<32, 32>(o, scratch, execution.stream());
        break;
    default:
        throw std::logic_error("hc_mix: unsupported SM87 up route");
    }
}
} // namespace ninfer::ops::detail::hc::mix
