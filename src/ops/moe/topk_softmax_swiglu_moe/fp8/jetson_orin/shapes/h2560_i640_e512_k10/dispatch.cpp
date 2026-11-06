#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_complete {
namespace {

Plan select_decode(int tokens, int, ExpertSubsetKind) {
    if (tokens <= 0 || tokens > 2) {
        throw std::invalid_argument(
            "SM87 block-FP8 MoE is qualified only for one/two-token decode");
    }
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    plan.block_scaled = {.format = QType::FP8_E4M3FN_BLOCK128X128_BF16,
                         .intermediate = 640,
                         .vector = true,
                         .shared_algorithm = kernel::block_scaled::SharedAlgorithm::Vector,
                         .shared_split_activation = true,
                         .network_topk = true,
                         .shared_columns = 8};
    return plan;
}

} // namespace

const ShapePolicy& fp8_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{3};
    static const ShapePolicy policy{.select = select_decode, .allocation_boundaries = boundaries,
                                    .masked_slice_tokens = 2};
    return policy;
}

} // namespace ninfer::ops::detail::moe_complete
