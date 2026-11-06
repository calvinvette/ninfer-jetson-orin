#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_complete {
namespace {

Plan select_decode(int tokens, int, ExpertSubsetKind) {
    if (tokens != 1)
        throw std::invalid_argument("SM87 NVFP4 MoE is qualified only for one-token decode");
    Plan plan;
    plan.implementation = kernel::ExpertImplementation::BlockScaled;
    plan.block_scaled = {.format = QType::NVFP4,
                         .intermediate = 640,
                         .vector = true,
                         .shared_algorithm = kernel::block_scaled::SharedAlgorithm::Vector,
                         .shared_split_activation = true,
                         .network_topk = true,
                         .shared_columns = 8};
    return plan;
}

} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() {
    static constexpr int boundaries[]{2};
    static const ShapePolicy policy{.select = select_decode, .allocation_boundaries = boundaries,
                                    .masked_slice_tokens = 1};
    return policy;
}

} // namespace ninfer::ops::detail::moe_complete
