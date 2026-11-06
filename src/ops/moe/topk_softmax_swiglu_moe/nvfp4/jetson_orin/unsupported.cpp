#include "ops/moe/topk_softmax_swiglu_moe/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_complete {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("topk_softmax_swiglu_moe: no SM87 block-scaled NVFP4/FP8 kernel");
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() { unsupported(); }
const ShapePolicy& nvfp4_h2560_i320_e512_k10() { unsupported(); }
const ShapePolicy& fp8_h2560_i640_e512_k10() { unsupported(); }
const ShapePolicy& fp8_h2560_i320_e512_k10() { unsupported(); }
} // namespace ninfer::ops::detail::moe_complete
