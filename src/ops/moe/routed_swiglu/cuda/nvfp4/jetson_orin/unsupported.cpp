#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail::moe_routed {
namespace {
[[noreturn]] void unsupported() {
    throw std::invalid_argument("routed_swiglu CUDA: no SM87 block-scaled NVFP4/FP8 kernel");
}
} // namespace

const ShapePolicy& nvfp4_h2560_i640_e512_k10() { unsupported(); }
const ShapePolicy& nvfp4_h2560_i320_e512_k10() { unsupported(); }
const ShapePolicy& fp8_h2560_i640_e512_k10() { unsupported(); }
const ShapePolicy& fp8_h2560_i320_e512_k10() { unsupported(); }
const ShapePolicy& nvfp4_h4096_i2048_e288_k8() { unsupported(); }
const ShapePolicy& nvfp4_h4096_i1024_e288_k8() { unsupported(); }
const ShapePolicy& fp8_h4096_i2048_e288_k8() { unsupported(); }
const ShapePolicy& fp8_h4096_i1024_e288_k8() { unsupported(); }
} // namespace ninfer::ops::detail::moe_routed
