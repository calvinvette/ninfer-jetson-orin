#include "ninfer/ops/moe.h"
#include "core/platform.h"

#include <stdexcept>

namespace ninfer::ops {

#if NINFER_TARGET_JETSON_ORIN
static constexpr const char* kUnavailable = "routed_swiglu CPU: no implementation for Jetson Orin";
#else
static constexpr const char* kUnavailable = "routed_swiglu CPU: no implementation for dgx_spark";
#endif

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile&, ExpertSubsetKind,
                                                   std::int32_t, std::int32_t) {
    throw std::invalid_argument(kUnavailable);
}

void routed_swiglu(const Tensor&, const RoutedSwiGLUProfile&, const CpuSwiGLUExperts&,
                   const ExpertRoutes&, ExpertSubset, Tensor&, MemoryArena&, CpuExecutionView) {
    throw std::invalid_argument(kUnavailable);
}

} // namespace ninfer::ops
