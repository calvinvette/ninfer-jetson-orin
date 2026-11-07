#include "ops/gated_residual/mix/select.h"
#include "ops/gated_residual/mix/workspace.h"
#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::ops::detail::hc::mix {
namespace {
struct Route { int first, last; Plan plan; };
constexpr int kLast = std::numeric_limits<int>::max();
constexpr std::array block_routes{
    Route{1, 1, {Down::Gemv, Up::Folded16x16}},
    // FP32 injections feed the BF16 Hyper-Connection residual update. Keep short
    // speculative windows on the non-split-K projection route so its per-column
    // reduction does not change when one request is verified as a token window.
    Route{2, 48, {Down::Simt, Up::Folded16x16}},
    Route{49, kLast, {Down::Split8x32, Up::Folded32x32}},
};
constexpr std::array final_routes{
    Route{1, 1, {Down::Gemv, Up::Folded16x16}},
    Route{2, 4, {Down::Simt, Up::Folded16x16}},
    Route{5, 32, {Down::Split8x16, Up::Folded16x16}},
    Route{33, kLast, {Down::Split8x32, Up::Folded32x32}},
};
std::span<const Route> routes(Kind kind) {
    return kind == Kind::Block ? std::span<const Route>{block_routes}
                               : std::span<const Route>{final_routes};
}
} // namespace

Plan select_plan(Kind kind, std::int32_t columns) {
    validate_interval(columns, columns);
    for (const auto& route : routes(kind))
        if (columns >= route.first && columns <= route.last) return route.plan;
    throw std::logic_error("hc_mix: missing SM87 route");
}

std::size_t capacity(Weights weights, std::int32_t first, std::int32_t last) {
    describe_weights(weights);
    validate_interval(first, last);
    std::size_t peak = 0;
    for (const auto& route : routes(weights.kind)) {
        if (route.last < first || route.first > last) continue;
        peak = std::max(peak,
                        layout(weights.kind, std::min(last, route.last), split_factor(route.plan.down)).bytes);
    }
    return peak;
}
} // namespace ninfer::ops::detail::hc::mix
