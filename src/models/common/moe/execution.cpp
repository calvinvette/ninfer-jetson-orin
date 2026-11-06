#include "models/common/moe/execution.h"
#include "core/platform.h"

namespace ninfer::models::moe {
namespace {
bool tokenwise_nvfp4(const ops::MoeProfile& profile) {
#if NINFER_TARGET_JETSON_ORIN
    return profile.routed.ffn.gate_up_type == QType::NVFP4;
#else
    (void)profile;
    return false;
#endif
}
} // namespace

std::size_t workspace_bytes(const ops::MoeProfile& profile, ops::MoeEpilogue epilogue,
                            std::int32_t first, std::int32_t last, ops::ExpertSubsetKind subset) {
    if (tokenwise_nvfp4(profile))
        return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, subset, epilogue,
                                                                     1, 1);
    return ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(profile, subset, epilogue, first,
                                                                 last);
}

void execute(const Tensor& hidden, const Parameters& parameters,
             const ExpertBankDeviceView& experts, ops::MoeEpilogue epilogue, Tensor& output,
             MemoryArena& workspace, CudaExecutionView execution,
             const ops::MoeExecutionHints& hints) {
    auto scope         = workspace.scope();
    const auto columns = hidden.ne[1];
    const auto storage = workspace.alloc_bytes(workspace_bytes(
        parameters.profile, epilogue, columns, columns, parameters.subset.kind));
    MemoryArena scratch(storage);
    const ops::MoeWeights weights{
        parameters.control, {experts, parameters.activation_calibration}, parameters.shared};
    if (tokenwise_nvfp4(parameters.profile) && columns > 1) {
        for (std::int32_t token = 0; token < columns; ++token) {
            const auto input_token = hidden.slice(1, token, 1);
            auto output_token = output.slice(1, token, 1);
            ops::topk_softmax_swiglu_moe(input_token, parameters.profile, weights,
                                         parameters.subset, epilogue, output_token, scratch,
                                         execution, hints);
        }
    } else {
        ops::topk_softmax_swiglu_moe(hidden, parameters.profile, weights, parameters.subset,
                                     epilogue, output, scratch, execution, hints);
    }
}

} // namespace ninfer::models::moe
