#include "artifact/reader.h"
#include "core/device.h"
#include "core/buffer.h"
#include "models/common/qwen/vision/vision.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/load.h"
#include "models/common/vision/resources.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_bf16.h>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace qwen = models::qwen;
using Clock = std::chrono::steady_clock;

std::pair<double, double> data(std::uint32_t features, const qwen::VisionConfig& config,
                         const qwen::VisionProfile& profile,
                         const qwen::VisionParameters& parameters,
                         DeviceContext& device, std::uint32_t warmup,
                         std::uint32_t repetitions) {
    constexpr std::uint32_t patches_per_feature = 4;
    const auto patch_count = features * patches_per_feature;
    const auto patch_width = static_cast<std::size_t>(config.patch_width());
    std::vector<std::uint16_t> patch_data(std::size_t(patch_count) * patch_width);
    for (std::size_t i = 0; i < patch_data.size(); ++i)
        patch_data[i] = __bfloat16_as_ushort(
            __float2bfloat16(static_cast<float>(std::sin(i * .031) * .1)));
    std::vector<std::span<const std::uint16_t>> patches;
    patches.reserve(features);
    for (std::uint32_t feature = 0; feature < features; ++feature)
        patches.emplace_back(std::span(patch_data).subspan(
            std::size_t(feature) * patches_per_feature * patch_width,
            patches_per_feature * patch_width));

    std::vector<std::int32_t> positions(std::size_t(2) * patch_count);
    std::vector<std::int32_t> indices(std::size_t(4) * patch_count);
    std::vector<float> weights(indices.size(), .25F);
    std::vector<std::int32_t> cu_seqlens(features + 1);
    for (std::uint32_t feature = 0; feature < features; ++feature) {
        cu_seqlens[feature + 1] = static_cast<std::int32_t>((feature + 1) * patches_per_feature);
        for (std::uint32_t patch = 0; patch < patches_per_feature; ++patch) {
            const auto token = feature * patches_per_feature + patch;
            positions[token] = static_cast<std::int32_t>(patch / 2);
            positions[patch_count + token] = static_cast<std::int32_t>(patch % 2);
            const auto base = static_cast<std::uint32_t>(positions[token] *
                                                         config.position_grid_side +
                                                         positions[patch_count + token]);
            for (std::uint32_t corner = 0; corner < 4; ++corner)
                indices[std::size_t(token) * 4 + corner] =
                    static_cast<std::int32_t>((base + corner) % config.num_position_embeddings);
        }
    }
    qwen::VisionEncoderBatch batch{patches, positions, indices, cu_seqlens, weights, features};

    const auto capacity = qwen::VisionContext::plan_resources(config, profile, features,
                                                               512ULL << 20);
    models::vision::Storage storage(device, capacity);
    DeviceBuffer workspace(device.device(), capacity.encoder.workspace_bytes);
    qwen::VisionContext encoder(device.execution_view(), config, parameters);
    auto output = storage.output(0, features);
    for (std::uint32_t i = 0; i < warmup; ++i)
        (void)encoder.encode(batch, output, workspace.span(), storage.staging());
    device.synchronize();

    CudaEventTimer gpu_timer(device, device.stream());
    std::vector<double> gpu_ms, wall_ms;
    gpu_ms.reserve(repetitions);
    wall_ms.reserve(repetitions);
    for (std::uint32_t i = 0; i < repetitions; ++i) {
        const auto start = Clock::now();
        gpu_timer.start();
        (void)encoder.encode(batch, output, workspace.span(), storage.staging());
        gpu_timer.record_stop();
        device.synchronize();
        const auto end = Clock::now();
        gpu_ms.push_back(gpu_timer.elapsed_ms());
        wall_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(gpu_ms.begin(), gpu_ms.end());
    std::sort(wall_ms.begin(), wall_ms.end());
    return {gpu_ms[gpu_ms.size() / 2], wall_ms[wall_ms.size() / 2]};
}

} // namespace

int main(int argc, char** argv) try {
    if (argc < 2 || argc > 4) {
        std::cerr << "usage: ninfer_qwen4_exp_vision_bench ARTIFACT [WARMUP=2] [REPETITIONS=5]\n";
        return 2;
    }
    const auto warmup = argc > 2 ? static_cast<std::uint32_t>(std::stoul(argv[2])) : 2U;
    const auto repetitions = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 5U;
    if (!repetitions) throw std::invalid_argument("repetitions must be positive");

    DeviceContext device;
    EngineOptions options;
    options.enable_vision = true;
    options.max_context = 128;
    options.prefill_chunk = 128;
    options.use_cuda_graph = false;
    artifact::Reader reader(argv[1]);
    auto definition = models::qwen4_exp::read_definition(reader, models::load_options(options));
    auto profile = models::qwen4_exp::execution::describe_model(*definition);
    auto load = models::qwen4_exp::resolve_load(reader, definition, device.device(), Device::cpu(),
                                               {}, true);
    auto model = models::qwen4_exp::materialize_model(std::move(load), device);
    models::qwen4_exp::execution::Parameters parameters(*model, profile);
    if (!profile->config.vision || !profile->vision || !parameters.vision)
        throw std::runtime_error("artifact does not bind Qwen Vision parameters");

    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::cout << "device=" << device.properties().name << " cuda_runtime=" << runtime
              << " model=qwen3.8-flash-next vision_depth=" << profile->config.vision->depth
              << " hidden=" << profile->config.vision->hidden_size << " warmup=" << warmup
              << " repetitions=" << repetitions << '\n';
    std::cout << "features,patches,gpu_ms,wall_ms,patches_per_s\n";
    for (const auto features : {1U, 4U, 16U}) {
        const auto [gpu, wall] = data(features, *profile->config.vision, *profile->vision,
                                     *parameters.vision, device, warmup, repetitions);
        const auto patches = features * 4;
        std::cout << features << ',' << patches << ',' << gpu << ',' << wall << ','
                  << patches * 1000.0 / wall << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ninfer_qwen4_exp_vision_bench: " << error.what() << '\n';
    return 1;
}
