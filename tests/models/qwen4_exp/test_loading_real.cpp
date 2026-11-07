#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "core/arena.h"
#include "core/buffer.h"
#include "models/qwen4_exp/program/resources.h"
#include "models/common/moe/weights.h"
#include "artifact/fixture.h"
#include "core/transfer.h"
#include "core/platform.h"

#include "ops/quantized_weight.h"
#include "ops/op_check.h"
#include "ninfer/ops/linear_split.h"
#include "ops/ple/gate_fixture.h"
#include "ops/ple/conv_fixture.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <thread>
#include <sys/resource.h>

namespace {
using namespace ninfer;
namespace q4 = models::qwen4_exp;
using test::artifact_fixture::require;
constexpr std::size_t GiB = 1024ULL * 1024 * 1024;

void qualify_real_ple(const q4::execution::PleParameters& parameters, const Tensor& next_norm,
                      DeviceContext& device, int tokens) {
    using namespace test::ple;
    GateFixture gate(tokens, 1);
    gate.eps = parameters.gate.rms_epsilon;
    gate.prepare(GateSignal::Random, 0, false);
    const auto bind_norm = [](const Tensor& source, Words& destination) {
        require(source.dtype == DType::BF16 && source.ne[0] == H && source.ne[1] == C &&
                    source.nb[0] == 2 && source.nb[1] == H * 2,
                "real PLE gate norm has unexpected shape or layout");
        const auto words = test::from_device<std::uint16_t>(source.data, H * C);
        for (int s = 0; s < C; ++s)
            for (int d = 0; d < H; ++d)
                destination.host[destination.index(d, s)] = words[s * H + d];
        destination.upload();
    };
    bind_norm(parameters.gate.norm_key, gate.norm_key);
    bind_norm(parameters.gate.norm_query, gate.norm_query);
    bind_norm(parameters.gate.norm_conv, gate.norm_conv);
    gate.launch(device.execution_view());
    device.synchronize();
    require(gate.verify("Flash-Next real PLE gate") == 0,
            "Flash-Next real PLE gate differs from FP64 mathematical oracle");
    std::cout << "PASS Flash-Next real PLE gate FP64 oracle\n";

    ConvFixture conv(tokens, 1);
    conv.prepare(0, false);
    conv.residual.host = gate.residual.host;
    conv.gated.host = gate.gated.download();
    conv.records.host = gate.records.download();
    std::fill(conv.history.host.begin(), conv.history.host.end(), 0);
    conv.residual.upload();
    conv.gated.upload();
    conv.records.upload();
    conv.history.upload();
    const auto bind = [](const Tensor& source, Words& destination, int x, int y) {
        require(source.dtype == DType::BF16 && source.ne[0] == x && source.ne[1] == y &&
                    source.nb[0] == 2 && source.nb[1] == x * 2,
                "real PLE convolution or output norm has unexpected shape or layout");
        const auto words = test::from_device<std::uint16_t>(source.data,
                                                            std::size_t(x) * y);
        for (int j = 0; j < y; ++j)
            for (int i = 0; i < x; ++i)
                destination.host[destination.index(i, j)] = words[j * x + i];
        destination.upload();
    };
    bind(parameters.convolution, conv.kernel, D, 4);
    bind(next_norm, conv.norm, H, C);
    conv.launch(true, device.execution_view());
    device.synchronize();
    require(conv.verify(true, "Flash-Next real PLE convolution") == 0,
            "Flash-Next real PLE convolution differs from FP64 mathematical oracle");
    std::cout << "PASS Flash-Next real PLE convolution FP64 oracle\n";

    Words after({D, 9, 1});
    after.fill(0x5a, 1.f, true);
    after.upload();
    DeviceBuffer destination_offset(device.device(), sizeof(std::int64_t));
    upload_offsets(destination_offset, after.tensor, {0});
    DeviceBuffer accepted_count(device.device(), sizeof(std::int32_t));
    const auto before_view = conv.state_view();
    ops::PleConvStateView after_view{
        history_field(after.tensor, after.device->span(),
                      Tensor(destination_offset.span(), DType::I64, {1}))};
    const auto batch_view = conv.batch_view();
    const Tensor counts(accepted_count.span(), DType::I32, {1});
    for (int accepted = 0; accepted <= conv.tokens; ++accepted) {
        after.upload();
        accepted_count.copy_from_host(&accepted, sizeof(accepted));
        ops::ple_state_extract(before_view, {conv.records.tensor, batch_view.valid_columns},
                               counts, after_view, device.execution_view());
        device.synchronize();
        auto expected = after.host;
        for (int c = 0; c < D; ++c)
            for (int j = 0; j < 9; ++j) {
                const int source = accepted + j;
                expected[after.index(c, j)] =
                    source < 9 ? conv.history.host[conv.history.index(c, source, 1)]
                               : conv.records.host[conv.records.index(c, source - 9, 0)];
            }
        require(test::verify_exact("Flash-Next real PLE committed state", after.download(),
                                   expected) == 0,
                "Flash-Next real PLE state differs from exact accepted-record transition");
    }
    std::cout << "PASS Flash-Next real PLE committed states T=" << tokens
              << " accepted extents 0.." << tokens << '\n';
}

void qualify_real_qsa_projection(const q4::execution::LinearParameters& projection,
                                 std::span<const std::int32_t> segments,
                                 DeviceContext& device, const char* name) {
    namespace qw = test::quantized_weight;
    const auto& source = projection.weight;
    if ((source.qtype != QType::Q8_G32_FP16 && source.qtype != QType::NVFP4 &&
         source.qtype != QType::BF16) ||
        source.k <= 0 || source.n <= 0)
        throw std::runtime_error(
            "Flash-Next QSA projection has unexpected weight format or geometry: qtype=" +
            std::to_string(static_cast<int>(source.qtype)) + " K=" +
            std::to_string(source.k) + " N=" + std::to_string(source.n));
    require(std::accumulate(segments.begin(), segments.end(), 0) == source.n,
            "Flash-Next QSA projection segments do not cover the parent weight");

    const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(source.n),
                                             static_cast<std::uint64_t>(source.k)};
    qw::PackedWeight packed;
    packed.weight = source;
    packed.weight.device = Device::cpu();
    if (source.qtype == QType::BF16) {
        const auto bytes = std::size_t(source.n) * source.k * sizeof(std::uint16_t);
        const void* data = source.qdata ? source.qdata : source.payload;
        require(data != nullptr, "Flash-Next QSA BF16 projection has no data plane");
        packed.payload.resize(bytes);
        CUDA_CHECK(cudaMemcpy(packed.payload.data(), data, bytes, cudaMemcpyDeviceToHost));
        packed.geometry = {};
    } else {
        packed.geometry = weight_geometry(source.qtype, source.layout, shape);
        require(source.qdata != nullptr && source.scales != nullptr,
                "Flash-Next QSA projection is missing a code or scale plane");
        const auto code_bytes = packed.geometry.code_bytes;
        const auto scale_bytes = packed.geometry.scale_bytes;
        packed.payload.resize(packed.geometry.bytes);
        CUDA_CHECK(cudaMemcpy(packed.payload.data(), source.qdata, code_bytes,
                              cudaMemcpyDeviceToHost));
        const auto scale_offset = source.qtype == QType::Q8_G32_FP16
                                      ? code_bytes
                                      : packed.geometry.scale_offset;
        CUDA_CHECK(cudaMemcpy(packed.payload.data() + scale_offset, source.scales, scale_bytes,
                              cudaMemcpyDeviceToHost));
        packed.code_plane_bytes = code_bytes;
        packed.high_plane_offset = packed.geometry.high_offset;
        packed.high_plane_bytes = packed.geometry.high_bytes;
        packed.scale_plane_bytes = scale_bytes;
        if (source.qtype == QType::Q8_G32_FP16) {
            packed.payload.resize(code_bytes + scale_bytes);
            packed.geometry = {};
            packed.scale_plane_offset = scale_offset;
        } else {
            std::memcpy(packed.payload.data() + packed.geometry.matrix_scale_offset,
                        &source.weight_scale, sizeof(source.weight_scale));
            packed.scale_plane_offset = packed.geometry.scale_offset;
        }
    }

    constexpr int columns = 5;
    std::vector<std::uint16_t> input_host(std::size_t(source.k) * columns);
    for (int column = 0; column < columns; ++column)
        for (int k = 0; k < source.k; ++k) {
            const float value = static_cast<float>((k * 13 + column * 7) % 31 - 15) / 128.0F;
            auto bits = std::bit_cast<std::uint32_t>(value);
            bits += 0x7fffU + ((bits >> 16U) & 1U);
            input_host[std::size_t(column) * source.k + k] =
                static_cast<std::uint16_t>(bits >> 16U);
        }
    DeviceBuffer input_buffer(device.device(), input_host.size() * sizeof(std::uint16_t));
    input_buffer.copy_from_host(input_host.data(), input_buffer.size());

    const auto run = [&](int token_count) {
        Tensor input(input_buffer.span(), DType::BF16, {source.k, token_count});
        std::vector<DeviceBuffer> output_buffers;
        std::vector<Tensor> outputs;
        output_buffers.reserve(segments.size());
        outputs.reserve(segments.size());
        for (const auto rows : segments) {
            output_buffers.emplace_back(device.device(),
                                        std::size_t(rows) * token_count * sizeof(std::uint16_t));
            outputs.emplace_back(output_buffers.back().span(), DType::BF16,
                                 std::initializer_list<std::int32_t>{rows, token_count});
        }
        const ops::ProjectionWeights weights = projection;
        const auto capacity = ops::linear_split_workspace_capacity_bytes(
            weights, segments, token_count, token_count);
        DeviceBuffer scratch(device.device(), std::max<std::size_t>(256, capacity));
        MemoryArena workspace(scratch.span());
        ops::linear_split(input, weights, outputs, workspace, device.execution_view());
        device.synchronize();
        std::vector<std::vector<std::uint16_t>> values(segments.size());
        for (std::size_t i = 0; i < segments.size(); ++i) {
            values[i].resize(std::size_t(segments[i]) * token_count);
            output_buffers[i].copy_to_host(values[i].data(), output_buffers[i].size());
        }
        return values;
    };
    const auto serial = run(1);
    const auto verify = run(columns);
    constexpr int rows_per_segment = 64;
    int first_column_differences = 0;
    for (std::size_t segment = 0; segment < segments.size(); ++segment)
        for (std::int32_t row = 0; row < segments[segment]; ++row)
            first_column_differences +=
                serial[segment][row] != verify[segment][row];

    std::vector<double> actual(rows_per_segment * segments.size() * columns);
    std::vector<double> expected(actual.size());
    std::vector<std::int32_t> begins(segments.size());
    for (std::size_t segment = 1; segment < segments.size(); ++segment)
        begins[segment] = begins[segment - 1] + segments[segment - 1];
    for (int job = 0; job < rows_per_segment * static_cast<int>(segments.size()); ++job) {
        const auto segment = static_cast<std::size_t>(job / rows_per_segment);
        const int sample = job % rows_per_segment;
        const int row = std::min(segments[segment] - 1,
                                 sample * segments[segment] / rows_per_segment);
        const int logical_row = begins[segment] + row;
        for (int column = 0; column < columns; ++column) {
            double sum = 0.0;
            for (int k = 0; k < source.k; ++k)
                sum += qw::logical_weight_fp64(packed, logical_row, k) *
                       static_cast<double>(qw::detail::bf16_to_f32(
                           input_host[std::size_t(column) * source.k + k]));
            const auto index = std::size_t(job) * columns + column;
            expected[index] = sum;
            actual[index] = static_cast<double>(qw::detail::bf16_to_f32(
                verify[segment][std::size_t(column) * segments[segment] + row]));
        }
    }
    constexpr test::ReductionCriterion criterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
    const auto stats = test::compute_reduction_stats(actual.data(), expected.data(),
                                                      static_cast<std::int64_t>(actual.size()));
    require(test::reduction_passes(stats, static_cast<std::int64_t>(actual.size()), criterion),
            "Flash-Next real QSA projection differs from its FP64 linear oracle");
    std::cout << "Flash-Next real QSA projection " << name << " oracle rel_l2="
              << stats.relative_l2 << " max_abs=" << stats.maximum_absolute_error
              << " T1/T5 first-column differing BF16 values=" << first_column_differences << '\n';
}

test::quantized_weight::PackedWeight decode_nvfp4_expert(
    const WeightBankView& bank, std::uint32_t expert,
    const std::function<void(std::uint64_t, std::span<std::byte>)>& read) {
    namespace qw = test::quantized_weight;
    const auto& layout = bank.layout;
    const auto& geometry = bank.parent->geometry;
    qw::PackedWeight packed;
    packed.code_plane_bytes = layout.codes.expert_bytes;
    packed.scale_plane_offset = packed.code_plane_bytes;
    packed.scale_plane_bytes = layout.scales.expert_bytes;
    packed.payload.resize(packed.code_plane_bytes + packed.scale_plane_bytes);
    read(layout.codes.offset + expert * layout.codes.expert_stride,
         std::as_writable_bytes(std::span(packed.payload).first(packed.code_plane_bytes)));
    read(layout.scales.offset + expert * layout.scales.expert_stride,
         std::as_writable_bytes(std::span(packed.payload).subspan(
             packed.scale_plane_offset, packed.scale_plane_bytes)));

    auto& weight = packed.weight;
    weight.device = Device::cpu();
    weight.qtype = QType::NVFP4;
    weight.layout = QuantLayout::BlockScaleK16M128x4;
    weight.scale_dtype = DType::FP8_E4M3FN;
    weight.ndim = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = layout.region.rows;
    weight.k = weight.shape[1] = weight.padded_shape[1] = layout.columns;
    weight.group = weight.group_size = 16;
    weight.weight_scale = bank.matrix_scale(expert);
    if (geometry.padded_columns != layout.columns || !weight.weight_scale)
        throw std::runtime_error("real NVFP4 expert geometry or matrix scale is invalid");
    return packed;
}

std::vector<double> real_nvfp4_moe_oracle(
    const models::moe::Parameters& parameters, const ExpertBankSource& bank,
    const std::function<void(std::uint64_t, std::span<std::byte>)>& read_gate_up,
    const std::function<void(std::uint64_t, std::span<std::byte>)>& read_down,
    std::span<const std::uint16_t> input) {
    namespace qw = test::quantized_weight;
    const auto& profile = parameters.profile;
    const auto hidden = profile.routed.ffn.hidden;
    const auto intermediate = profile.routed.ffn.intermediate;
    const auto experts = profile.routed.experts;
    const auto top_k = profile.routed.top_k;
    if (input.size() != static_cast<std::size_t>(hidden) ||
        profile.routed.ffn.gate_up_type != QType::NVFP4 ||
        profile.routed.ffn.down_type != QType::NVFP4)
        throw std::runtime_error("real-layer oracle received an unsupported profile");

    std::vector<std::uint16_t> control_bits(
        std::size_t(experts + 1) * hidden);
    CUDA_CHECK(cudaMemcpy(control_bits.data(), parameters.control.qdata,
                          control_bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
    const auto control = [&](int row, int column) {
        return static_cast<double>(qw::detail::bf16_to_f32(
            control_bits[std::size_t(row) * hidden + column]));
    };
    std::vector<double> logits(experts + 1, 0.0);
    for (int row = 0; row <= experts; ++row)
        for (int k = 0; k < hidden; ++k)
            logits[row] += control(row, k) * qw::detail::bf16_to_f32(input[k]);
    std::vector<int> order(experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    double normalization = 0.0;
    for (int i = 0; i < top_k; ++i) normalization += std::exp(logits[order[i]] - logits[order[0]]);

    std::vector<double> result(hidden, 0.0);
    std::vector<double> activation(intermediate);
    for (int selected = 0; selected < top_k; ++selected) {
        const auto expert = static_cast<std::uint32_t>(order[selected]);
        const auto gate_up = decode_nvfp4_expert(bank.gate_up, expert, read_gate_up);
        const auto down = decode_nvfp4_expert(bank.down, expert, read_down);
        const auto coefficient = std::exp(logits[expert] - logits[order[0]]) / normalization;
        for (int row = 0; row < intermediate; ++row) {
            double gate = 0.0, up = 0.0;
            for (int k = 0; k < hidden; ++k) {
                const auto x = static_cast<double>(qw::detail::bf16_to_f32(input[k]));
                gate += qw::logical_weight_fp64(gate_up, row, k) * x;
                up += qw::logical_weight_fp64(gate_up, row + intermediate, k) * x;
            }
            if (profile.routed.ffn.clamp_limit) {
                const auto limit = static_cast<double>(*profile.routed.ffn.clamp_limit);
                gate = std::min(gate, limit);
                up = std::clamp(up, -limit, limit);
            }
            activation[row] = (gate >= 0.0 ? gate / (1.0 + std::exp(-gate))
                                            : gate * std::exp(gate) / (1.0 + std::exp(gate))) * up;
        }
        for (int row = 0; row < hidden; ++row) {
            double value = 0.0;
            for (int k = 0; k < intermediate; ++k)
                value += qw::logical_weight_fp64(down, row, k) * activation[k];
            result[row] += coefficient * value;
        }
    }
    if (profile.shared.intermediate != 0) {
        const auto& gate_weight = parameters.shared.gate_up.weight;
        const auto& down_weight = parameters.shared.down.weight;
        if (gate_weight.qtype != QType::BF16 || down_weight.qtype != QType::BF16 ||
            gate_weight.k != hidden || gate_weight.n != 2 * profile.shared.intermediate ||
            down_weight.k != profile.shared.intermediate || down_weight.n != hidden)
            throw std::runtime_error("real-layer oracle received unsupported shared weights");
        std::vector<std::uint16_t> gate_bits(std::size_t(gate_weight.n) * gate_weight.k);
        std::vector<std::uint16_t> down_bits(std::size_t(down_weight.n) * down_weight.k);
        CUDA_CHECK(cudaMemcpy(gate_bits.data(), gate_weight.qdata,
                             gate_bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(down_bits.data(), down_weight.qdata,
                             down_bits.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        std::vector<double> shared_activation(profile.shared.intermediate);
        for (int row = 0; row < profile.shared.intermediate; ++row) {
            double gate = 0.0, up = 0.0;
            for (int k = 0; k < hidden; ++k) {
                const auto x = static_cast<double>(qw::detail::bf16_to_f32(input[k]));
                gate += qw::detail::bf16_to_f32(gate_bits[std::size_t(row) * hidden + k]) * x;
                up += qw::detail::bf16_to_f32(
                          gate_bits[std::size_t(row + profile.shared.intermediate) * hidden + k]) *
                      x;
            }
            if (profile.shared.clamp_limit) {
                const auto limit = static_cast<double>(*profile.shared.clamp_limit);
                gate = std::min(gate, limit);
                up = std::clamp(up, -limit, limit);
            }
            shared_activation[row] =
                (gate >= 0.0 ? gate / (1.0 + std::exp(-gate))
                             : gate * std::exp(gate) / (1.0 + std::exp(gate))) * up;
        }
        const auto shared_coefficient =
            1.0 / (1.0 + std::exp(-std::clamp(logits[experts], -700.0, 700.0)));
        for (int row = 0; row < hidden; ++row) {
            double value = 0.0;
            for (int k = 0; k < profile.shared.intermediate; ++k)
                value += qw::detail::bf16_to_f32(
                             down_bits[std::size_t(row) * profile.shared.intermediate + k]) *
                         shared_activation[k];
            result[row] += shared_coefficient * value;
        }
    }

    for (auto& value : result) {
        auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(value));
        bits += 0x7fffU + ((bits >> 16U) & 1U);
        const auto bf16 = static_cast<std::uint16_t>(bits >> 16U);
        value = qw::detail::bf16_to_f32(bf16);
    }
    return result;
}

void rank_loading(const artifact::Reader& reader, DeviceContext& device, std::uint32_t rank,
                  bool load, std::uint32_t ep) {
    EngineOptions options;
    options.use_cuda_graph   = false;
    options.enable_vision    = true;
    options.max_context      = 4096;
    options.prefill_chunk    = 4096;
    options.max_concurrency  = 1;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(512ULL << 20);
    options.context_cache.host_capacity_bytes = 0;
    options.speculative = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 3};
    auto definition     = q4::read_definition(reader, models::load_options(options));
    DeploymentOptions deployment;
    deployment.control_host = "127.0.0.1";
    deployment.parallel     = {2, ep, 1, 1};
    deployment.nodes        = {{0, {}, "/worker", "/model.ninfer", {}},
                               {1, "rank1", "/worker", "/model.ninfer", {}}};
    deployment.ranks        = {{0, 0, 0}, {1, 1, 0}};
    const auto parallel     = q4::plan_parallel(*definition, deployment);
    const auto& r           = parallel.ranks.at(rank);
    auto selected           = q4::resolve_rank_load(reader, definition, r, device.device());
    auto profile            = q4::execution::describe_model(*definition, selected.weight_data(),
                                                            selected.expert_data(), r);
    const auto& g           = std::get<q4::execution::GdnProfile>(profile->text.layers[0].mixer);
    require(g.output.n == 2560 && g.output.k == 3072 && profile->local.gdn.key_heads == 8 &&
                profile->local.gdn.value_heads == 24 && profile->local.qsa.query_heads == 12 &&
                profile->local.qsa.kv_heads == 1 && profile->text.head.n == 124160 &&
                profile->text.layers[0].moe.math.routed.experts == 512,
            "rank-local geometry or global routing domain differs");
    for (std::size_t b = 0; b < r.experts.size(); ++b) {
        std::vector<int> owners(512);
        for (const auto& p : parallel.ranks) {
            require(p.experts[b].ids.size() == 512 / ep, "expert partition size");
            for (const auto e : p.experts[b].ids) ++owners.at(e);
        }
        require(std::all_of(owners.begin(), owners.end(),
                            [&](int n) { return n == (ep == 1 ? 2 : 1); }),
                "expert ID coverage");
    }
    require(
        profile->text.layers[0].moe.math.routed.ffn.intermediate == (ep == 1 ? 320 : 640) &&
            profile->mtp->layer.moe.math.routed.ffn.intermediate == (ep == 1 ? 320 : 640) &&
            profile->mtp->layer.moe.math.routed.ffn.gate_up_type ==
                (ep == 1 ? QType::FP8_E4M3FN_BLOCK64X64_BF16 : QType::FP8_E4M3FN_BLOCK128X128_BF16),
        "local expert geometry or FP8 refinement differs");
    const auto weight_bytes = selected.materialization().device_capacity_bytes;
    auto plan               = q4::plan_execution(profile, {}, options, device, 16 * GiB);
    require(plan && !plan->ple_prepare_bytes && !plan->ple_upload.device_bytes &&
                plan->profile->local.gdn.value_heads == 24,
            "rank state/PLE planning differs");
    std::cout << "rank=" << rank << " weights=" << weight_bytes
              << " runtime=" << plan->device_reservation_bytes()
              << " state_stride=" << plan->persistent.state.images.image_bytes
              << " workspace=" << plan->workspace.bytes << std::endl;
    if (!load) return;
    auto model = q4::materialize_model(std::move(selected), device);
    q4::execution::Parameters parameters(*model, profile);
    require(parameters.mtp->head.weight.qdata == parameters.text.head.weight.qdata &&
                parameters.mtp->embedding.qdata == parameters.text.embedding.qdata,
            "rank Main/MTP physical sharing");
    for (std::size_t b = 0; b < r.experts.size(); ++b) {
        const auto* bank = model->resident_experts({static_cast<std::uint32_t>(b)});
        require(bank && bank->view().count == 512, "resident global domain");
        std::vector<std::uint64_t> offsets(512);
        if (ep == 2) {
            CUDA_CHECK(cudaMemcpy(offsets.data(), bank->view().gate_up.offsets, offsets.size() * 8,
                                  cudaMemcpyDeviceToHost));
            for (std::uint32_t e = 0; e < 512; ++e)
                require((offsets[e] != kExpertAbsentOffset) ==
                            std::binary_search(r.experts[b].ids.begin(), r.experts[b].ids.end(), e),
                        "indexed bank ownership");
        } else {
            require(!bank->view().gate_up.offsets &&
                        bank->subset().kind == ops::ExpertSubsetKind::All,
                    "TP expert bank is not affine");
            for (std::uint32_t e = 0; e < 512; ++e)
                offsets[e] = e * bank->view().gate_up.code_stride;
        }
        const auto e        = r.experts[b].ids[r.experts[b].ids.size() / 2];
        const auto& source  = definition->expert_banks()[b].gate_up;
        const auto& object  = reader.directory().tensor(source.object);
        const auto expected = reader.read_range(
            object.offset + source.layout.codes.offset + e * source.layout.codes.expert_stride +
                r.experts[b].intermediate.begin * source.layout.codes.expert_bytes /
                    source.layout.region.rows,
            64);
        std::array<std::byte, 64> actual;
        CUDA_CHECK(cudaMemcpy(actual.data(), bank->view().gate_up.codes + offsets[e], 64,
                              cudaMemcpyDeviceToHost));
        require(std::equal(actual.begin(), actual.end(), expected.begin()),
                "resident expert source words");
    }
    require(std::get<q4::DevicePleTable>(model->ple_table()).shards.size() == 64 &&
                model->storage_stats().host_capacity_bytes == 0,
            "rank PLE residency or unexpected Host weight backing");
    q4::Resources resources(parameters, std::move(*plan), device);
    require(!resources.ple_prepare && !resources.ple_upload && !resources.experts,
            "rank created Host producers/cache");
    const auto& stats = model->storage_stats();
    std::cout << "loaded rank=" << rank << " read=" << stats.read_bytes
              << " h2d=" << stats.h2d_bytes << " staging=" << stats.peak_staging_bytes
              << " seconds=" << stats.upload_seconds << " metadata=" << model->metadata_bytes()
              << std::endl;
}

void placement(const q4::ResolvedLoadPlan& load, bool host) {
    const auto& plan = load.materialization();
    require(load.definition().ple_parameters().rows == 320001536, "PLE table extent");
    if (!host) {
        require(plan.registered_host_objects.empty() && plan.registered_host_bytes == 0,
                "resident weights retained Host expert backing");
        require(plan.device_capacity_bytes > 63 * GiB && plan.device_capacity_bytes < 76 * GiB,
                "resident weight capacity differs");
        return;
    }
    require(plan.registered_host_objects.size() == 96,
            "Main banks must have one Host backing per parent");
    require(plan.registered_host_bytes >= 67947921408ULL &&
                plan.registered_host_bytes < 67947921408ULL + 2 * 1024 * 1024,
            "Main Host bank capacity differs");
    require(plan.device_capacity_bytes < 11 * GiB, "PLE or Main banks were placed on GPU");
}

void payloads(const artifact::Reader& reader, const q4::ModelShard& model) {
    const auto& w = model.definition().weights();
    const auto& g = std::get<q4::GdnWeights>(w.text.layers[0].mixer);
    const auto& q = std::get<q4::QsaWeights>(w.text.layers[3].mixer);
    for (const auto id :
         {w.text.embedding, w.text.head, g.convolution, g.query, q.key, q.gate,
          w.text.layers.back().moe.gate, w.mtp->layer.moe.down, w.vision->layers[0].fc1}) {
        const auto& weight = model.weight(id);
        const auto& parent = *weight.view.parts.front().parent;
        const auto& object = reader.directory().tensor(
            reader.directory().object_index.at(weight.source_objects.front()));
        for (const auto offset : {std::uint64_t{0}, parent.geometry.bytes - 64}) {
            std::array<std::byte, 64> expected{}, actual{};
            reader.read_into(object.offset + offset, expected);
            if (parent.device == Device::cpu())
                std::memcpy(actual.data(), parent.data + offset, 64);
            else
                CUDA_CHECK(
                    cudaMemcpy(actual.data(), parent.data + offset, 64, cudaMemcpyDeviceToHost));
            require(expected == actual, "materialized words differ from the artifact");
        }
    }
}

void check_parameters(const q4::ModelShard& model, const q4::execution::Parameters& p, bool host) {
    require(model.expert_banks().size() == 49, "Main/MTP bank count");
    for (std::size_t l = 0; l < 48; ++l) {
        const auto& source = model.expert_banks()[l];
        if (host) {
            require(source.gate_up.parent->device == Device::cpu() &&
                        source.gate_up_upload.device_address,
                    "Main source must remain registered Host memory");
            require(!p.text.layers[l].moe.resident.count,
                    "Host Main Parameters acquired mutable GPU cache");
        } else {
            require(source.gate_up.parent->device.type() == DeviceType::CUDA &&
                        p.text.layers[l].moe.resident.count == 512,
                    "Main resident bank was not bound");
        }
        for (std::uint32_t e : {0U, 17U, 511U}) {
            const auto gu   = matrix_weight(source.gate_up, e);
            const auto down = matrix_weight(source.down, e);
            require(gu.n == 1280 && gu.k == 2560 && down.n == 2560 && down.k == 640,
                    "expert native geometry");
            require(gu.weight_scale == source.gate_up.matrix_scale(e) &&
                        down.weight_scale == source.down.matrix_scale(e),
                    "expert divisor addressing");
        }
    }
    require(p.mtp && p.vision && p.mtp->embedding.qdata == p.text.embedding.qdata &&
                p.mtp->head.weight.qdata == p.text.head.weight.qdata,
            "MTP shared matrices");
    require(p.mtp->layer.moe.resident.count == 512 &&
                p.mtp->layer.moe.profile.routed.ffn.gate_up_policy == ops::LinearPolicy::AllowA8,
            "MTP resident FP8 bank");
    const auto& gdn = std::get<q4::execution::GdnParameters>(p.text.layers[0].mixer);
    require(gdn.convolution.ne[0] == 10240 && gdn.convolution.ne[1] == 4 &&
                p.text.ple.convolution.ne[0] == 10240,
            "stored convolution layout");
}

void ple_window(q4::Resources& resources, const q4::ModelShard& model, DeviceContext& device) {
    const std::array<std::int32_t, 3> ids{123, 456, 789};
    const auto eos = model.definition().config().text.ple.eos;
    const q4::ple::InputRow row{{1, 1, 0, {eos, eos}}, ids};
    const auto ticket = resources.ple_prepare->prepare(3, {&row, 1});
    require(ticket.has_value(), "PLE slot unavailable");
    const std::array<std::uint32_t, 1> valid{3};
    resources.ple_prepare->seal(*ticket, valid);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        const auto status = resources.ple_prepare->poll(*ticket);
        if (status.error) std::rethrow_exception(status.error);
        if (status.status == q4::ple::disk::PrepareStatus::HostReady && status.drained) break;
        require(std::chrono::steady_clock::now() < deadline, "PLE timed out");
        std::this_thread::yield();
    }
    const auto host = resources.ple_prepare->embedding(*ticket);
    resources.ple_upload->upload(host, resources.ple_transfer());
    auto binding = resources.ple_upload->acquire(*ticket, device.execution_view());
    binding.ready.wait(device.stream());
    auto e = binding.values;
    DeviceBuffer copy(device.device(), e.bytes());
    copy_bytes_async(copy.span(), {e.data, e.bytes(), device.device()}, device.execution_view());
    binding.consumed.record(device.stream());
    resources.ple_upload->consumer_submitted(*ticket, device.execution_view());
    device.synchronize();
    std::vector<std::byte> bytes(e.bytes());
    copy.copy_to_host(bytes.data(), bytes.size());
    require(std::memcmp(bytes.data(), host.values.data, bytes.size()) == 0,
            "PLE upload changed words");
    require(resources.ple_upload->try_release(*ticket), "PLE consumer did not retire");
    resources.ple_prepare->release(*ticket);
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) return 77;
    try {
        DeviceContext device;
        bool materialize = false, host = false, stream_one_layer = false;
        int rank = -1;
        std::uint32_t ep = 2;
        for (int i = 2; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--materialize")
                materialize = true;
            else if (arg == "--host-experts")
                host = true;
            else if (arg == "--stream-one-layer")
                stream_one_layer = true;
            else if (arg == "--ep" && i + 1 < argc)
                ep = std::stoul(argv[++i]);
            else if (arg == "--rank" && i + 1 < argc)
                rank = std::stoi(argv[++i]);
            else
                throw std::invalid_argument(
                    "expected --materialize, --host-experts, --stream-one-layer, --rank N, or --ep N");
        }
        if (rank >= 0) {
            artifact::Reader reader(argv[1]);
            rank_loading(reader, device, rank, materialize, ep);
            return 0;
        }
        const auto plan_for = [&](const auto& profile, const auto& geometry,
                                  const EngineOptions& options, std::size_t available,
                                  bool disk = false) {
            auto prototype = disk ? q4::describe_moe_plan(*profile, options.moe, true)
                                  : host ? q4::describe_moe_plan(*profile, options.moe) : nullptr;
            require(!(host || disk) || prototype != nullptr,
                    "target does not provide the requested expert schedule");
            return q4::plan_execution(profile, geometry, options, device, available, prototype);
        };
#if NINFER_TARGET_JETSON_ORIN
        if (materialize)
            throw std::invalid_argument("Flash-Next resident materialization exceeds Orin RAM");
        for (const bool mtp : {false, true}) {
            EngineOptions options;
            options.use_cuda_graph  = false;
            options.max_context     = 4096;
            options.prefill_chunk   = 128;
            options.context_capacity =
                ContextCapacityPolicy::explicit_capacity(512ULL << 20);
            options.context_cache.host_capacity_bytes = 0;
            if (mtp) {
                options.speculative = {.backend = SpeculativeBackend::Mtp, .draft_tokens = 5};
            }
            artifact::Reader reader(argv[1]);
            auto definition = q4::read_definition(reader, models::load_options(options));
            auto load = q4::resolve_load(reader, definition, device.device());
            auto profile = q4::execution::describe_model(*definition);
            if (definition->expert_banks().size() != (mtp ? 49 : 48) ||
                profile->text.layers.size() != 48 || profile->mtp.has_value() != mtp) {
                throw std::runtime_error(
                    "Flash-Next definition binding differs: banks=" +
                    std::to_string(definition->expert_banks().size()) + " layers=" +
                    std::to_string(profile->text.layers.size()) + " mtp=" +
                    std::to_string(profile->mtp.has_value()));
            }
            placement(load, false);
            auto disk_load = q4::resolve_load(reader, definition, device.device(), Device::cpu(),
                                              load.ple_table_owner(), true);
            require(disk_load.materialization().device_capacity_bytes < 16 * GiB &&
                        disk_load.materialization().registered_host_bytes == 0,
                    "disk expert planning still materializes or registers the expert banks");
            std::vector<artifact::ObjectHandle> expert_objects;
            for (const auto& bank : definition->expert_banks()) {
                expert_objects.push_back(bank.gate_up.object);
                expert_objects.push_back(bank.down.object);
            }
            auto file_backing = std::make_shared<artifact::FileBacking>(
                artifact::prepare_file_backing(reader, expert_objects));
            const auto disk_sources =
                models::moe::bind_file_banks(definition->expert_banks(), file_backing);
            for (const auto index : {std::size_t{0}, std::size_t{47},
                                     definition->expert_banks().size() - 1}) {
                const auto& source = definition->expert_banks()[index].gate_up;
                const auto plane  = source.layout.codes;
                const auto expert = std::uint32_t{511};
                std::array<std::byte, 64> actual{};
                disk_sources[index].gate_up_read(
                    plane.offset + expert * plane.expert_stride, actual);
                const auto& object = reader.directory().tensor(source.object);
                const auto expected = reader.read_range(
                    object.offset + plane.offset + expert * plane.expert_stride, actual.size());
                require(std::equal(actual.begin(), actual.end(), expected.begin()) &&
                            disk_sources[index].gate_up.matrix_scale(expert) > 0,
                        "file-backed Flash-Next expert range or matrix scale differs");
            }
            auto disk_plan = plan_for(profile, load.ple_table().geometry(), options, 16 * GiB, true);
            require(disk_plan.has_value() && disk_plan->moe &&
                        disk_plan->moe->description().cache_pinned_bytes > 0 &&
                        disk_plan->moe->description().cache_bytes > 0,
                    "Orin must plan bounded file-backed Flash-Next expert execution");
            std::cout << "mtp=" << mtp << " weights="
                      << load.materialization().device_capacity_bytes << " expert_cache="
                      << disk_plan->moe->description().cache_bytes << " prefill=tokenwise NVFP4\n";
            if (stream_one_layer && !mtp) {
                auto model = q4::materialize_model(std::move(disk_load), device);
                q4::execution::Parameters parameters(*model, profile);
                require(parameters.text.layers.size() > 1,
                        "Flash-Next PLE output has no following text layer");
                for (const int tokens : {1, 3, 5})
                    qualify_real_ple(parameters.text.ple,
                                     parameters.text.layers[1].attention_hc.norm, device, tokens);
                const auto qsa_layer = std::find_if(
                    parameters.text.layers.begin(), parameters.text.layers.end(),
                    [](const auto& layer) {
                        return std::holds_alternative<q4::execution::QsaParameters>(layer.mixer);
                    });
                require(qsa_layer != parameters.text.layers.end(),
                        "Flash-Next has no main QSA projection to qualify");
                const auto* main_qsa = std::get_if<q4::execution::MainQsaProjection>(
                    &std::get<q4::execution::QsaParameters>(qsa_layer->mixer).input);
                require(main_qsa != nullptr,
                        "Flash-Next QSA layer does not use the registered main projections");
                const auto& qsa_config = profile->config.text.qsa;
                const std::array<std::int32_t, 2> query_gate_rows{
                    static_cast<std::int32_t>(qsa_config.query_width()),
                    static_cast<std::int32_t>(qsa_config.query_width())};
                const std::array<std::int32_t, 4> kv_index_rows{
                    static_cast<std::int32_t>(qsa_config.kv_width()),
                    static_cast<std::int32_t>(qsa_config.kv_width()),
                    static_cast<std::int32_t>(qsa_config.index_query_width()),
                    static_cast<std::int32_t>(qsa_config.index_key_width())};
                qualify_real_qsa_projection(main_qsa->query_gate, query_gate_rows, device,
                                            "query_gate");
                qualify_real_qsa_projection(main_qsa->kv_index, kv_index_rows, device,
                                            "kv_index");
                std::vector<const models::moe::Parameters*> layers;
                for (const auto& layer : parameters.text.layers) layers.push_back(&layer.moe);
                auto schedule = disk_plan->moe->create(layers, model->expert_banks(), device,
                                                       models::moe::ResourcePurpose::Execution);
                const auto bank = parameters.text.layers.front().moe.bank;
                auto activity = schedule->begin(1);
                while (!schedule->try_before_layer(bank)) std::this_thread::yield();
                const auto hidden_size = profile->config.text.hidden;
                DeviceBuffer hidden_buffer(device.device(), std::size_t(hidden_size) * 2);
                DeviceBuffer output_buffer(device.device(), std::size_t(hidden_size) * 2);
                DeviceBuffer workspace_buffer(
                    device.device(), std::max<std::size_t>(256, disk_plan->moe->workspace_bytes(
                                                                    bank, 1, 1)));
                std::vector<std::uint16_t> input(hidden_size);
                for (std::size_t i = 0; i < input.size(); ++i) {
                    const float value = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0F;
                    auto bits = std::bit_cast<std::uint32_t>(value);
                    bits += 0x7fffU + ((bits >> 16U) & 1U);
                    input[i] = static_cast<std::uint16_t>(bits >> 16U);
                }
                auto alternate = input;
                for (auto& value : alternate) value ^= 0x8000U;
                const auto hidden_width = static_cast<std::int32_t>(hidden_size);
                Tensor hidden(hidden_buffer.span(), DType::BF16, {hidden_width, 1});
                Tensor output(output_buffer.span(), DType::BF16, {hidden_width, 1});
                MemoryArena workspace(workspace_buffer.span());
                const auto run_layer = [&](const std::vector<std::uint16_t>& values) {
                    hidden_buffer.copy_from_host(values.data(), hidden_buffer.size());
                    bool complete = schedule->start(bank, hidden, output, workspace, {});
                    while (!complete) {
                        complete = schedule->poll();
                        std::this_thread::yield();
                    }
                    schedule->synchronize();
                };
                run_layer(input);
                const auto first_upload = schedule->uploaded_bytes();
                run_layer(input);
                require(schedule->uploaded_bytes() == first_upload,
                        "repeated file-backed route did not reuse its expert cache entries");
                run_layer(alternate);
                const auto alternate_upload = schedule->uploaded_bytes();
                require(alternate_upload > first_upload,
                        "alternate real route did not miss in the bounded expert cache");
                run_layer(alternate);
                require(schedule->uploaded_bytes() == alternate_upload,
                        "repeated alternate route did not reuse its expert cache entries");
                run_layer(input);
                const auto refilled_upload = schedule->uploaded_bytes();
                require(refilled_upload > alternate_upload,
                        "returning to the original real route did not refill evicted experts");
                run_layer(input);
                require(schedule->uploaded_bytes() == refilled_upload,
                        "refilled real route did not remain resident for its next use");
                auto graph_definition = schedule->capture_layer(
                    bank, device.execution_view(), 1, [] {},
                    [&](const models::moe::GpuCall& call) {
                        call(hidden, output, workspace, {});
                    },
                    [] {}, {});
                auto graph = graph_definition->instantiate();
                graph->upload(device.execution_view().stream());
                hidden_buffer.copy_from_host(alternate.data(), hidden_buffer.size());
                bool graph_complete = graph->start(output, device.execution_view());
                while (!graph_complete) {
                    graph_complete = graph->poll();
                    std::this_thread::yield();
                }
                schedule->synchronize();
                require(schedule->uploaded_bytes() > refilled_upload,
                        "file-backed Graph route did not stage its changed expert set");
                std::vector<std::uint16_t> values(hidden_size);
                output_buffer.copy_to_host(values.data(), output_buffer.size());
                require(std::any_of(values.begin(), values.end(), [](auto value) {
                            const auto fp = std::bit_cast<float>(std::uint32_t(value) << 16);
                            return std::isfinite(fp) && fp != 0.0F;
                        }),
                        "file-backed Flash-Next MoE produced no finite nonzero output");
                const auto& source = disk_sources.at(bank.index);
                const auto expected = real_nvfp4_moe_oracle(
                    parameters.text.layers.front().moe, source, source.gate_up_read,
                    source.down_read, alternate);
                std::vector<double> actual(values.size());
                std::transform(values.begin(), values.end(), actual.begin(), [](auto value) {
                    return static_cast<double>(test::quantized_weight::detail::bf16_to_f32(value));
                });
                constexpr test::ReductionCriterion criterion{0.25, 1.0e-7, 0.35};
                const auto stats = test::compute_reduction_stats(
                    actual.data(), expected.data(), static_cast<std::int64_t>(actual.size()));
                require(test::reduction_passes(stats, static_cast<std::int64_t>(actual.size()),
                                               criterion),
                        "real file-backed NVFP4 Graph MoE output differs from decoded-weight FP64 oracle");
                std::cout << "real NVFP4 oracle rel_l2=" << stats.relative_l2
                          << " max_abs=" << stats.maximum_absolute_error << '\n';
                std::cout << "PASS one real file-backed NVFP4 MoE layer; uploaded="
                          << schedule->uploaded_bytes() << " bytes\n";
            }
        }
        std::cout << (stream_one_layer ? "PASS V3 metadata binding and one-layer execution\n"
                                      : "PASS V3 metadata binding; no materialization or inference\n");
        return 0;
#endif
        std::unique_ptr<q4::ModelShard> model;
        std::shared_ptr<const q4::execution::ModelProfile> profile;
        std::optional<q4::ExecutionPlan> selected;
        {
            artifact::Reader reader(argv[1]);
            {
                auto scoring             = EngineOptions{};
                scoring.purpose          = EnginePurpose::CausalScoring;
                scoring.use_cuda_graph   = false;
                scoring.enable_vision    = false;
                scoring.max_context      = 4096;
                scoring.prefill_chunk    = 4096;
                scoring.context_capacity = ContextCapacityPolicy::explicit_capacity(512ULL << 20);
                scoring.context_cache    = {.enabled = false, .host_capacity_bytes = 0};
                auto definition = q4::read_definition(reader, models::load_options(scoring));
                auto load       = q4::resolve_load(reader, definition, device.device(),
                                             host ? Device::cpu() : device.device());
                auto p          = q4::execution::describe_model(*definition);
                const auto plan = plan_for(p, load.ple_table().geometry(), scoring, 8 * GiB);
                require(plan && !plan->persistent.buffers.logits &&
                            !plan->persistent.buffers.token_counts &&
                            plan->pinned_output_bytes == 4096 &&
                            plan->workspace.finish >= 248320ULL * 1024 * 2 + 8192,
                        "scoring resource lifetime");
            }
            for (const bool vision : {false, true})
                for (const bool mtp : {false, true}) {
                    EngineOptions options;
                    options.use_cuda_graph  = false;
                    options.enable_vision   = vision;
                    options.max_context     = 4096;
                    options.prefill_chunk   = 4096;
                    options.max_concurrency = 1;
                    options.context_capacity =
                        ContextCapacityPolicy::explicit_capacity(512ULL << 20);
                    options.context_cache.host_capacity_bytes =
                        NINFER_SHARED_PHYSICAL_MEMORY ? 0 : 256 * 1024 * 1024;
                    if (mtp) {
                        options.speculative.backend      = SpeculativeBackend::Mtp;
                        options.speculative.draft_tokens = 5;
                    }
                    auto definition = q4::read_definition(reader, models::load_options(options));
                    auto load       = q4::resolve_load(reader, definition, device.device(),
                                                 host ? Device::cpu() : device.device());
                    profile         = q4::execution::describe_model(*definition);
                    placement(load, host);
                    const auto weights   = load.materialization().device_capacity_bytes;
                    const auto available = host ? 16 * GiB - weights - 256 * 1024 * 1024 : 16 * GiB;
#if NINFER_TARGET_JETSON_ORIN
                    if (mtp) {
                        bool rejected = false;
                        try {
                            (void)plan_for(profile, load.ple_table().geometry(), options,
                                           available);
                        } catch (const std::invalid_argument&) { rejected = true; }
                        require(rejected, "Orin rejects the unqualified FP8 MTP expert route");
                        continue;
                    }
#endif
                    selected = plan_for(profile, load.ple_table().geometry(), options, available);
                    require(selected.has_value(), "runtime plan must fit the supplied capacity");
                    require(selected->ple.max_positions == 4096,
                            "PLE Mmax differs from actual call capacity");
                    require(selected->profile->mtp.has_value() == mtp &&
                                selected->persistent.mtp.has_value() == mtp,
                            "MTP selection leaked resources");
                    if (vision) {
                        auto long_context        = options;
                        long_context.max_context = 65536;
                        const auto workspace     = q4::execution::plan_workspace(
                            *profile, long_context, 4096, device.multiprocessor_count(), nullptr);
                        require(workspace.vision->max_features > 0 &&
                                    workspace.vision->device_bytes() <=
                                        long_context.vision_memory_budget_bytes,
                                "Vision execution exceeded its configured budget");
                    }
                    std::cout << "vision=" << vision << " mtp=" << mtp << " weights=" << weights
                              << " state_stride=" << selected->persistent.state.images.image_bytes
                              << " persistent=" << selected->persistent.bytes
                              << " workspace=" << selected->workspace.bytes << " cache="
                              << (selected->moe ? selected->moe->description().cache_bytes : 0)
                              << " windows="
                              << (selected->moe ? selected->moe->description().window_count : 0)
                              << " runtime=" << selected->device_bytes()
                              << " ple_host=" << selected->ple_prepare_bytes << '\n';
                    if (vision == mtp) {
                        auto multi             = options;
                        multi.max_concurrency  = 8;
                        multi.context_capacity = ContextCapacityPolicy::automatic();
                        for (const auto format : {KvCacheStorage::BFloat16,
                                                  KvCacheStorage::Int8Group64}) {
                            multi.kv_cache = format;
                            auto plan =
                                plan_for(profile, load.ple_table().geometry(), multi, available);
                            require(plan &&
                                        plan->capacity.device_bytes >=
                                            plan->capacity.minimum_device_bytes &&
                                        plan->ple.max_positions == 4096,
                                    "mixed-batch automatic capacity");
                            require(plan->device_reservation_bytes() <= available,
                                    "execution exceeds Device budget");
                        }
#if NINFER_TARGET_JETSON_ORIN
                        for (const auto format : {KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                                  KvCacheStorage::Nvfp4Group16,
                                                  KvCacheStorage::Fp8KeyNvfp4Value}) {
                            multi.kv_cache = format;
                            bool rejected = false;
                            try {
                                (void)plan_for(profile, load.ple_table().geometry(), multi,
                                               available);
                            } catch (const std::invalid_argument&) { rejected = true; }
                            require(rejected,
                                    "Orin rejects paged KV formats without a qualified consumer");
                        }
#else
                        for (const auto format : {KvCacheStorage::Fp8E4m3fnGroup128Ue8m0,
                                                  KvCacheStorage::Nvfp4Group16,
                                                  KvCacheStorage::Fp8KeyNvfp4Value}) {
                            multi.kv_cache = format;
                            auto plan =
                                plan_for(profile, load.ple_table().geometry(), multi, available);
                            require(plan &&
                                        plan->capacity.device_bytes >=
                                            plan->capacity.minimum_device_bytes &&
                                        plan->ple.max_positions == 4096,
                                    "mixed-batch automatic capacity");
                            require(plan->device_reservation_bytes() <= available,
                                    "execution exceeds Device budget");
                        }
#endif
                        require(!plan_for(profile, load.ple_table().geometry(), options, 1),
                                "insufficient Device budget admitted");
                    }
                    if (vision && mtp && materialize) {
                        model = q4::materialize_model(std::move(load), device);
                        payloads(reader, *model);
                    }
                }
        }
        if (materialize) {
            const q4::execution::Parameters parameters(*model, profile);
            check_parameters(*model, parameters, host);
            q4::Resources resources(parameters, std::move(*selected), device);
            ple_window(resources, *model, device);

            struct rusage usage {};

            getrusage(RUSAGE_SELF, &usage);
            const auto& s = model->storage_stats();
            std::cout << "loaded host=" << s.host_capacity_bytes
                      << " registered=" << s.registered_host_bytes
                      << " gpu=" << s.device_capacity_bytes << " staging=" << s.peak_staging_bytes
                      << " read=" << s.read_bytes << " rss_kib=" << usage.ru_maxrss << '\n';
        }
        std::cout << "PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
