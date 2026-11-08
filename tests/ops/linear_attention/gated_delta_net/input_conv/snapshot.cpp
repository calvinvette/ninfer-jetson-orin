#include "support.h"
#include "ops/state_fixture.h"
#include "core/weight.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/device.h"
#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 fused projection/conv/snapshot Op.
constexpr ReductionCriterion kGdnInputProjConvSnapshotA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};
// Q8 A16 includes BF16 weight reconstruction and, in the composed arithmetic
// profile, BF16 projection storage before convolution. The FP64 oracle keeps
// both intermediates ideal. Allow two BF16 unit roundoffs in L2 for the projection
// and nonlinear output, with a finite gross cap; fused routes avoid the inner cast.
constexpr ReductionCriterion kQ8GdnInputProjConvSnapshotA16Tolerance{2.0 / 256.0, 1.0 / 256.0,
                                                                     2.0 / 256.0};
constexpr ReductionCriterion kGdnInputProjConvSnapshotA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr ReductionCriterion kFp8GdnInputProjConvSnapshotA16Tolerance{1.0 / 256.0, 1.0 / 256.0,
                                                                      2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjConvSnapshotA8Tolerance{0.04, 1.0 / 256.0, 0.06};

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

// Full-reference cases compare every output row and state channel. Sampled cases retain
// the same complete temporal formula at fixed logical coordinates.
inline constexpr std::int32_t kFullReferenceSamples = 1 << 20;

std::int32_t snapshot_sample_count(std::int32_t tokens, bool full_reference = false) {
    if (full_reference) { return kFullReferenceSamples; }
    return tokens <= 4 ? 32 : 7;
}

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<float> make_conv_weight(std::int32_t channels, std::uint32_t seed) {
    std::vector<float> weight(static_cast<std::size_t>(channels) * 4);
    fill_uniform(weight, seed, -0.02F, 0.02F);
    round_to_bf16(weight);
    return weight;
}

std::vector<std::uint16_t> make_state(std::int32_t channels, std::int32_t slots,
                                      std::int32_t initial_slot, std::uint32_t seed) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> state(slot_stride * slots, 0xffffU);
    std::vector<float> initial(slot_stride);
    fill_uniform(initial, seed, -0.05F, 0.05F);
    round_to_bf16(initial);
    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * slot_stride;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        state[initial_base + index] = f32_to_bf16(initial[index]);
    }
    return state;
}

struct SnapshotOracle {
    std::vector<double> query;
    std::vector<double> key;
    std::vector<double> value;
    std::vector<double> state;
};

template <class Projection>
SnapshotOracle
snapshot_oracle(std::int32_t value_rows, std::int32_t tokens, const std::vector<float>& conv_weight,
                std::span<const std::uint16_t> initial_state, Projection&& projection,
                bool full_reference = false, int qk_rows = kQueryRows) {
    const std::int32_t channels = qk_rows + qk_rows + value_rows;
    SnapshotOracle oracle;
    const std::int32_t sample_count                = snapshot_sample_count(tokens, full_reference);
    const std::vector<std::int32_t> query_rows     = sampled_rows(qk_rows, sample_count);
    const std::vector<std::int32_t> key_rows       = sampled_rows(qk_rows, sample_count);
    const std::vector<std::int32_t> value_selected = sampled_rows(value_rows, sample_count);
    const std::size_t sampled_channel_count =
        query_rows.size() + key_rows.size() + value_selected.size();
    oracle.query.reserve(query_rows.size() * static_cast<std::size_t>(tokens));
    oracle.key.reserve(key_rows.size() * static_cast<std::size_t>(tokens));
    oracle.value.reserve(value_selected.size() * static_cast<std::size_t>(tokens));
    oracle.state.reserve(sampled_channel_count * 3 * static_cast<std::size_t>(tokens));

    const auto evaluate = [&](std::int32_t global_row, std::vector<double>& output) {
        double state0        = bf16_to_f32(initial_state[global_row]);
        double state1        = bf16_to_f32(initial_state[channels + global_row]);
        double state2        = bf16_to_f32(initial_state[2 * channels + global_row]);
        const double weight0 = conv_weight[global_row];
        const double weight1 = conv_weight[channels + global_row];
        const double weight2 = conv_weight[2 * channels + global_row];
        const double weight3 = conv_weight[3 * channels + global_row];
        for (std::int32_t token = 0; token < tokens; ++token) {
            const double projected = projection(global_row, token);
            const double convolved =
                weight0 * state0 + weight1 * state1 + weight2 * state2 + weight3 * projected;
            output.push_back(silu_fp64(convolved));
            const double persisted_projected =
                bf16_to_f32(f32_to_bf16(static_cast<float>(projected)));
            oracle.state.push_back(state1);
            oracle.state.push_back(state2);
            oracle.state.push_back(persisted_projected);
            state0 = state1;
            state1 = state2;
            state2 = persisted_projected;
        }
    };

    for (const std::int32_t row : query_rows) { evaluate(row, oracle.query); }
    for (const std::int32_t row : key_rows) { evaluate(qk_rows + row, oracle.key); }
    for (const std::int32_t row : value_selected) {
        evaluate(qk_rows + qk_rows + row, oracle.value);
    }
    return oracle;
}

std::vector<double> gather_state(const std::vector<std::uint16_t>& full, std::int32_t channels,
                                 std::int32_t value_rows, std::int32_t tokens,
                                 std::int32_t snapshot_base_slot, bool full_reference = false,
                                 int qk_rows = kQueryRows) {
    std::vector<double> gathered;
    const auto append = [&](std::int32_t global_row) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t token_base =
                static_cast<std::size_t>(snapshot_base_slot + token) * 3 * channels;
            for (std::int32_t history = 0; history < 3; ++history) {
                gathered.push_back(bf16_to_f32(
                    full[token_base + static_cast<std::size_t>(history) * channels + global_row]));
            }
        }
    };
    const std::int32_t sample_count = snapshot_sample_count(tokens, full_reference);
    for (const std::int32_t row : sampled_rows(qk_rows, sample_count)) { append(row); }
    for (const std::int32_t row : sampled_rows(qk_rows, sample_count)) { append(qk_rows + row); }
    for (const std::int32_t row : sampled_rows(value_rows, sample_count)) {
        append(qk_rows + qk_rows + row);
    }
    return gathered;
}

int verify_state_effects(std::string_view label, const std::vector<std::uint16_t>& before,
                         const std::vector<std::uint16_t>& after, std::int32_t channels,
                         std::int32_t tokens, std::int32_t slots, std::int32_t snapshot_base_slot) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t slot = snapshot_base_slot; slot < snapshot_base_slot + tokens; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        for (std::size_t index = 0; index < slot_stride; ++index) {
            if (!std::isfinite(bf16_to_f32(after[base + index]))) {
                std::cerr << label << ": state slot " << slot << " was not fully written\n";
                return 1;
            }
        }
    }
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        if (slot >= snapshot_base_slot && slot < snapshot_base_slot + tokens) { continue; }
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        if (!std::equal(before.begin() + static_cast<std::ptrdiff_t>(base),
                        before.begin() + static_cast<std::ptrdiff_t>(base + slot_stride),
                        after.begin() + static_cast<std::ptrdiff_t>(base))) {
            std::cerr << label << ": state slot " << slot << " was modified\n";
            return 1;
        }
    }
    return 0;
}

int verify_snapshot_outputs(
    std::string_view suffix, const GuardedBf16Tensor& query, const GuardedBf16Tensor& key,
    const GuardedBf16Tensor& value, std::int32_t value_rows, std::int32_t tokens,
    const SnapshotOracle& oracle,
    const ReductionCriterion& criterion = kGdnInputProjConvSnapshotA16Tolerance,
    bool full_reference                 = false) {
    int failures                    = 0;
    const std::int32_t sample_count = snapshot_sample_count(tokens, full_reference);
    failures += query.verify_guards("snapshot query" + std::string(suffix));
    failures += key.verify_guards("snapshot key" + std::string(suffix));
    failures += value.verify_guards("snapshot value" + std::string(suffix));
    failures += query.verify_fully_written("snapshot query" + std::string(suffix));
    failures += key.verify_fully_written("snapshot key" + std::string(suffix));
    failures += value.verify_fully_written("snapshot value" + std::string(suffix));
    failures +=
        compare("snapshot query" + std::string(suffix),
                gather_rows(query.values(), kQueryRows, 0, kQueryRows, tokens, sample_count),
                oracle.query, criterion);
    failures += compare("snapshot key" + std::string(suffix),
                        gather_rows(key.values(), kKeyRows, 0, kKeyRows, tokens, sample_count),
                        oracle.key, criterion);
    failures +=
        compare("snapshot value" + std::string(suffix),
                gather_rows(value.values(), value_rows, 0, value_rows, tokens, sample_count),
                oracle.value, criterion);
    return failures;
}

std::vector<double> gather_batch_rows(const std::vector<double>& full, std::int32_t full_rows,
                                      std::int32_t row_count, std::int32_t width,
                                      std::int32_t batch_row, std::int32_t valid,
                                      std::int32_t sample_count) {
    std::vector<double> gathered;
    const std::vector<std::int32_t> selected = sampled_rows(row_count, sample_count);
    gathered.reserve(selected.size() * static_cast<std::size_t>(valid));
    for (const std::int32_t row : selected) {
        for (std::int32_t column = 0; column < valid; ++column) {
            const std::int32_t flat_column = batch_row * width + column;
            gathered.push_back(full[static_cast<std::size_t>(flat_column) * full_rows + row]);
        }
    }
    return gathered;
}

int verify_zero_tail(std::string_view label, const GuardedBf16Tensor& output, std::int32_t rows,
                     std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    if (valid_columns.empty()) { return 0; }
    const std::vector<std::uint16_t> bits = output.bits();
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t column = valid_columns[static_cast<std::size_t>(batch_row)];
             column < width; ++column) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + column) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (bits[base + static_cast<std::size_t>(row)] != 0) {
                    std::cerr << label << ": invalid tail was not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_batched_state_effects(std::string_view label, const std::vector<std::uint16_t>& before,
                                 const std::vector<std::uint16_t>& after, std::int32_t channels,
                                 std::int32_t slots, std::int32_t width, std::int32_t batch,
                                 const std::vector<std::int32_t>& valid_columns,
                                 const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<bool> written(static_cast<std::size_t>(slots), false);
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        const std::int32_t valid =
            valid_columns.empty() ? width : valid_columns[static_cast<std::size_t>(batch_row)];
        for (std::int32_t column = 0; column < valid; ++column) {
            written[static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                             column)] = true;
        }
    }
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        if (!written[static_cast<std::size_t>(slot)]) {
            if (!std::equal(before.begin() + static_cast<std::ptrdiff_t>(base),
                            before.begin() + static_cast<std::ptrdiff_t>(base + slot_stride),
                            after.begin() + static_cast<std::ptrdiff_t>(base))) {
                std::cerr << label << ": unwritten state slot " << slot << " was modified\n";
                return 1;
            }
            continue;
        }
        for (std::size_t index = 0; index < slot_stride; ++index) {
            if (!std::isfinite(bf16_to_f32(after[base + index]))) {
                std::cerr << label << ": state slot " << slot << " was not fully written\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class ProjectQkv, class ProjectZ, class Launch>
int run_batched_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
                     std::int32_t z_rows, std::int32_t width, std::int32_t batch,
                     std::vector<std::int32_t> valid_columns, const std::vector<float>& conv_weight,
                     std::size_t workspace_bytes, const ReductionCriterion& criterion,
                     ProjectQkv&& project_qkv, ProjectZ&& project_z, Launch&& launch,
                     int qk_rows = kQueryRows, bool graph = false, bool full_reference = false) {
    const std::int32_t channels          = qk_rows + qk_rows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t slots             = aggregate_columns + batch;
    const std::size_t slot_stride        = static_cast<std::size_t>(channels) * 3;
    std::vector<float> activation =
        make_bf16_activation(hidden, aggregate_columns, 1103U + width + batch);
    std::vector<std::uint16_t> activation_bits        = bf16_bits(activation);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    std::vector<std::uint16_t> state_before(slot_stride * slots, 0xffffU);
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 0 ? 0 : aggregate_columns + batch_row;
        std::vector<float> initial(slot_stride);
        fill_uniform(initial, 1201U + static_cast<std::uint32_t>(batch_row), -0.05F, 0.05F);
        round_to_bf16(initial);
        const std::size_t base =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) *
            slot_stride;
        for (std::size_t index = 0; index < slot_stride; ++index) {
            state_before[base + index] = f32_to_bf16(initial[index]);
        }
    }

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_snapshot_base = to_device(snapshot_bases);
    DeviceBuffer device_valid;
    if (!valid_columns.empty()) { device_valid = to_device(valid_columns); }
    GuardedBf16Tensor state(channels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(qk_rows, aggregate_columns);
    GuardedBf16Tensor key(qk_rows, aggregate_columns);
    GuardedBf16Tensor value(value_rows, aggregate_columns);
    GuardedBf16Tensor z(z_rows, aggregate_columns);

    Tensor x(device_activation.span(), DType::BF16, {hidden, width, batch});
    Tensor conv(device_conv_weight.span(), DType::BF16, {channels, 4});
    Tensor conv_state(state.span(), DType::BF16, {channels, 3, slots});
    RecordStateFixture history(conv_state, 2);
    const bool source_aliases_snapshots = width == 4 && batch == 2;
    auto initial_offsets                = history.record_offsets(initial_slots);
    if (source_aliases_snapshots)
        for (int row = 0; row < batch; ++row)
            initial_offsets[row] = std::int64_t(initial_slots[row]) * conv_state.nb[2];
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor valid;
    if (!valid_columns.empty()) { valid = Tensor(device_valid.span(), DType::I32, {batch}); }
    Tensor initial(device_initial.span(), DType::I64, {batch});
    const auto source = source_aliases_snapshots ? dense_record_field(conv_state, 2, initial)
                                                 : history.view(initial);
    Tensor snapshot_base(device_snapshot_base.span(), DType::I32, {batch});
    auto destination_storage = to_device(history.record_offsets(snapshot_bases));
    Tensor destination_offsets(destination_storage.span(), DType::I64, {batch});
    Tensor q(query.span(), DType::BF16, {qk_rows, width, batch});
    Tensor k(key.span(), DType::BF16, {qk_rows, width, batch});
    Tensor v(value.span(), DType::BF16, {value_rows, width, batch});
    Tensor z_output(z.span(), DType::BF16, {z_rows, width, batch});
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(1, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    const auto invoke = [&](CudaExecutionView execution) {
        launch(x, conv, source, history.view(destination_offsets), conv_state, valid, snapshot_base,
               q, k, v, z_output, workspace, execution);
    };
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        cuda_synchronize();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto stream_execution = cuda_environment().context.execution_view(stream);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        invoke(stream_execution);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        state.copy_from_bits(state_before);
        history.upload();
        if (!valid_columns.empty()) {
            valid_columns[0] = width > 1 ? width - 1 : 1;
            CUDA_CHECK(cudaMemcpyAsync(device_valid.data(), valid_columns.data(),
                                       device_valid.size(), cudaMemcpyHostToDevice, stream));
        }
        activation      = make_bf16_activation(hidden, aggregate_columns, 1783U + width + batch);
        activation_bits = bf16_bits(activation);
        CUDA_CHECK(cudaMemcpyAsync(device_activation.data(), activation_bits.data(),
                                   device_activation.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        invoke(cuda_environment().default_execution);
        cuda_synchronize();
    }

    int failures                           = 0;
    const std::vector<double> query_values = query.values();
    const std::vector<double> key_values   = key.values();
    const std::vector<double> value_values = value.values();
    const std::vector<double> z_values     = z.values();
    if (width == 1) history.copy_back();
    const std::vector<std::uint16_t> state_after = state.bits();
    std::vector<double> query_actual;
    std::vector<double> query_expected;
    std::vector<double> key_actual;
    std::vector<double> key_expected;
    std::vector<double> value_actual;
    std::vector<double> value_expected;
    std::vector<double> state_actual;
    std::vector<double> state_expected;
    std::vector<double> z_actual;
    std::vector<double> z_expected;
    const auto append = [](std::vector<double>& destination, std::vector<double> source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        const std::int32_t valid_extent =
            valid_columns.empty() ? width : valid_columns[static_cast<std::size_t>(batch_row)];
        const std::size_t initial_base =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) *
            slot_stride;
        const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                           slot_stride);
        const SnapshotOracle oracle = snapshot_oracle(
            value_rows, valid_extent, conv_weight, initial_state,
            [&](std::int32_t row, std::int32_t column) {
                return project_qkv(row, batch_row * width + column, activation);
            },
            full_reference, qk_rows);
        const std::int32_t sample_count = snapshot_sample_count(valid_extent, full_reference);
        append(query_actual, gather_batch_rows(query_values, qk_rows, qk_rows, width, batch_row,
                                               valid_extent, sample_count));
        append(query_expected, oracle.query);
        append(key_actual, gather_batch_rows(key_values, qk_rows, qk_rows, width, batch_row,
                                             valid_extent, sample_count));
        append(key_expected, oracle.key);
        append(value_actual, gather_batch_rows(value_values, value_rows, value_rows, width,
                                               batch_row, valid_extent, sample_count));
        append(value_expected, oracle.value);
        append(state_actual, gather_state(state_after, channels, value_rows, valid_extent,
                                          snapshot_bases[static_cast<std::size_t>(batch_row)],
                                          full_reference, qk_rows));
        append(state_expected, oracle.state);

        std::vector<double> row_z_expected;
        for (const std::int32_t row : sampled_rows(z_rows, sample_count)) {
            for (std::int32_t column = 0; column < width; ++column) {
                row_z_expected.push_back(project_z(row, batch_row * width + column, activation));
            }
        }
        append(z_actual,
               gather_batch_rows(z_values, z_rows, z_rows, width, batch_row, width, sample_count));
        append(z_expected, std::move(row_z_expected));
    }
    failures += compare(std::string(label) + " query", query_actual, query_expected, criterion);
    failures += compare(std::string(label) + " key", key_actual, key_expected, criterion);
    failures += compare(std::string(label) + " value", value_actual, value_expected, criterion);
    failures += compare(std::string(label) + " state", state_actual, state_expected, criterion);
    failures += compare(std::string(label) + " z", z_actual, z_expected, criterion);
    std::cout << label << ": sampled FP64 reference over " << width << " columns x " << batch
              << " batch rows (" << aggregate_columns << " aggregate columns), compared "
              << (query_actual.size() + key_actual.size() + value_actual.size()) << " output values"
              << " + " << state_actual.size() << " state values, failures " << failures << '\n';

    failures += query.verify_guards(std::string(label) + " query");
    failures += key.verify_guards(std::string(label) + " key");
    failures += value.verify_guards(std::string(label) + " value");
    failures += z.verify_guards(std::string(label) + " z");
    failures += history.verify_guards("snapshot history");
    failures += state.verify_guards(std::string(label) + " state");
    failures += query.verify_fully_written(std::string(label) + " query");
    failures += key.verify_fully_written(std::string(label) + " key");
    failures += value.verify_fully_written(std::string(label) + " value");
    failures += z.verify_fully_written(std::string(label) + " z");
    failures += verify_zero_tail(std::string(label) + " query", query, qk_rows, width, batch,
                                 valid_columns);
    failures +=
        verify_zero_tail(std::string(label) + " key", key, qk_rows, width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", value, value_rows, width, batch,
                                 valid_columns);
    failures += verify_batched_state_effects(label, state_before, state_after, channels, slots,
                                             width, batch, valid_columns, snapshot_bases);
    failures += verify_preserved(std::string(label) + " x", device_activation, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures +=
        verify_preserved(std::string(label) + " initial slots", device_initial, initial_offsets);
    failures += verify_preserved(std::string(label) + " snapshot bases", device_snapshot_base,
                                 snapshot_bases);
    if (!valid_columns.empty()) {
        failures +=
            verify_preserved(std::string(label) + " valid columns", device_valid, valid_columns);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens, std::int32_t initial_slot, bool full_reference = false) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation      = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 607U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 613U + tokens);
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.span(), DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.span(), DType::BF16, {kChannels, 4});
    Tensor conv_state(state.span(), DType::BF16, {kChannels, 3, slots});
    RecordStateFixture history(conv_state, 2);
    const bool source_aliases_snapshots =
        initial_slot >= snapshot_base_value[0] && initial_slot < snapshot_base_value[0] + tokens;
    const auto initial_offsets =
        source_aliases_snapshots
            ? std::vector<std::int64_t>{std::int64_t(initial_slot) * conv_state.nb[2]}
            : history.record_offsets(initial_value);
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor initial(device_initial.span(), DType::I64, {1});
    const auto source = source_aliases_snapshots ? dense_record_field(conv_state, 2, initial)
                                                 : history.view(initial);
    Tensor snapshot_base(device_snapshot_base.span(), DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        ops::PairedProjectionWeights{{query_key.view()}, {value_z_weight.view()}}, 1, tokens,
        tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(1, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    ops::gdn_input_proj_conv_snapshot(
        x, ops::PairedProjectionWeights{{query_key.view()}, {value_z_weight.view()}}, conv,
        {source, conv_state, snapshot_base}, Tensor{}, {q, k, v, z_output}, workspace,
        ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state,
        [&](std::int32_t row, std::int32_t token) {
            const float* token_activation =
                activation.data() + static_cast<std::size_t>(token) * kHidden;
            if (row < kQueryRows + kKeyRows) {
                return quantized_weight::dot_fp64(query_key.host, row, token_activation, kHidden);
            }
            return quantized_weight::dot_fp64(value_z_weight.host, row - kQueryRows - kKeyRows,
                                              token_activation, kHidden);
        },
        full_reference);
    const std::int32_t sample_count              = snapshot_sample_count(tokens, full_reference);
    const std::vector<std::uint16_t> state_after = state.bits();
    const std::string suffix                     = " Q4/Q5 A16 T=" + std::to_string(tokens) +
                               " initial=" + std::to_string(initial_slot) +
                               " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle,
                                           kGdnInputProjConvSnapshotA16Tolerance, full_reference);
    failures += compare(
        "snapshot state" + suffix,
        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot, full_reference),
        oracle.state, kGdnInputProjConvSnapshotA16Tolerance);
    failures += history.verify_guards("snapshot history");
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures += compare("snapshot z" + suffix,
                        gather_rows(z.values(), kZRows, 0, kZRows, tokens, sample_count),
                        projection_oracle(value_z_weight.host, kValueRows, kZRows, activation,
                                          kHidden, tokens, sample_count),
                        kGdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_offsets);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += query_key.verify_preserved("snapshot query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("snapshot value/z weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    // Record what was actually compared against the FP64 formula, so the run log distinguishes a
    // complete-output reference from a sampled one without inferring it from the case list.
    std::cout << "snapshot" << suffix << ": " << (full_reference ? "complete" : "sampled")
              << " FP64 reference, compared "
              << (oracle.query.size() + oracle.key.size() + oracle.value.size()) << " output values"
              << " + " << oracle.state.size() << " state values, workspace capacity "
              << workspace_bytes << " peak " << workspace.peak_used() << ", failures " << failures
              << '\n';
    return failures;
}

int run_q8_case(DevicePackedWeight& parent, std::int32_t tokens, std::int32_t initial_slot,
                bool graph = false, std::uint32_t route_seed = 0,
                ninfer::test::gdn_input_conv::Q8RouteCapture* route_capture = nullptr,
                const std::vector<std::uint16_t>* activation_override = nullptr,
                const std::vector<std::uint16_t>* initial_state_override = nullptr) {
    const auto weight                        = parent.view();
    const std::int32_t kHidden               = weight.k;
    const std::int32_t kValueRows            = weight.n == 16384 ? 6144 : 4096;
    const std::int32_t kZRows                = kValueRows;
    const std::int32_t kChannels             = weight.n - kZRows;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const auto activation_seed = route_seed ? route_seed : 701U + tokens;
    const auto state_seed = route_seed ? route_seed + 1 : 719U + tokens;
    std::vector<float> activation = make_bf16_activation(kHidden, tokens, activation_seed);
    std::vector<std::uint16_t> activation_bits        = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 709U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    if (activation_override) {
        if (activation_override->size() != activation_bits.size())
            throw std::invalid_argument("Q8 route activation override has incorrect size");
        activation_bits = *activation_override;
        for (std::size_t i = 0; i < activation.size(); ++i)
            activation[i] = bf16_to_f32(activation_bits[i]);
    }
    std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, state_seed);
    if (initial_state_override) {
        const std::size_t stride = static_cast<std::size_t>(kChannels) * 3;
        if (initial_state_override->size() != stride)
            throw std::invalid_argument("Q8 route state override has incorrect size");
        std::copy(initial_state_override->begin(), initial_state_override->end(),
                  state_before.begin() + static_cast<std::size_t>(initial_slot) * stride);
    }
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.span(), DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.span(), DType::BF16, {kChannels, 4});
    Tensor conv_state(state.span(), DType::BF16, {kChannels, 3, slots});
    RecordStateFixture history(conv_state, 2);
    const bool source_aliases_snapshots =
        initial_slot >= snapshot_base_value[0] && initial_slot < snapshot_base_value[0] + tokens;
    const auto initial_offsets =
        source_aliases_snapshots
            ? std::vector<std::int64_t>{std::int64_t(initial_slot) * conv_state.nb[2]}
            : history.record_offsets(initial_value);
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor initial(device_initial.span(), DType::I64, {1});
    const auto source = source_aliases_snapshots ? dense_record_field(conv_state, 2, initial)
                                                 : history.view(initial);
    Tensor snapshot_base(device_snapshot_base.span(), DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        ops::SingleProjectionWeight{parent.view(), ops::LinearPolicy::A16Only}, 1, tokens, tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(1, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    const auto launch = [&](CudaExecutionView cuda) {
        ops::gdn_input_proj_conv_snapshot(
            x, ops::SingleProjectionWeight{parent.view(), ops::LinearPolicy::A16Only}, conv,
            {source, conv_state, snapshot_base}, Tensor{}, {q, k, v, z_output}, workspace, cuda);
    };
    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        cuda_synchronize();
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto stream_execution =
            ninfer::test::cuda_environment().context.execution_view(stream);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch(stream_execution);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpyAsync(state.data(), state_before.data(),
                                   state_before.size() * sizeof(std::uint16_t),
                                   cudaMemcpyHostToDevice, stream));
        activation      = make_bf16_activation(kHidden, tokens, 1701U + tokens);
        activation_bits = bf16_bits(activation);
        CUDA_CHECK(cudaMemcpyAsync(device_activation.data(), activation_bits.data(),
                                   device_activation.size(), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else {
        launch(cuda_environment().default_execution);
        cuda_synchronize();
    }

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const std::vector<std::uint16_t> state_after = state.bits();
    if (route_capture) {
        const auto copy_first_column = [](std::vector<std::uint16_t>& output,
                                          const GuardedBf16Tensor& input, std::int32_t rows) {
            const auto bits = input.bits();
            output.assign(bits.begin(), bits.begin() + rows);
        };
        copy_first_column(route_capture->query, query, kQueryRows);
        copy_first_column(route_capture->key, key, kKeyRows);
        copy_first_column(route_capture->value, value, kValueRows);
        copy_first_column(route_capture->z, z, kZRows);
        const auto base = static_cast<std::size_t>(kSnapshotBaseSlot) * 3 * kChannels;
        route_capture->first_state.assign(state_after.begin() + base,
                                          state_after.begin() + base + 3 * kChannels);
        route_capture->sequence_query = query.bits();
        route_capture->sequence_key   = key.bits();
        route_capture->sequence_value = value.bits();
        route_capture->sequence_z     = z.bits();
        route_capture->activation     = activation_bits;
        const auto initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
        route_capture->initial_state.assign(state_before.begin() + initial_base,
                                            state_before.begin() + initial_base + 3 * kChannels);
        route_capture->state_after.clear();
        for (std::int32_t token = 0; token < tokens; ++token) {
            const auto token_base = static_cast<std::size_t>(kSnapshotBaseSlot + token) *
                                    3 * kChannels;
            route_capture->state_after.insert(route_capture->state_after.end(),
                                              state_after.begin() + token_base,
                                              state_after.begin() + token_base + 3 * kChannels);
        }
    }
    const std::string suffix =
        " Q8 K=" + std::to_string(kHidden) + " A16 T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle,
                                           kQ8GdnInputProjConvSnapshotA16Tolerance);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, kQ8GdnInputProjConvSnapshotA16Tolerance);
    failures += history.verify_guards("snapshot history");
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens),
                kQ8GdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_offsets);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                   std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kRows             = kChannels + kZRows;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 809U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 811U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 821U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.span(), DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.span(), DType::BF16, {kChannels, 4});
    Tensor conv_state(state.span(), DType::BF16, {kChannels, 3, slots});
    RecordStateFixture history(conv_state, 2);
    const bool source_aliases_snapshots =
        initial_slot >= snapshot_base_value[0] && initial_slot < snapshot_base_value[0] + tokens;
    const auto initial_offsets =
        source_aliases_snapshots
            ? std::vector<std::int64_t>{std::int64_t(initial_slot) * conv_state.nb[2]}
            : history.record_offsets(initial_value);
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor initial(device_initial.span(), DType::I64, {1});
    const auto source = source_aliases_snapshots ? dense_record_field(conv_state, 2, initial)
                                                 : history.view(initial);
    Tensor snapshot_base(device_snapshot_base.span(), DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        ops::SingleProjectionWeight{parent.view(), policy}, 1, tokens, tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(256, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    ops::gdn_input_proj_conv_snapshot(x, ops::SingleProjectionWeight{parent.view(), policy}, conv,
                                      {source, conv_state, snapshot_base}, Tensor{},
                                      {q, k, v, z_output}, workspace,
                                      ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const bool a4 = policy == ops::LinearPolicy::AllowA4;
    const ReductionCriterion& criterion =
        a4 ? kGdnInputProjConvSnapshotA4Tolerance : kGdnInputProjConvSnapshotA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(kSnapshotBaseSlot);
    const std::vector<std::uint16_t> state_after = state.bits();
    int failures =
        verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle, criterion);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, criterion);
    failures += history.verify_guards("snapshot history");
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix,
                gather_rows(z.values(), kZRows, 0, kZRows, tokens, snapshot_sample_count(tokens)),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens,
                                  snapshot_sample_count(tokens)),
                criterion);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_offsets);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                 std::int32_t initial_slot) {
    constexpr std::int32_t kHidden               = 5120;
    constexpr std::int32_t kValueRows            = 6144;
    constexpr std::int32_t kZRows                = 6144;
    constexpr std::int32_t kChannels             = 10240;
    constexpr std::int32_t kRows                 = kChannels + kZRows;
    constexpr std::int32_t kSeparateSnapshotBase = 1;
    const std::int32_t snapshot_base_slot        = kSeparateSnapshotBase;
    const std::int32_t slots = std::max(tokens + snapshot_base_slot + 1, initial_slot + 1);
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 907U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 911U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 919U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{snapshot_base_slot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.span(), DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.span(), DType::BF16, {kChannels, 4});
    Tensor conv_state(state.span(), DType::BF16, {kChannels, 3, slots});
    RecordStateFixture history(conv_state, 2);
    const bool source_aliases_snapshots =
        initial_slot >= snapshot_base_value[0] && initial_slot < snapshot_base_value[0] + tokens;
    const auto initial_offsets =
        source_aliases_snapshots
            ? std::vector<std::int64_t>{std::int64_t(initial_slot) * conv_state.nb[2]}
            : history.record_offsets(initial_value);
    DeviceBuffer device_initial = to_device(initial_offsets);
    Tensor initial(device_initial.span(), DType::I64, {1});
    const auto source    = source_aliases_snapshots ? dense_record_field(conv_state, 2, initial)
                                                    : history.view(initial);
    Tensor snapshot_base = Tensor(device_snapshot_base.span(), DType::I32, {1});
    Tensor q             = query.tensor();
    Tensor k             = key.tensor();
    Tensor v             = value.tensor();
    Tensor z_output      = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        ops::SingleProjectionWeight{parent.view(), policy}, 1, tokens, tokens);
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(),
                                           std::max<std::size_t>(256, workspace_bytes));
    MemoryArena workspace(workspace_storage.span());

    ops::gdn_input_proj_conv_snapshot(x, ops::SingleProjectionWeight{parent.view(), policy}, conv,
                                      {source, conv_state, snapshot_base}, Tensor{},
                                      {q, k, v, z_output}, workspace,
                                      ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const bool uses_a8                  = policy == ops::LinearPolicy::AllowA8;
    const ReductionCriterion& criterion = uses_a8 ? kFp8GdnInputProjConvSnapshotA8Tolerance
                                                  : kFp8GdnInputProjConvSnapshotA16Tolerance;
    const std::string suffix =
        std::string(" FP8 ") + (uses_a8 ? "A8" : "A16") + " T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(snapshot_base_slot);
    const std::vector<std::uint16_t> state_after = state.bits();
    int failures =
        verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle, criterion);
    failures +=
        compare("snapshot state" + suffix,
                gather_state(state_after, kChannels, kValueRows, tokens, snapshot_base_slot),
                oracle.state, criterion);
    failures += history.verify_guards("snapshot history");
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, snapshot_base_slot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix,
                gather_rows(z.values(), kZRows, 0, kZRows, tokens, snapshot_sample_count(tokens)),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens,
                                  snapshot_sample_count(tokens)),
                criterion);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_offsets);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

namespace ninfer::test::gdn_input_conv {
int run_snapshot(Weights& weights, const SnapshotCase& c) {
    const auto p      = weights.profile;
    const int initial = c.initial < 0 ? c.width + 1 : c.initial;
    if (!c.batch && p.qk_rows == 2048) {
        switch (p.format) {
        case QType::Q4_G64_FP16:
            return run_q4_q5_case(weights.first, *weights.second, c.width, initial,
                                  c.full_reference);
        case QType::Q8_G32_FP16:
            return run_q8_case(weights.first, c.width, initial, c.graph);
        case QType::NVFP4:
            return run_nvfp4_case(weights.first, c.width, c.policy, initial);
        case QType::FP8_E4M3FN_ROW_BF16:
            return run_fp8_case(weights.first, c.width, c.policy, initial);
        default:
            throw std::logic_error("unknown snapshot test format");
        }
    }
    const int channels    = 2 * p.qk_rows + p.value_rows();
    const auto conv       = make_conv_weight(channels, c.conv_seed);
    const auto projection = weights.projection(c.policy);
    const auto capacity   = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        projection, c.batch ? c.batch : 1, c.width, c.width);
    const auto criterion =
        p.format == QType::Q8_G32_FP16 ? kQ8GdnInputProjConvSnapshotA16Tolerance
        : p.format == QType::FP8_E4M3FN_ROW_BF16
            ? (c.policy == ops::LinearPolicy::A16Only ? kFp8GdnInputProjConvSnapshotA16Tolerance
                                                      : kFp8GdnInputProjConvSnapshotA8Tolerance)
        : c.policy == ops::LinearPolicy::AllowA4 ? kGdnInputProjConvSnapshotA4Tolerance
                                                 : kGdnInputProjConvSnapshotA16Tolerance;
    return run_batched_case(
        std::string(p.name) + "/w" + std::to_string(c.width) + "/b" + std::to_string(c.batch),
        p.hidden, p.value_rows(), p.value_rows(), c.width, c.batch ? c.batch : 1, c.valid, conv,
        capacity, criterion,
        [&](int row, int column, const std::vector<float>& x) {
            return weights.project(row, x.data() + std::size_t(column) * p.hidden);
        },
        [&](int row, int column, const std::vector<float>& x) {
            return weights.project(channels + row, x.data() + std::size_t(column) * p.hidden);
        },
        [&](const Tensor& x, const Tensor& conv, StateFieldView source, StateFieldView destination,
            Tensor& state, const Tensor& valid, const Tensor& bases, Tensor& q, Tensor& k,
            Tensor& v, Tensor& z, MemoryArena& arena, CudaExecutionView execution) {
            if (x.ne[1] == 1)
                ops::gdn_input_proj_conv_update(x, projection, conv, source, destination, valid,
                                                {q, k, v, z}, arena, execution);
            else
                ops::gdn_input_proj_conv_snapshot(x, projection, conv, {source, state, bases},
                                                  valid, {q, k, v, z}, arena, execution);
        },
        p.qk_rows, c.graph, c.full_reference);
}

int q8_first_column_invariance(Weights& weights) {
    if (weights.profile.format != QType::Q8_G32_FP16 || weights.profile.hidden != 2560 ||
        weights.profile.qk_rows != 2048)
        throw std::invalid_argument("Q8 first-column test requires the Flash-Next projection shape");
    Q8RouteCapture single;
    int failures = run_q8_case(weights.first, 1, 1, false, 0x51a7U, &single);
    const auto check = [&](const char* label, const std::vector<std::uint16_t>& a,
                           const std::vector<std::uint16_t>& b) {
        if (a == b) return 0;
        std::size_t index = 0;
        while (index < std::min(a.size(), b.size()) && a[index] == b[index]) ++index;
        std::cerr << label << ": first-column route mismatch at element " << index << '\n';
        return 1;
    };
    for (int width = 2; width <= 5; ++width) {
        Q8RouteCapture window;
        failures += run_q8_case(weights.first, width, 1, false, 0x51a7U, &window);
        const auto suffix = " T=" + std::to_string(width);
        failures += check(("Q8 GDN first-column query" + suffix).c_str(), single.query,
                          window.query);
        failures += check(("Q8 GDN first-column key" + suffix).c_str(), single.key, window.key);
        failures += check(("Q8 GDN first-column value" + suffix).c_str(), single.value,
                          window.value);
        failures += check(("Q8 GDN first-column z" + suffix).c_str(), single.z, window.z);
        failures += check(("Q8 GDN first-column convolution state" + suffix).c_str(),
                          single.first_state, window.first_state);
    }
    return failures;
}

int q8_sequence_single_step_equivalence(Weights& weights) {
    constexpr std::int32_t width = 5;
    constexpr std::int32_t hidden = 2560;
    constexpr std::int32_t qk_rows = 2048;
    constexpr std::int32_t value_rows = 6144;
    constexpr std::int32_t channels = 2 * qk_rows + value_rows;
    constexpr std::uint32_t seed = 0x61b9U;
    if (weights.profile.format != QType::Q8_G32_FP16 || weights.profile.hidden != hidden ||
        weights.profile.qk_rows != qk_rows)
        throw std::invalid_argument("Q8 sequence test requires the Flash-Next projection shape");

    Q8RouteCapture sequence;
    int failures = run_q8_case(weights.first, width, 0, false, seed, &sequence);
    const auto compare_column = [&](std::string_view label,
                                    const std::vector<std::uint16_t>& batched,
                                    const std::vector<std::uint16_t>& single,
                                    std::int32_t rows, std::int32_t token) {
        const auto begin = batched.begin() + static_cast<std::ptrdiff_t>(token) * rows;
        if (std::equal(begin, begin + rows, single.begin())) return 0;
        std::int32_t index = 0;
        while (index < rows && *(begin + index) == single[static_cast<std::size_t>(index)]) ++index;
        std::cerr << label << ": T=" << width << " column=" << token
                  << " differs from serial step at element " << index << '\n';
        return 1;
    };
    const std::size_t state_stride = static_cast<std::size_t>(channels) * 3;
    const std::size_t activation_stride = hidden;
    for (std::int32_t token = 0; token < width; ++token) {
        const auto& source = token == 0
                                 ? sequence.initial_state
                                 : std::vector<std::uint16_t>(
                                       sequence.state_after.begin() +
                                           static_cast<std::ptrdiff_t>(token - 1) * state_stride,
                                       sequence.state_after.begin() +
                                           static_cast<std::ptrdiff_t>(token) * state_stride);
        const std::vector<std::uint16_t> activation(
            sequence.activation.begin() + static_cast<std::ptrdiff_t>(token) * activation_stride,
            sequence.activation.begin() +
                static_cast<std::ptrdiff_t>(token + 1) * activation_stride);
        Q8RouteCapture serial;
        failures += run_q8_case(weights.first, 1, 0, false, seed, &serial, &activation, &source);
        failures += compare_column("Q8 GDN query", sequence.sequence_query, serial.query,
                                   qk_rows, token);
        failures += compare_column("Q8 GDN key", sequence.sequence_key, serial.key, qk_rows,
                                   token);
        failures += compare_column("Q8 GDN value", sequence.sequence_value, serial.value,
                                   value_rows, token);
        failures += compare_column("Q8 GDN Z", sequence.sequence_z, serial.z, value_rows, token);
        if (serial.state_after.size() != state_stride ||
            !std::equal(sequence.state_after.begin() +
                            static_cast<std::ptrdiff_t>(token) * state_stride,
                        sequence.state_after.begin() +
                            static_cast<std::ptrdiff_t>(token + 1) * state_stride,
                        serial.state_after.begin())) {
            std::cerr << "Q8 GDN convolution state: T=" << width << " column=" << token
                      << " differs from serial step\n";
            ++failures;
        }
    }
    return failures;
}

int q8_batched_decode_record_equivalence(Weights& weights, int batch, int width) {
    constexpr int hidden = 2560, qk_rows = 2048, value_rows = 6144;
    constexpr int channels = 2 * qk_rows + value_rows;
    const int slots = 2 * batch;
    constexpr std::uint32_t seed = 0x7d31U;
    if (weights.profile.format != QType::Q8_G32_FP16 || weights.profile.hidden != hidden ||
        weights.profile.qk_rows != qk_rows)
        throw std::invalid_argument("Q8 route comparison requires the Flash-Next projection shape");

    const auto projection = weights.projection(ops::LinearPolicy::A16Only);
    const auto verify_activation = bf16_bits(make_bf16_activation(hidden, batch * width, seed));
    std::vector<std::uint16_t> update_activation(static_cast<std::size_t>(hidden) * batch);
    for (int row = 0; row < batch; ++row)
        std::copy_n(verify_activation.begin() + static_cast<std::ptrdiff_t>(row * width * hidden),
                    hidden, update_activation.begin() + static_cast<std::ptrdiff_t>(row * hidden));
    const auto conv_weight = bf16_bits(make_conv_weight(channels, seed + 1));
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> initial(slot_stride * slots, 0);
    std::vector<int> source_slots(static_cast<std::size_t>(batch));
    std::vector<int> destination_slots(static_cast<std::size_t>(batch));
    for (int row = 0; row < batch; ++row) {
        source_slots[row] = 2 * row;
        destination_slots[row] = source_slots[row] + 1;
        const auto slot = make_state(channels, 1, 0, seed + 2 + row);
        std::copy_n(slot.begin(), slot_stride,
                    initial.begin() + static_cast<std::ptrdiff_t>(source_slots[row] * slot_stride));
    }

    DeviceBuffer update_x_device = to_device(update_activation);
    DeviceBuffer verify_x_device = to_device(verify_activation);
    DeviceBuffer conv_device = to_device(conv_weight);
    DeviceBuffer valid_update_device = to_device(std::vector<int>(batch, 1));
    DeviceBuffer valid_verify_device = to_device(std::vector<int>(batch, width));
    GuardedBf16Tensor state(channels * 3, slots);
    state.copy_from_bits(initial);
    Tensor state_tensor = state.tensor().view({channels, 3, slots});
    RecordStateFixture history(state_tensor, 2);
    const auto source_bytes = history.record_offsets(source_slots);
    const auto destination_bytes = history.record_offsets(destination_slots);
    DeviceBuffer source_device = to_device(source_bytes);
    DeviceBuffer destination_device = to_device(destination_bytes);
    Tensor source_offsets(source_device.span(), DType::I64, {batch});
    Tensor destination_offsets(destination_device.span(), DType::I64, {batch});
    const auto source = history.view(source_offsets);
    const auto destination = history.view(destination_offsets);
    Tensor valid_update(valid_update_device.span(), DType::I32, {batch});
    Tensor valid_verify(valid_verify_device.span(), DType::I32, {batch});
    Tensor x_update(update_x_device.span(), DType::BF16, {hidden, 1, batch});
    Tensor x_verify(verify_x_device.span(), DType::BF16, {hidden, width, batch});
    Tensor conv(conv_device.span(), DType::BF16, {channels, 4});

    GuardedBf16Tensor update_query(qk_rows, batch), update_key(qk_rows, batch);
    GuardedBf16Tensor update_value(value_rows, batch), update_z(value_rows, batch);
    GuardedBf16Tensor verify_query(qk_rows, width * batch), verify_key(qk_rows, width * batch);
    GuardedBf16Tensor verify_value(value_rows, width * batch), verify_z(value_rows, width * batch);
    GuardedBf16Tensor conv_record(channels, width * batch);
    Tensor conv_record_tensor = conv_record.tensor().view({channels, width, batch});
    const auto output = [batch](GuardedBf16Tensor& q, GuardedBf16Tensor& k,
                                GuardedBf16Tensor& v, GuardedBf16Tensor& z, int tokens) {
        return ops::GdnConvOutputs{q.tensor().view({q.tensor().ne[0], tokens, batch}),
                                   k.tensor().view({k.tensor().ne[0], tokens, batch}),
                                   v.tensor().view({v.tensor().ne[0], tokens, batch}),
                                   z.tensor().view({z.tensor().ne[0], tokens, batch})};
    };
    const auto exec = cuda_environment().default_execution;
    DeviceBuffer update_workspace_storage(
        cuda_environment().context.device(),
        std::max<std::size_t>(256, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                      projection, batch, 1, 1)));
    DeviceBuffer record_workspace_storage(
        cuda_environment().context.device(),
        std::max<std::size_t>(256, ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                                      projection, batch, width, width)));
    MemoryArena update_workspace(update_workspace_storage.span());
    MemoryArena record_workspace(record_workspace_storage.span());

    ops::gdn_input_proj_conv_update(x_update, projection, conv, source, destination,
                                    valid_update, output(update_query, update_key, update_value,
                                                        update_z, 1),
                                    update_workspace, exec);
    ops::gdn_input_proj_conv_record(x_verify, projection, conv, source, valid_verify,
                                   output(verify_query, verify_key, verify_value, verify_z, width),
                                   conv_record_tensor, record_workspace, exec);
    cuda_synchronize();

    int failures = 0;
    const auto compare_first = [&](const char* label, const GuardedBf16Tensor& ordinary,
                                   const GuardedBf16Tensor& verify, int rows) {
        const auto a = ordinary.bits(), b = verify.bits();
        for (int row = 0; row < batch; ++row) {
            const auto ai = a.begin() + static_cast<std::ptrdiff_t>(row * rows);
            const auto bi = b.begin() + static_cast<std::ptrdiff_t>(row * width * rows);
            if (!std::equal(ai, ai + rows, bi)) {
                ++failures;
                std::cerr << "Q8 identical-input route mismatch " << label << " row=" << row << '\n';
            }
        }
    };
    compare_first("query", update_query, verify_query, qk_rows);
    compare_first("key", update_key, verify_key, qk_rows);
    compare_first("value", update_value, verify_value, value_rows);
    compare_first("z", update_z, verify_z, value_rows);
    history.copy_back();
    const auto state_after = state.bits(), record_bits = conv_record.bits();
    for (int row = 0; row < batch; ++row) {
        const auto state_base = static_cast<std::size_t>(destination_slots[row]) * slot_stride +
                                2ULL * channels;
        const auto record_base = static_cast<std::size_t>(row * width) * channels;
        if (!std::equal(state_after.begin() + static_cast<std::ptrdiff_t>(state_base),
                        state_after.begin() + static_cast<std::ptrdiff_t>(state_base + channels),
                        record_bits.begin() + static_cast<std::ptrdiff_t>(record_base))) {
            ++failures;
            std::cerr << "Q8 identical-input route mismatch convolution state row=" << row << '\n';
        }
    }
    return failures;
}
} // namespace ninfer::test::gdn_input_conv
