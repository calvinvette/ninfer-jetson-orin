#include "models/common/qwen/vision/vision.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
using namespace ninfer;
using namespace ninfer::test;
namespace qwen = models::qwen;
using Values = std::vector<double>;

// Qwen3.8-Flash-Next Vision geometry from its registered artifact config.
constexpr int kHidden = 1152;
constexpr int kHeadDim = 72;
constexpr int kHeads = 16;
constexpr int kIntermediate = 4304;
constexpr int kDepth = 27;
constexpr int kPositionRows = 2304;
constexpr int kFeatures = 2;
constexpr int kPatches = 4 * kFeatures;
constexpr int kOutput = 2560;

double bf16(double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); }

struct Vector {
    Values values;
    DeviceBuffer storage;
    Tensor tensor;

    Vector(int n, double base, double amplitude, int seed)
        : values(static_cast<std::size_t>(n)), storage(), tensor() {
        std::vector<float> host(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const auto value = base + amplitude * ((i * 7 + seed * 13) % 29 - 14);
            host[static_cast<std::size_t>(i)] = static_cast<float>(bf16(value));
            values[static_cast<std::size_t>(i)] = host[static_cast<std::size_t>(i)];
        }
        storage = to_device_bf16(host);
        tensor = Tensor(storage.span(), DType::BF16, {n});
    }
};

struct Matrix {
    int n, k, seed;
    std::vector<std::uint16_t> bits;
    DeviceBuffer storage;
    ops::SingleProjectionWeight parameter;

    Matrix(int rows, int columns, int salt)
        : n(rows), k(columns), seed(salt), bits(static_cast<std::size_t>(rows) * columns), storage() {
        for (int r = 0; r < n; ++r)
            for (int j = 0; j < 2; ++j)
                bits[static_cast<std::size_t>(r) * k + index(r, j)] =
                    f32_to_bf16(static_cast<float>(weight(r, j)));
        storage = to_device(bits);
        Weight w;
        w.device = storage.span().device;
        w.payload = w.qdata = storage.data();
        w.payload_bytes = storage.size();
        w.qtype = QType::BF16;
        w.layout = QuantLayout::Contiguous;
        w.n = n;
        w.k = k;
        w.ndim = 2;
        w.shape[0] = w.padded_shape[0] = n;
        w.shape[1] = w.padded_shape[1] = k;
        parameter = {w, ops::LinearPolicy::A16Only};
    }

    int index(int row, int item) const { return (row * 13 + seed * 19 + item * 17) % k; }
    double weight(int row, int item) const {
        return bf16((item ? -0.25 : 0.5) + (row % 5) / 64.0);
    }
    Values apply(const Values& x, int tokens) const {
        Values y(static_cast<std::size_t>(n) * tokens);
        for (int token = 0; token < tokens; ++token)
            for (int row = 0; row < n; ++row) {
                double sum = 0;
                for (int column = 0; column < k; ++column) {
                    const auto w = bf16_to_f32(bits[static_cast<std::size_t>(row) * k + column]);
                    sum += w * x[static_cast<std::size_t>(token) * k + column];
                }
                y[static_cast<std::size_t>(token) * n + row] = bf16(sum);
            }
        return y;
    }
};

void add_bias(Values& x, int width, int tokens, const Values& bias) {
    for (int token = 0; token < tokens; ++token)
        for (int d = 0; d < width; ++d) {
            const auto at = static_cast<std::size_t>(token) * width + d;
            x[at] = bf16(x[at] + bias[static_cast<std::size_t>(d)]);
        }
}

Values norm(const Values& x, const Vector& gamma, const Vector& beta, int width, int tokens) {
    Values y(x.size());
    for (int token = 0; token < tokens; ++token) {
        const auto begin = static_cast<std::size_t>(token) * width;
        double mean = 0, variance = 0;
        for (int d = 0; d < width; ++d) mean += x[begin + d];
        mean /= width;
        for (int d = 0; d < width; ++d) variance += std::pow(x[begin + d] - mean, 2);
        const auto inv = 1.0 / std::sqrt(variance / width + 1.0e-6);
        for (int d = 0; d < width; ++d)
            y[begin + d] = bf16((x[begin + d] - mean) * inv * gamma.values[d] + beta.values[d]);
    }
    return y;
}

Values residual(const Values& x, const Values& y) {
    Values out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = bf16(x[i] + y[i]);
    return out;
}

void gelu(Values& x, bool exact) {
    constexpr double c = 0.79788456080286535587989211986876;
    for (auto& v : x) {
        const auto input = v;
        const auto output =
            exact ? 0.5 * input * (1.0 + std::erf(input / std::sqrt(2.0)))
                  : 0.5 * input * (1.0 + std::tanh(
                                               c * (input + 0.044715 * input * input * input)));
        v = bf16(output);
    }
}

Values reference(const qwen::VisionConfig& config, const Matrix& patch,
                 const Vector& patch_bias, const std::vector<double>& positions,
                 const std::vector<std::int32_t>& position_indices,
                 const std::vector<float>& position_weights,
                 const std::vector<double>& position_table, const Matrix& qkv,
                 const Vector& qkv_bias, const Vector& norm1_gamma, const Vector& norm1_beta,
                 const Matrix& attention_out, const Vector& attention_bias,
                 const Vector& norm2_gamma, const Vector& norm2_beta, const Matrix& fc1,
                 const Vector& fc1_bias, const Matrix& fc2, const Vector& fc2_bias,
                 const Vector& merger_gamma, const Vector& merger_beta,
                 const Matrix& merger_fc1, const Vector& merger_fc1_bias,
                 const Matrix& merger_fc2, const Vector& merger_fc2_bias,
                 const Values& patch_input, const std::vector<std::int32_t>& cu_seqlens) {
    Values x = patch.apply(patch_input, kPatches);
    add_bias(x, kHidden, kPatches, patch_bias.values);
    for (int token = 0; token < kPatches; ++token)
        for (int d = 0; d < kHidden; ++d) {
            double embedding = 0;
            for (int corner = 0; corner < 4; ++corner) {
                const auto id = position_indices[static_cast<std::size_t>(token) * 4 + corner];
                embedding += position_table[static_cast<std::size_t>(id) * kHidden + d] *
                             position_weights[static_cast<std::size_t>(token) * 4 + corner];
            }
            const auto at = static_cast<std::size_t>(token) * kHidden + d;
            x[at] = bf16(x[at] + embedding);
        }

    for (int layer = 0; layer < config.depth; ++layer) {
        auto h = norm(x, norm1_gamma, norm1_beta, kHidden, kPatches);
        auto packed = qkv.apply(h, kPatches);
        add_bias(packed, 3 * kHidden, kPatches, qkv_bias.values);
        for (int token = 0; token < kPatches; ++token)
            for (int head = 0; head < kHeads; ++head)
                for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                    const int axis = pair / (kHeadDim / 4);
                    const int lane = pair % (kHeadDim / 4);
                    const double frequency = std::pow(10000.0, -2.0 * lane / (kHeadDim / 2));
                    const double phase = positions[static_cast<std::size_t>(axis) * kPatches + token] *
                                         frequency;
                    const double cosine = std::cos(phase), sine = std::sin(phase);
                    for (int component = 0; component < 2; ++component) {
                        const auto base = static_cast<std::size_t>(token) * 3 * kHidden +
                                          component * kHidden + head * kHeadDim;
                        const auto lo = base + pair, hi = base + pair + kHeadDim / 2;
                        const auto a = packed[lo], b = packed[hi];
                        packed[lo] = bf16(a * cosine - b * sine);
                        packed[hi] = bf16(b * cosine + a * sine);
                    }
                }
        Values attended(static_cast<std::size_t>(kPatches) * kHidden);
        for (std::size_t segment = 1; segment < cu_seqlens.size(); ++segment) {
            const int begin = cu_seqlens[segment - 1], end = cu_seqlens[segment];
            for (int query = begin; query < end; ++query)
                for (int head = 0; head < kHeads; ++head) {
                    std::vector<double> scores(static_cast<std::size_t>(end - begin));
                    double maximum = -INFINITY;
                    for (int key = begin; key < end; ++key) {
                        double score = 0;
                        for (int d = 0; d < kHeadDim; ++d) {
                            const auto qi = static_cast<std::size_t>(query) * 3 * kHidden +
                                            head * kHeadDim + d;
                            const auto ki = static_cast<std::size_t>(key) * 3 * kHidden +
                                            kHidden + head * kHeadDim + d;
                            score += packed[qi] * packed[ki] / std::sqrt(kHeadDim);
                        }
                        scores[static_cast<std::size_t>(key - begin)] = score;
                        maximum = std::max(maximum, score);
                    }
                    double denominator = 0;
                    for (auto& score : scores) {
                        score = std::exp(score - maximum);
                        denominator += score;
                    }
                    for (int d = 0; d < kHeadDim; ++d) {
                        double value = 0;
                        for (int key = begin; key < end; ++key) {
                            const auto vi = static_cast<std::size_t>(key) * 3 * kHidden +
                                            2 * kHidden + head * kHeadDim + d;
                            value += scores[static_cast<std::size_t>(key - begin)] / denominator *
                                     packed[vi];
                        }
                        attended[static_cast<std::size_t>(query) * kHidden + head * kHeadDim + d] =
                            bf16(value);
                    }
                }
        }
        auto projected = attention_out.apply(attended, kPatches);
        add_bias(projected, kHidden, kPatches, attention_bias.values);
        x = residual(x, projected);

        h = norm(x, norm2_gamma, norm2_beta, kHidden, kPatches);
        auto up = fc1.apply(h, kPatches);
        add_bias(up, kIntermediate, kPatches, fc1_bias.values);
        gelu(up, false);
        auto down = fc2.apply(up, kPatches);
        add_bias(down, kHidden, kPatches, fc2_bias.values);
        x = residual(x, down);
    }

    auto normalized = norm(x, merger_gamma, merger_beta, kHidden, kPatches);
    auto merged = merger_fc1.apply(normalized, kFeatures);
    add_bias(merged, static_cast<int>(config.merger_width()), kFeatures, merger_fc1_bias.values);
    gelu(merged, true);
    auto output = merger_fc2.apply(merged, kFeatures);
    add_bias(output, kOutput, kFeatures, merger_fc2_bias.values);
    return output;
}

int run() {
    auto& env = cuda_environment();
    const qwen::VisionConfig config{.depth = kDepth,
                                    .hidden_size = kHidden,
                                    .intermediate_size = kIntermediate,
                                    .num_heads = kHeads,
                                    .patch_size = 16,
                                    .temporal_patch_size = 2,
                                    .spatial_merge_size = 2,
                                    .num_position_embeddings = kPositionRows,
                                    .position_grid_side = 48};
    Matrix patch(kHidden, static_cast<int>(config.patch_width()), 1);
    Vector patch_bias(kHidden, 0, .002, 1), norm1_gamma(kHidden, 1, .001, 2),
        norm1_beta(kHidden, 0, .001, 3), qkv_bias(3 * kHidden, 0, .002, 4),
        attention_bias(kHidden, 0, .002, 5), norm2_gamma(kHidden, 1, .001, 6),
        norm2_beta(kHidden, 0, .001, 7), fc1_bias(kIntermediate, 0, .001, 8),
        fc2_bias(kHidden, 0, .002, 9), merger_gamma(kHidden, 1, .001, 10),
        merger_beta(kHidden, 0, .001, 11),
        merger_fc1_bias(static_cast<int>(config.merger_width()), 0, .001, 12),
        merger_fc2_bias(kOutput, 0, .002, 13);
    Matrix qkv(3 * kHidden, kHidden, 14), attention_out(kHidden, kHidden, 15),
        fc1(kIntermediate, kHidden, 16), fc2(kHidden, kIntermediate, 17),
        merger_fc1(static_cast<int>(config.merger_width()),
                   static_cast<int>(config.merger_width()), 18),
        merger_fc2(kOutput, static_cast<int>(config.merger_width()), 19);
    std::vector<std::uint16_t> table_bits(static_cast<std::size_t>(kPositionRows) * kHidden);
    std::vector<double> table(table_bits.size());
    for (std::size_t i = 0; i < table.size(); ++i) {
        table[i] = bf16(std::sin(i * .003) * .05);
        table_bits[i] = f32_to_bf16(static_cast<float>(table[i]));
    }
    auto table_storage = to_device(table_bits);
    Tensor position_table(table_storage.span(), DType::BF16, {kHidden, kPositionRows});
    qwen::VisionBlockParameters block{
        {norm1_gamma.tensor, norm1_beta.tensor},
        {norm2_gamma.tensor, norm2_beta.tensor},
        qkv.parameter,
        qkv_bias.tensor,
        attention_out.parameter,
        fc1.parameter,
        fc2.parameter,
        attention_bias.tensor,
        fc1_bias.tensor,
        fc2_bias.tensor};
    // Reuse one deterministic block to exercise the complete 27-layer composition at low cost.
    const std::vector<qwen::VisionBlockParameters> blocks(kDepth, block);
    qwen::VisionParameters parameters{patch.parameter,
                                      patch_bias.tensor,
                                      position_table,
                                      blocks,
                                      {merger_gamma.tensor, merger_beta.tensor},
                                      merger_fc1.parameter,
                                      merger_fc2.parameter,
                                      merger_fc1_bias.tensor,
                                      merger_fc2_bias.tensor};
    const qwen::VisionBlockProfile layer_profile{ops::linear_profile(block.qkv),
                                                  ops::linear_profile(block.output),
                                                  ops::linear_profile(block.fc1),
                                                  ops::linear_profile(block.fc2)};
    qwen::VisionProfile profile{ops::linear_profile(parameters.patch_embedding),
                                ops::linear_profile(parameters.merger_fc1),
                                ops::linear_profile(parameters.merger_fc2),
                                std::vector<qwen::VisionBlockProfile>(kDepth, layer_profile)};

    const auto patch_count = static_cast<std::size_t>(kPatches) * config.patch_width();
    std::vector<std::uint16_t> patch_bits(patch_count);
    Values patch_input(patch_count);
    for (std::size_t i = 0; i < patch_count; ++i) {
        patch_bits[i] = f32_to_bf16(static_cast<float>(std::sin(i * .031) * .1));
        patch_input[i] = bf16_to_f32(patch_bits[i]);
    }
    const auto positions = std::vector<double>{0, 0, 1, 1, 0, 1, 0, 1,
                                                2, 2, 3, 3, 2, 3, 2, 3};
    std::vector<std::int32_t> position_indices(static_cast<std::size_t>(kPatches) * 4);
    std::vector<float> position_weights(position_indices.size(), .25F);
    for (int token = 0; token < kPatches; ++token)
        for (int corner = 0; corner < 4; ++corner)
            position_indices[static_cast<std::size_t>(token) * 4 + corner] =
                (token + corner) % kPositionRows;
    std::vector<std::span<const std::uint16_t>> patches{
        std::span(patch_bits).first(static_cast<std::size_t>(4) * config.patch_width()),
        std::span(patch_bits).subspan(static_cast<std::size_t>(4) * config.patch_width())};
    qwen::VisionEncoderBatch batch{patches,
                                   [&] {
                                       std::vector<std::int32_t> v(kPatches * 2);
                                       for (int axis = 0; axis < 2; ++axis)
                                           for (int t = 0; t < kPatches; ++t)
                                               v[static_cast<std::size_t>(axis) * kPatches + t] =
                                                   static_cast<std::int32_t>(
                                                       positions[static_cast<std::size_t>(axis) *
                                                                 kPatches + t]);
                                       return v;
                                   }(),
                                   position_indices, {0, 4, 8}, position_weights,
                                   static_cast<std::uint32_t>(kFeatures)};

    const auto expected = reference(config, patch, patch_bias, positions, position_indices,
                                    position_weights, table, qkv, qkv_bias, norm1_gamma,
                                    norm1_beta, attention_out, attention_bias, norm2_gamma,
                                    norm2_beta, fc1, fc1_bias, fc2, fc2_bias, merger_gamma,
                                    merger_beta, merger_fc1, merger_fc1_bias, merger_fc2,
                                    merger_fc2_bias, patch_input, batch.cu_seqlens);
    const auto capacity = qwen::VisionContext::plan_resources(config, profile, kFeatures,
                                                               256ULL << 20);
    models::vision::Storage storage(env.context, capacity);
    DeviceBuffer workspace(env.context.device(), capacity.encoder.workspace_bytes);
    auto output = storage.output(0, kFeatures);
    qwen::VisionContext encoder(env.default_execution, config, parameters);
    (void)encoder.encode(batch, output, workspace.span(), storage.staging());
    CUDA_CHECK(cudaStreamSynchronize(env.default_execution.stream()));
    const auto actual = from_device_bf16(output.data, expected.size());
    double error = 0, energy = 0, maximum = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        error += std::pow(actual[i] - expected[i], 2);
        energy += std::pow(expected[i], 2);
        maximum = std::max(maximum, std::abs(actual[i] - expected[i]));
    }
    const auto relative = std::sqrt(error / std::max(energy, 1.0e-30));
    std::cout << "Qwen Vision composed FP64 reference: rel_l2=" << relative
              << " max_abs=" << maximum << '\n';
    return relative <= .015 && maximum <= .04 ? 0 : 1;
}
} // namespace

int main() {
    if (cuda_unavailable()) return 77;
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
