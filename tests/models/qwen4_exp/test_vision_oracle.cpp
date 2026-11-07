#include "models/common/qwen/vision/vision.h"
#include "models/qwen4_exp/execution/parameters.h"
#include "models/qwen4_exp/load.h"
#include "ops/op_tester.h"
#include "cuda_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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

    explicit Vector(Values stored_values) : values(std::move(stored_values)) {}
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

    Matrix(int rows, int columns, std::vector<std::uint16_t> stored_bits)
        : n(rows), k(columns), seed(0), bits(std::move(stored_bits)), storage() {
        if (bits.size() != static_cast<std::size_t>(n) * k)
            throw std::invalid_argument("Vision oracle matrix extent differs from its BF16 data");
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

struct LayerOracle {
    const Matrix* qkv;
    const Vector* qkv_bias;
    const Vector* norm1_gamma;
    const Vector* norm1_beta;
    const Matrix* attention_out;
    const Vector* attention_bias;
    const Vector* norm2_gamma;
    const Vector* norm2_beta;
    const Matrix* fc1;
    const Vector* fc1_bias;
    const Matrix* fc2;
    const Vector* fc2_bias;
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
                 const std::vector<double>& position_table,
                 const std::vector<LayerOracle>& layers,
                 const Vector& merger_gamma, const Vector& merger_beta,
                 const Matrix& merger_fc1, const Vector& merger_fc1_bias,
                 const Matrix& merger_fc2, const Vector& merger_fc2_bias,
                 const Values& patch_input, const std::vector<std::int32_t>& cu_seqlens) {
    const int patch_count = cu_seqlens.back();
    const int feature_count = static_cast<int>(cu_seqlens.size() - 1);
    Values x = patch.apply(patch_input, patch_count);
    add_bias(x, kHidden, patch_count, patch_bias.values);
    for (int token = 0; token < patch_count; ++token)
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

    for (const auto& layer : layers) {
        auto h = norm(x, *layer.norm1_gamma, *layer.norm1_beta, kHidden, patch_count);
        auto packed = layer.qkv->apply(h, patch_count);
        add_bias(packed, 3 * kHidden, patch_count, layer.qkv_bias->values);
        for (int token = 0; token < patch_count; ++token)
            for (int head = 0; head < kHeads; ++head)
                for (int pair = 0; pair < kHeadDim / 2; ++pair) {
                    const int axis = pair / (kHeadDim / 4);
                    const int lane = pair % (kHeadDim / 4);
                    const double frequency = std::pow(10000.0, -2.0 * lane / (kHeadDim / 2));
                    const double phase = positions[static_cast<std::size_t>(axis) * patch_count + token] *
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
        Values attended(static_cast<std::size_t>(patch_count) * kHidden);
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
        auto projected = layer.attention_out->apply(attended, patch_count);
        add_bias(projected, kHidden, patch_count, layer.attention_bias->values);
        x = residual(x, projected);

        h = norm(x, *layer.norm2_gamma, *layer.norm2_beta, kHidden, patch_count);
        auto up = layer.fc1->apply(h, patch_count);
        add_bias(up, kIntermediate, patch_count, layer.fc1_bias->values);
        gelu(up, false);
        auto down = layer.fc2->apply(up, patch_count);
        add_bias(down, kHidden, patch_count, layer.fc2_bias->values);
        x = residual(x, down);
    }

    auto normalized = norm(x, merger_gamma, merger_beta, kHidden, patch_count);
    auto merged = merger_fc1.apply(normalized, feature_count);
    add_bias(merged, static_cast<int>(config.merger_width()), feature_count,
             merger_fc1_bias.values);
    gelu(merged, true);
    auto output = merger_fc2.apply(merged, feature_count);
    add_bias(output, merger_fc2.n, feature_count, merger_fc2_bias.values);
    return output;
}

std::vector<std::uint16_t> read_bf16(const Tensor& tensor, std::size_t elements) {
    if (tensor.dtype != DType::BF16 || !tensor.data || !tensor.is_contiguous() ||
        tensor.numel() != static_cast<std::int64_t>(elements))
        throw std::invalid_argument("Vision oracle expected a contiguous BF16 tensor");
    std::vector<std::uint16_t> result(elements);
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return result;
}

Matrix read_matrix(const ops::SingleProjectionWeight& projection) {
    const auto& weight = projection.weight;
    if (weight.qtype != QType::BF16 || weight.n <= 0 || weight.k <= 0)
        throw std::invalid_argument("artifact Vision oracle requires BF16 matrix weights");
    const auto* data = weight.qdata ? weight.qdata : weight.payload;
    if (!data) throw std::invalid_argument("artifact Vision matrix has no weight data");
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(weight.n) * weight.k);
    CUDA_CHECK(cudaMemcpy(bits.data(), data, bits.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));
    return Matrix(weight.n, weight.k, std::move(bits));
}

Vector read_vector(const Tensor& tensor) {
    const auto bits = read_bf16(tensor, static_cast<std::size_t>(tensor.numel()));
    Values values(bits.size());
    std::transform(bits.begin(), bits.end(), values.begin(),
                   [](std::uint16_t value) { return bf16_to_f32(value); });
    return Vector(std::move(values));
}

int run_artifact(const std::string& artifact_path) {
    auto& env = cuda_environment();
    EngineOptions options;
    options.use_cuda_graph = false;
    options.enable_vision = true;
    options.max_context = 128;
    options.prefill_chunk = 128;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(512ULL << 20);
    options.context_cache.host_capacity_bytes = 0;

    artifact::Reader reader(artifact_path);
    auto definition = models::qwen4_exp::read_definition(reader, models::load_options(options));
    auto profile = models::qwen4_exp::execution::describe_model(*definition);
    auto load = models::qwen4_exp::resolve_load(reader, definition, env.context.device(),
                                               Device::cpu(), {}, true);
    auto model = models::qwen4_exp::materialize_model(std::move(load), env.context);
    models::qwen4_exp::execution::Parameters loaded(*model, profile);
    if (!profile->config.vision || !profile->vision || !loaded.vision ||
        profile->vision->layers.empty() || loaded.vision->layers.empty())
        throw std::runtime_error("artifact does not bind Qwen Vision layer parameters");

    const auto config = *profile->config.vision;
    const auto& vision_parameters = *loaded.vision;
    const auto& vision_profile = *profile->vision;

    const auto patch = read_matrix(vision_parameters.patch_embedding);
    const auto patch_bias = read_vector(vision_parameters.patch_embedding_bias);
    const auto position_bits = read_bf16(
        vision_parameters.position_embedding,
        static_cast<std::size_t>(config.hidden_size) * config.num_position_embeddings);
    Values position_table(position_bits.size());
    std::transform(position_bits.begin(), position_bits.end(), position_table.begin(),
                   [](std::uint16_t value) { return bf16_to_f32(value); });
    struct LayerData {
        Matrix qkv;
        Vector qkv_bias, norm1_gamma, norm1_beta;
        Matrix attention_out;
        Vector attention_bias, norm2_gamma, norm2_beta;
        Matrix fc1;
        Vector fc1_bias;
        Matrix fc2;
        Vector fc2_bias;
    };
    std::vector<LayerData> layer_data;
    layer_data.reserve(vision_parameters.layers.size());
    for (const auto& layer : vision_parameters.layers)
        layer_data.push_back({read_matrix(layer.qkv), read_vector(layer.qkv_bias),
                              read_vector(layer.norm1.weight), read_vector(layer.norm1.bias),
                              read_matrix(layer.output), read_vector(layer.output_bias),
                              read_vector(layer.norm2.weight), read_vector(layer.norm2.bias),
                              read_matrix(layer.fc1), read_vector(layer.fc1_bias),
                              read_matrix(layer.fc2), read_vector(layer.fc2_bias)});
    std::vector<LayerOracle> layer_oracle;
    layer_oracle.reserve(layer_data.size());
    for (const auto& layer : layer_data)
        layer_oracle.push_back({&layer.qkv, &layer.qkv_bias, &layer.norm1_gamma,
                                &layer.norm1_beta, &layer.attention_out, &layer.attention_bias,
                                &layer.norm2_gamma, &layer.norm2_beta, &layer.fc1, &layer.fc1_bias,
                                &layer.fc2, &layer.fc2_bias});
    const auto merger_gamma = read_vector(vision_parameters.merger_norm.weight);
    const auto merger_beta = read_vector(vision_parameters.merger_norm.bias);
    const auto merger_fc1 = read_matrix(vision_parameters.merger_fc1);
    const auto merger_fc1_bias = read_vector(vision_parameters.merger_fc1_bias);
    const auto merger_fc2 = read_matrix(vision_parameters.merger_fc2);
    const auto merger_fc2_bias = read_vector(vision_parameters.merger_fc2_bias);

    constexpr int features = 1;
    const int patches_count = 4 * features;
    const auto patch_count = static_cast<std::size_t>(patches_count) * config.patch_width();
    std::vector<std::uint16_t> patch_bits(patch_count);
    Values patch_input(patch_count);
    for (std::size_t i = 0; i < patch_count; ++i) {
        patch_bits[i] = f32_to_bf16(static_cast<float>(std::sin(i * .031) * .1));
        patch_input[i] = bf16_to_f32(patch_bits[i]);
    }
    const std::vector<std::int32_t> positions{0, 0, 1, 1, 0, 1, 0, 1};
    const std::vector<double> positions64(positions.begin(), positions.end());
    std::vector<std::int32_t> position_indices(16);
    std::vector<float> position_weights(16, .25F);
    for (int patch_index = 0; patch_index < patches_count; ++patch_index)
        for (int corner = 0; corner < 4; ++corner)
            position_indices[static_cast<std::size_t>(patch_index) * 4 + corner] =
                (patch_index + corner) % static_cast<int>(config.num_position_embeddings);
    qwen::VisionEncoderBatch batch{{std::span<const std::uint16_t>(patch_bits)}, positions,
                                   position_indices, {0, 4}, position_weights, features};
    const auto expected = reference(config, patch, patch_bias, positions64, position_indices,
                                    position_weights, position_table, layer_oracle, merger_gamma,
                                    merger_beta, merger_fc1, merger_fc1_bias, merger_fc2,
                                    merger_fc2_bias, patch_input, batch.cu_seqlens);
    const auto capacity = qwen::VisionContext::plan_resources(config, vision_profile, features,
                                                               256ULL << 20);
    models::vision::Storage storage(env.context, capacity);
    DeviceBuffer workspace(env.context.device(), capacity.encoder.workspace_bytes);
    auto output = storage.output(0, features);
    qwen::VisionContext encoder(env.default_execution, config, vision_parameters);
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
    std::cout << "Artifact 27-layer Qwen Vision FP64 reference: rel_l2=" << relative
              << " max_abs=" << maximum << '\n';
    return relative <= .04 && maximum <= .01 ? 0 : 1;
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

    const LayerOracle synthetic_layer{&qkv, &qkv_bias, &norm1_gamma, &norm1_beta,
                                      &attention_out, &attention_bias, &norm2_gamma,
                                      &norm2_beta, &fc1, &fc1_bias, &fc2, &fc2_bias};
    const std::vector<LayerOracle> synthetic_layers(kDepth, synthetic_layer);
    const auto expected = reference(config, patch, patch_bias, positions, position_indices,
                                    position_weights, table, synthetic_layers, merger_gamma,
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

int main(int argc, char** argv) {
    if (cuda_unavailable()) return 77;
    try {
        if (run() != 0) return 1;
        if (argc == 1) return 0;
        if (argc != 2 || std::string_view(argv[1]) != "--artifact")
            throw std::invalid_argument("expected no argument or --artifact");
        const char* artifact = std::getenv("NINFER_VISION_ORACLE_ARTIFACT");
        if (!artifact || !*artifact) {
            std::cout << "skip: NINFER_VISION_ORACLE_ARTIFACT is not set\n";
            return 77;
        }
        return run_artifact(artifact);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
