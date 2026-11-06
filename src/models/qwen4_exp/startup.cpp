#include "models/qwen4_exp/startup.h"
#include "models/common/frontend/digest.h"

#include <nlohmann/json.hpp>

namespace ninfer::models::qwen4_exp {
Startup::Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
                 const EngineOptions& options, Device device)
    : reader_(rank_source), definition_(read_definition(definition_source, load_options(options))) {
    if (options.deployment) {
        const auto parallel = plan_parallel(*definition_, *options.deployment);
        if (parallel.ranks.size() != 1)
            throw std::invalid_argument("Qwen4Exp distributed generation is not implemented");
    }
    profile       = execution::describe_model(*definition_);
    auto resident = resolve_load(reader_, definition_, device);
    table_        = resident.ple_table_owner();
    placements_.push_back({WeightPlacement::Resident,
                           resident.materialization().device_capacity_bytes,
                           resident.materialization().host_capacity_bytes});
    offload_ = describe_moe_plan(*profile, options.moe);
    if (offload_) {
        const auto host = resolve_load(reader_, definition_, device, Device::cpu(), table_);
        placements_.push_back({WeightPlacement::HostExperts,
                               host.materialization().device_capacity_bytes,
                               host.materialization().host_capacity_bytes});
    }
    disk_offload_ = describe_moe_plan(*profile, options.moe, true);
    if (disk_offload_) {
        const auto disk = resolve_load(reader_, definition_, device, Device::cpu(), table_, true);
        placements_.push_back({WeightPlacement::DiskExperts,
                               disk.materialization().device_capacity_bytes,
                               disk.materialization().host_capacity_bytes});
    }
}

std::optional<Startup::ExecutionPlan>
Startup::plan(DeviceContext& device, const EngineOptions& options, WeightPlacement placement,
              std::size_t available, std::size_t available_host_bytes) const {
    const bool host = placement == WeightPlacement::HostExperts;
    const bool disk = placement == WeightPlacement::DiskExperts;
    const auto& schedule = host ? offload_ : disk ? disk_offload_ : offload_;
    if ((host || disk) && !schedule) return {};
    auto effective_options = options;
    if (disk) effective_options.use_cuda_graph = false;
    auto planned = plan_execution(profile, table_->geometry(), effective_options, device, available,
                                  host || disk ? schedule : nullptr, true,
                                  available_host_bytes);
    if (!planned) return {};
    const auto capacity = planned->capacity;
    const auto chunk    = planned->shape.prefill_chunk;
    const auto moe      = planned->moe;
    return ExecutionPlan{std::move(*planned), capacity, placement, chunk, {moe}};
}

std::unique_ptr<ModelShard> Startup::materialize(DeviceContext& device, WeightPlacement placement,
                                                 const StartupObserver& observer) {
    auto resolved = resolve_load(reader_, definition_, device.device(),
                                 placement == WeightPlacement::Resident ? device.device()
                                                                        : Device::cpu(),
                                 table_, placement == WeightPlacement::DiskExperts);
    return materialize_model(std::move(resolved), device, &observer);
}

qwen::Frontend Startup::frontend(const EngineOptions& o) const {
    return create_frontend(*definition_, o);
}

qwen::Frontend create_frontend(const ModelDefinition& definition, const EngineOptions& o) {
    const auto generation =
        nlohmann::json::parse(definition.resources().text.generation_config_json);
    SamplingPreset preset;
    preset.temperature = generation.value("temperature", 1.0F);
    preset.top_k       = generation.value("top_k", 20);
    preset.top_p       = generation.value("top_p", 0.95F);
    preset.min_p       = generation.value("min_p", 0.0F);
    return qwen::make_frontend(definition.resources(),
                               {.chat_template_path       = o.chat_template_path,
                                .grammar_cache_bytes      = o.grammar_cache_bytes,
                                .vision_enabled           = o.enable_vision,
                                .max_context              = o.max_context,
                                .media_cache_bytes        = o.media_cache_bytes,
                                .media_live_bytes         = o.media_live_bytes,
                                .media_preprocess_threads = o.media_preprocess_threads},
                               {preset, preset});
}

std::unique_ptr<RankProgram> create_rank_program(const execution::Parameters& p,
                                                 qwen4_exp::ExecutionPlan&& plan,
                                                 DeviceContext& device,
                                                 const StartupObserver& observer,
                                                 GraphPreparationMode mode) {
    auto program = std::make_unique<RankProgram>(p, std::move(plan), device, observer);
    program->prepare_execution(mode);
    return program;
}

GraphPreparationPlan Startup::graph_plan(const EngineOptions& options,
                                         const ExecutionPreparation& preparation) const {
    auto effective_options = options;
    if (preparation.experts && preparation.experts->description().cache_pinned_bytes)
        effective_options.use_cuda_graph = false;
    return describe_graph_preparation(effective_options, preparation.experts.get());
}

Startup::Calibration Startup::calibrate(const execution::Parameters& parameters,
                                        DeviceContext& device, const EngineOptions& options,
                                        const ExecutionPreparation& preparation) const {
    auto effective_options = options;
    if (preparation.experts && preparation.experts->description().cache_pinned_bytes)
        effective_options.use_cuda_graph = false;
    return calibrate_graphs(parameters, device, effective_options, preparation.experts);
}

std::string prefill_signature(const ModelDefinition& model, const moe::Plan* moe,
                              std::uint32_t chunk) {
    using Json    = nlohmann::json;
    const auto& c = model.config().text;
    Json facts{{"implementation", "qwen4-exp-eager-1"},
               {"chunk", chunk},
               {"hidden", c.hidden},
               {"vocabulary", c.vocabulary},
               {"hc", {c.hc.streams, c.hc.lowrank}},
               {"gdn", {c.gdn.key_heads, c.gdn.value_heads, c.gdn.key_dim, c.gdn.value_dim}},
               {"qsa",
                {c.qsa.query_heads, c.qsa.kv_heads, c.qsa.head_dim, c.qsa.compression,
                 c.qsa.block_budget}},
               {"ple", {c.ple.layer, c.ple.shards, c.ple.rows_per_shard}}};
    auto& layers = facts["layers"] = Json::array();
    if (const auto& vision = model.config().vision)
        facts["vision"] = {vision->depth,
                           vision->hidden_size,
                           vision->intermediate_size,
                           vision->num_heads,
                           vision->patch_size,
                           vision->temporal_patch_size,
                           vision->spatial_merge_size,
                           vision->num_position_embeddings};
    for (const auto& layer : c.layers) layers.push_back(static_cast<int>(layer.mixer));
    auto& weights = facts["weights"] = Json::array();
    for (const auto& w : model.weight_data()) {
        Json item{{"role", w.name},
                  {"shape", w.view.shape},
                  {"parts", Json::array()},
                  {"uses", Json::array()}};
        for (const auto& part : w.view.parts)
            item["parts"].push_back(
                {part.parent->geometry.format, part.parent->geometry.layout, part.begin, part.end});
        for (const auto& use : w.uses) item["uses"].push_back({use.input, use.policy});
        weights.push_back(std::move(item));
    }
    if (moe)
        facts["moe"] = {moe->description().cache_bytes, moe->description().fixed_bytes,
                        moe->description().window_count, moe->description().cpu_workers};
    return models::frontend::sha256_hex(models::frontend::sha256(facts.dump()));
}

std::string Startup::prefill_signature(const ExecutionPreparation& preparation,
                                       std::uint32_t chunk) const {
    return qwen4_exp::prefill_signature(*definition_, preparation.experts.get(), chunk);
}

} // namespace ninfer::models::qwen4_exp
