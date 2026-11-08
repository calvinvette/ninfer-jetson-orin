#include "models/common/moe/file_schedule.h"

#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/platform.h"
#include "core/transfer.h"
#include "models/common/moe/execution.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <unordered_set>

namespace ninfer::models::moe {
namespace {

constexpr auto kUploadBytes = std::size_t{96} << 20;
constexpr std::uint32_t kHotExpertsPerLayer = 16;
constexpr std::uint32_t kDynamicSlotsPerLayer = 16;

std::vector<ExpertKey> routed_keys(ExpertBankId bank, std::span<const std::int32_t> ids,
                                   std::int32_t top_k, std::int32_t tokens) {
    std::vector<ExpertKey> keys;
    keys.reserve(ids.size());
    std::unordered_set<std::uint32_t> seen;
    seen.reserve(ids.size());
    for (std::int32_t token = 0; token < tokens; ++token)
        for (std::int32_t rank = 0; rank < top_k; ++rank) {
            const auto id = ids[std::size_t(rank) * tokens + token];
            if (id < 0) throw std::runtime_error("GPU router returned a negative expert ID");
            const auto expert = static_cast<std::uint32_t>(id);
            if (seen.insert(expert).second) keys.push_back({bank, expert});
        }
    return keys;
}

std::vector<ExpertPlacement> placements_for(std::span<const ExpertKey> keys,
                                            std::size_t entry_bytes) {
    std::vector<ExpertPlacement> placements;
    placements.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i)
        placements.push_back({keys[i], i * entry_bytes});
    return placements;
}

class FilePlan;
class FileLayerDefinition;
class FileLayerExecutable;

class FileSchedule final : public Schedule {
public:
    FileSchedule(const FilePlan&, std::span<const Parameters* const>,
                 std::span<const ExpertBankSource>, const DeviceContext&);
    ~FileSchedule() override;

    bool try_before_layer(ExpertBankId) override { return true; }
    bool start(ExpertBankId, const Tensor&, Tensor&, MemoryArena&,
               const ops::MoeExecutionHints&) override;
    bool poll() override;
    void cancel() override;
    std::shared_ptr<const LayerGraphDefinition>
    capture_layer(ExpertBankId, CudaExecutionView, std::int32_t, const std::function<void()>&,
                  const GpuBody&, const std::function<void()>&,
                  const std::shared_ptr<const LayerGraphDefinition>&) override;
    void restore_cache_after_preparation() override;
    void synchronize() const override;
    std::uint64_t uploaded_bytes() const noexcept override { return cache_.uploaded_bytes(); }
    ExpertCacheUsage cache_usage() const override { return cache_.usage(); }
    const ScheduleDescription& description() const noexcept override { return description_; }

private:
    friend class FileLayerDefinition;
    friend class FileLayerExecutable;

    struct Pending {
        ExpertBankId bank;
        const Tensor* hidden;
        Tensor* output;
        MemoryArena* workspace;
        ops::MoeExecutionHints hints;
    };
    void begin_activity(std::int32_t) override;
    void end_activity() noexcept override;
    void begin_preparation() override;
    void end_preparation() noexcept override;
    void route_only(ExpertBankId, const Tensor&, MemoryArena&, CudaExecutionView);
    void prepare_from_route(ExpertBankId, std::int32_t, CudaExecutionView);
    void route_and_prepare(ExpertBankId, const Tensor&, MemoryArena&);
    bool advance();

    std::vector<const Parameters*> parameters_;
    std::span<const ExpertBankSource> sources_;
    const DeviceContext& device_;
    CudaExecutionView cuda_;
    CudaStream upload_stream_;
    CudaExecutionView upload_;
    ExpertCache cache_;
    ExpertCacheRequirements requirements_;
    ScheduleDescription description_;
    std::int32_t route_tokens_ = 0, tokens_ = 0, top_k_ = 0;
    DeviceBuffer route_ids_, route_coefficients_, shared_coefficients_;
    PinnedHostBuffer host_route_ids_;
    std::vector<ExpertKey> required_;
    std::vector<ExpertKey> fixed_keys_;
    std::vector<std::vector<std::optional<ExpertKey>>> layer_slots_;
    std::vector<LayoutRegion> slot_regions_;
    LayoutRegion overflow_region_;
    std::vector<ExpertPlacement> placements_;
    std::optional<ExpertCacheUse> use_;
    std::optional<Pending> pending_;
    ExpertBankId active_bank_{};
    bool preparation_ = false;
    bool graph_active_ = false;
};

class FileLayerDefinition final : public LayerGraphDefinition {
public:
    FileLayerDefinition(FileSchedule& owner, ExpertBankId bank, std::int32_t tokens)
        : owner(owner), bank(bank), tokens(tokens) {}

    std::unique_ptr<LayerGraphExecutable> instantiate() const override;

    FileSchedule& owner;
    ExpertBankId bank;
    std::int32_t tokens;
    DecodeGraphDefinition route, compute;
};

class FileLayerExecutable final : public LayerGraphExecutable {
public:
    explicit FileLayerExecutable(const FileLayerDefinition& source)
        : owner_(source.owner), bank_(source.bank), tokens_(source.tokens),
          execution_(source.owner.cuda_) {
        route_.instantiate(source.route);
        compute_.instantiate(source.compute);
    }

    void update(const LayerGraphDefinition& source) override {
        const auto& definition = dynamic_cast<const FileLayerDefinition&>(source);
        if (&definition.owner != &owner_ || definition.bank != bank_)
            throw std::logic_error("file-backed Graph update changed its schedule or bank");
        route_.update(definition.route);
        compute_.update(definition.compute);
        tokens_ = definition.tokens;
    }

    void upload(cudaStream_t stream) override {
        route_.upload(stream);
        compute_.upload(stream);
    }

    bool start(Tensor&, CudaExecutionView cuda,
               std::function<void()> after_gpu_submit) override {
        if (pending_ || owner_.graph_active_ || owner_.pending_)
            throw std::logic_error("file-backed Graph layer is already active");
        if (owner_.tokens_ != tokens_)
            throw std::logic_error("file-backed Graph route width differs from its activity");
        execution_ = cuda;
        after_gpu_submit_ = std::move(after_gpu_submit);
        route_.launch(cuda.stream());
        owner_.prepare_from_route(bank_, tokens_, cuda);
        owner_.graph_active_ = true;
        pending_ = true;
        return poll();
    }

    bool poll() override {
        if (!pending_) return true;
        try {
            const auto progress = owner_.cache_.progress();
            if (!progress.required_published) return false;
            if (!use_) use_ = owner_.cache_.acquire(owner_.required_, execution_);
            if (!use_) return false;
            compute_.launch(execution_.stream());
            owner_.cache_.release(*use_, execution_);
            use_.reset();
            owner_.required_.clear();
            owner_.graph_active_ = false;
            pending_ = false;
            if (after_gpu_submit_) after_gpu_submit_();
            after_gpu_submit_ = {};
            return true;
        } catch (...) {
            if (use_) {
                try { owner_.cache_.release(*use_, execution_); } catch (...) {}
                use_.reset();
            }
            owner_.graph_active_ = false;
            pending_ = false;
            owner_.required_.clear();
            after_gpu_submit_ = {};
            throw;
        }
    }

    void cancel() override {
        if (use_) {
            owner_.cache_.release(*use_, execution_);
            use_.reset();
        }
        if (pending_) {
            owner_.graph_active_ = false;
            owner_.required_.clear();
        }
        pending_ = false;
        after_gpu_submit_ = {};
    }

private:
    FileSchedule& owner_;
    ExpertBankId bank_;
    std::int32_t tokens_;
    DecodeGraphExecutable route_, compute_;
    CudaExecutionView execution_;
    std::optional<ExpertCacheUse> use_;
    std::function<void()> after_gpu_submit_;
    bool pending_ = false;
};

std::unique_ptr<LayerGraphExecutable> FileLayerDefinition::instantiate() const {
    return std::make_unique<FileLayerExecutable>(*this);
}

class FilePlan final : public Plan {
public:
    FilePlan(std::vector<ExpertBankCacheSpec> banks, std::vector<LayerSpec> layers,
             ExpertCacheConfig config, std::size_t capacity,
             std::vector<ExpertPlacement> fixed_layout, std::int32_t route_tokens = 0)
        : banks_(std::move(banks)), layers_(std::move(layers)), config_(config),
          capacity_(capacity), fixed_layout_(std::move(fixed_layout)), route_tokens_(route_tokens) {
        config_.capacity_bytes = capacity_;
        requirements_ = describe_expert_cache(banks_, config_);
        const auto routed_ids = std::size_t(route_tokens_) * layers_.front().profile.routed.top_k;
        const auto route_bytes = routed_ids * 2 * sizeof(std::int32_t) +
                                 std::size_t(route_tokens_) * sizeof(float);
        description_.cache_bytes = capacity_;
        description_.device_bytes = capacity_ + requirements_.offset_bytes +
                                    requirements_.divisor_bytes + route_bytes;
        description_.target_bytes = description_.device_bytes;
        description_.cache_pinned_bytes = requirements_.pinned_bytes +
                                          routed_ids * sizeof(std::int32_t);
        for (const auto& layer : layers_)
            description_.layers.push_back({layer.bank, INT32_MAX, 0, 0, layer.priority});
    }

    const ScheduleDescription& description() const noexcept override { return description_; }

    std::shared_ptr<const Plan> resolve(std::size_t available, std::int32_t max_tokens,
                                        std::uint32_t) const override {
        const auto resolved = std::make_shared<FilePlan>(banks_, layers_, config_, capacity_,
                                                         fixed_layout_, max_tokens);
        if (resolved->description().device_bytes > available) return {};
        return resolved;
    }

    std::size_t workspace_bytes(ExpertBankId bank, std::int32_t first,
                                std::int32_t last) const override {
        const auto& layer = layers_.at(layer_index(bank));
        const auto moe = ::ninfer::models::moe::workspace_bytes(
            layer.profile, layer.epilogue, first, last, ops::ExpertSubsetKind::All);
        const ops::RouteProfile route{layer.profile.routed.ffn.hidden,
                                      layer.profile.routed.experts,
                                      layer.profile.routed.top_k};
        return std::max(moe, ops::topk_softmax_route_workspace_capacity_bytes(route, first, last));
    }

    GraphPlan graph_plan(ExpertBankId, std::int32_t) const override {
        GraphPlan plan;
        plan.subsets[0] = true;
        plan.steady = ops::ExpertSubsetKind::All;
        return plan;
    }

    std::unique_ptr<Schedule> create(std::span<const Parameters* const> layers,
                                     std::span<const ExpertBankSource> sources,
                                     const DeviceContext& device, ResourcePurpose) const override {
        return std::make_unique<FileSchedule>(*this, layers, sources, device);
    }

    std::size_t layer_index(ExpertBankId bank) const {
        const auto found = std::find_if(layers_.begin(), layers_.end(),
                                        [&](const auto& layer) { return layer.bank == bank; });
        if (found == layers_.end()) throw std::invalid_argument("MoE bank is outside file schedule");
        return static_cast<std::size_t>(found - layers_.begin());
    }

    const std::vector<LayerSpec>& layers() const noexcept { return layers_; }
    const ExpertCacheRequirements& requirements() const noexcept { return requirements_; }
    std::span<const ExpertPlacement> fixed_layout() const noexcept { return fixed_layout_; }
    std::int32_t route_tokens() const noexcept { return route_tokens_; }

private:
    std::vector<ExpertBankCacheSpec> banks_;
    std::vector<LayerSpec> layers_;
    ExpertCacheConfig config_;
    std::size_t capacity_;
    std::vector<ExpertPlacement> fixed_layout_;
    std::int32_t route_tokens_;
    ExpertCacheRequirements requirements_;
    ScheduleDescription description_;
};

FileSchedule::FileSchedule(const FilePlan& plan, std::span<const Parameters* const> parameters,
                           std::span<const ExpertBankSource> sources, const DeviceContext& device)
    : parameters_(parameters.begin(), parameters.end()), sources_(sources), device_(device),
      cuda_(device.execution_view()), upload_stream_(device),
      upload_(device.execution_view(upload_stream_.stream())),
      cache_(sources, plan.requirements(), device, upload_, plan.fixed_layout()),
      requirements_(plan.requirements()), description_(plan.description()),
      route_tokens_(plan.route_tokens()) {
    if (parameters.size() != plan.layers().size() || sources.size() != requirements_.banks.size() ||
        route_tokens_ <= 0)
        throw std::invalid_argument("file-backed MoE schedule directory or route capacity is invalid");
    for (std::size_t i = 0; i < parameters.size(); ++i)
        if (!parameters[i] || parameters[i]->bank != plan.layers()[i].bank)
            throw std::invalid_argument("file-backed MoE parameter order differs from its plan");
    top_k_ = parameters_.front()->profile.routed.top_k;
    fixed_keys_.reserve(plan.fixed_layout().size());
    for (const auto& placement : plan.fixed_layout()) fixed_keys_.push_back(placement.key);
    std::sort(fixed_keys_.begin(), fixed_keys_.end(), [](auto a, auto b) {
        return a.bank.index != b.bank.index ? a.bank.index < b.bank.index : a.expert < b.expert;
    });
    layer_slots_.resize(requirements_.banks.size());
    auto slot_offset = requirements_.config.fixed_bytes;
    for (std::size_t bank = 0; bank < requirements_.banks.size(); ++bank) {
        const auto& layout = requirements_.banks[bank];
        if (layout.resident) continue;
        layer_slots_[bank].resize(kDynamicSlotsPerLayer);
        slot_regions_.resize(requirements_.banks.size() * kDynamicSlotsPerLayer);
        for (std::uint32_t slot = 0; slot < kDynamicSlotsPerLayer; ++slot) {
            slot_regions_[bank * kDynamicSlotsPerLayer + slot] =
                {slot_offset, layout.entry_bytes, 256};
            slot_offset += layout.entry_bytes;
        }
    }
    overflow_region_ = {slot_offset, requirements_.config.capacity_bytes - slot_offset, 256};
    const auto id_bytes = std::size_t(route_tokens_) * top_k_ * sizeof(std::int32_t);
    route_ids_ = DeviceBuffer(device.device(), id_bytes);
    route_coefficients_ = DeviceBuffer(device.device(), id_bytes);
    shared_coefficients_ = DeviceBuffer(device.device(), std::size_t(route_tokens_) * sizeof(float));
    host_route_ids_ = PinnedHostBuffer(id_bytes);
}

FileSchedule::~FileSchedule() {
    try { synchronize(); } catch (...) {}
}

void FileSchedule::route_and_prepare(ExpertBankId bank, const Tensor& hidden,
                                     MemoryArena& workspace) {
    route_only(bank, hidden, workspace, cuda_);
    prepare_from_route(bank, hidden.ne[1], cuda_);
}

void FileSchedule::route_only(ExpertBankId bank, const Tensor& hidden,
                              MemoryArena& workspace, CudaExecutionView cuda) {
    if (hidden.ne[1] > route_tokens_)
        throw std::invalid_argument("file-backed MoE route exceeds its planned token capacity");
    const auto found = std::find_if(parameters_.begin(), parameters_.end(),
                                    [&](const auto* p) { return p->bank == bank; });
    if (found == parameters_.end()) throw std::invalid_argument("MoE layer is outside file schedule");
    const auto& p = **found;
    const auto columns = hidden.ne[1];
    const auto id_bytes = std::size_t(top_k_) * columns * sizeof(std::int32_t);
    Tensor ids(route_ids_.span().subspan(0, id_bytes), DType::I32, {top_k_, columns});
    Tensor coefficients(route_coefficients_.span().subspan(0, id_bytes), DType::FP32,
                        {top_k_, columns});
    Tensor shared(shared_coefficients_.span().subspan(0, std::size_t(columns) * sizeof(float)),
                  DType::FP32, {columns});
    const ops::RouteProfile route{p.profile.routed.ffn.hidden, p.profile.routed.experts,
                                  p.profile.routed.top_k};
    {
        auto scope = workspace.scope();
        const auto bytes = ops::topk_softmax_route_workspace_capacity_bytes(route, columns, columns);
        MemoryArena scratch(workspace.alloc_bytes(bytes + 768));
        ops::ExpertRoutes routes{ids, coefficients};
#if NINFER_TARGET_JETSON_ORIN
        // The registered Orin NVFP4 expert path routes one token at a time. Use the
        // same route slices for file-cache planning so its expert set matches execution.
        if (p.profile.routed.ffn.gate_up_type == QType::NVFP4) {
            auto column_ids = scratch.alloc(DType::I32, {top_k_, 1});
            auto column_coefficients = scratch.alloc(DType::FP32, {top_k_, 1});
            auto column_shared = scratch.alloc(DType::FP32, {1});
            for (int column = 0; column < columns; ++column) {
                ops::ExpertRoutes column_routes{column_ids, column_coefficients};
                ops::topk_softmax_route(
                    hidden.slice(1, column, 1), route, p.control, column_routes, column_shared,
                    scratch, cuda);
                const auto ids_column = ids.slice(1, column, 1);
                const auto coefficients_column = coefficients.slice(1, column, 1);
                const auto shared_column = shared.slice(0, column, 1);
                copy_bytes_async({ids_column.data, ids_column.bytes(), ids_column.device},
                                 {column_ids.data, column_ids.bytes(), column_ids.device}, cuda);
                copy_bytes_async(
                    {coefficients_column.data, coefficients_column.bytes(),
                     coefficients_column.device},
                    {column_coefficients.data, column_coefficients.bytes(),
                     column_coefficients.device}, cuda);
                copy_bytes_async({shared_column.data, shared_column.bytes(), shared_column.device},
                                 {column_shared.data, column_shared.bytes(), column_shared.device},
                                 cuda);
            }
        } else
#endif
        {
            ops::topk_softmax_route(hidden, route, p.control, routes, shared, scratch, cuda);
        }
    }
}

void FileSchedule::prepare_from_route(ExpertBankId bank, std::int32_t columns,
                                      CudaExecutionView cuda) {
    const auto id_bytes = std::size_t(top_k_) * columns * sizeof(std::int32_t);
    Tensor ids(route_ids_.span().subspan(0, id_bytes), DType::I32, {top_k_, columns});
    copy_bytes_async(host_route_ids_.span().subspan(0, id_bytes),
                     MemorySpan{ids.data, ids.bytes(), ids.device}, cuda);
    CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
    const auto* ids_host = static_cast<const std::int32_t*>(host_route_ids_.data());
    required_ = routed_keys(bank, {ids_host, std::size_t(top_k_) * columns}, top_k_, columns);
    if (required_.empty()) throw std::logic_error("GPU router selected no experts");
    const auto& layout = requirements_.banks.at(bank.index);
    if (required_.size() > layout.count)
        throw std::logic_error("GPU router selected more unique experts than its bank");
    std::vector<ExpertKey> dynamic;
    for (const auto key : required_)
        if (!std::binary_search(fixed_keys_.begin(), fixed_keys_.end(), key,
                                [](auto a, auto b) {
                                    return a.bank.index != b.bank.index ? a.bank.index < b.bank.index
                                                                        : a.expert < b.expert;
                                }))
            dynamic.push_back(key);
    auto& slots = layer_slots_.at(bank.index);
    std::vector<std::optional<ExpertKey>> next_slots(kDynamicSlotsPerLayer);
    std::vector<ExpertKey> remaining;
    remaining.reserve(dynamic.size());
    for (const auto key : dynamic) {
        const auto found_slot = std::find(slots.begin(), slots.end(), std::optional{key});
        if (found_slot != slots.end()) {
            next_slots[static_cast<std::size_t>(found_slot - slots.begin())] = key;
        } else {
            remaining.push_back(key);
        }
    }
    std::size_t next = 0;
    for (std::uint32_t slot = 0; slot < kDynamicSlotsPerLayer && next < remaining.size(); ++slot)
        if (!next_slots[slot]) next_slots[slot] = remaining[next++];
    std::vector<ExpertKey> overflow(remaining.begin() + next, remaining.end());
    cache_.replace(overflow_region_, {});
    for (std::uint32_t slot = 0; slot < kDynamicSlotsPerLayer; ++slot) {
        if (next_slots[slot] == slots[slot]) continue;
        const auto region = slot_regions_.at(std::size_t(bank.index) * kDynamicSlotsPerLayer + slot);
        if (next_slots[slot])
        {
            std::array<ExpertPlacement, 1> placement{{{*next_slots[slot], 0}}};
            cache_.replace(region, placement);
        } else {
            cache_.replace(region, {});
        }
    }
    slots = std::move(next_slots);
    placements_ = placements_for(overflow, layout.entry_bytes);
    cache_.replace(overflow_region_, placements_);
    cache_.prepare({required_, {}});
    active_bank_ = bank;
}

bool FileSchedule::start(ExpertBankId bank, const Tensor& hidden, Tensor& output,
                         MemoryArena& workspace, const ops::MoeExecutionHints& hints) {
    if (hidden.ne[1] != tokens_) throw std::logic_error("file-backed MoE token count changed");
    if (pending_) throw std::logic_error("file-backed MoE layer is already pending");
    route_and_prepare(bank, hidden, workspace);
    pending_.emplace(Pending{bank, &hidden, &output, &workspace, hints});
    return advance();
}

bool FileSchedule::advance() {
    if (!pending_) return true;
    const auto progress = cache_.progress();
    if (!progress.required_published) return false;
    if (!use_) use_ = cache_.acquire(required_, cuda_);
    if (!use_) return false;
    const auto found = std::find_if(parameters_.begin(), parameters_.end(),
                                    [&](const auto* p) { return p->bank == pending_->bank; });
    if (found == parameters_.end()) throw std::logic_error("pending MoE layer disappeared");
    try {
        ::ninfer::models::moe::execute(*pending_->hidden, **found, cache_.view(pending_->bank),
                                       ops::MoeEpilogue::Write, *pending_->output,
                                       *pending_->workspace, cuda_, pending_->hints);
        cache_.release(*use_, cuda_);
        use_.reset();
        pending_.reset();
        required_.clear();
        return true;
    } catch (...) {
        try { CUDA_CHECK(cudaStreamSynchronize(cuda_.stream())); } catch (...) {}
        cache_.release(*use_, cuda_);
        use_.reset();
        pending_.reset();
        throw;
    }
}

bool FileSchedule::poll() { return advance(); }

void FileSchedule::cancel() {
    if (use_) {
        cache_.release(*use_, cuda_);
        use_.reset();
    }
    pending_.reset();
    required_.clear();
}

std::shared_ptr<const LayerGraphDefinition>
FileSchedule::capture_layer(ExpertBankId bank, CudaExecutionView cuda, std::int32_t tokens,
                            const std::function<void()>& mixer, const GpuBody& body,
                            const std::function<void()>& feature,
                            const std::shared_ptr<const LayerGraphDefinition>& shared) {
    if (shared) {
        const auto& previous = dynamic_cast<const FileLayerDefinition&>(*shared);
        if (&previous.owner != this || previous.bank != bank)
            throw std::logic_error("file-backed Graph sharing changed its schedule or bank");
    }
    const auto found = std::find_if(parameters_.begin(), parameters_.end(),
                                    [&](const auto* p) { return p->bank == bank; });
    if (found == parameters_.end()) throw std::invalid_argument("MoE layer is outside file schedule");
    const auto* parameters = *found;
    auto definition = std::make_shared<FileLayerDefinition>(*this, bank, tokens);
    definition->route.capture(cuda.stream(), [&] {
        mixer();
        body([&](const Tensor& hidden, Tensor&, MemoryArena& workspace,
                 const ops::MoeExecutionHints&) {
            route_only(bank, hidden, workspace, cuda);
        });
    });
    definition->compute.capture(cuda.stream(), [&] {
        body([&](const Tensor& hidden, Tensor& output, MemoryArena& workspace,
                 const ops::MoeExecutionHints& hints) {
            ::ninfer::models::moe::execute(hidden, *parameters, cache_.view(bank),
                                           ops::MoeEpilogue::Write, output, workspace, cuda, hints);
        });
        if (feature) feature();
    });
    return definition;
}

void FileSchedule::restore_cache_after_preparation() { synchronize(); }

void FileSchedule::synchronize() const {
    CUDA_CHECK(cudaStreamSynchronize(upload_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(cuda_.stream()));
}

void FileSchedule::begin_activity(std::int32_t tokens) {
    if (tokens <= 0 || tokens > route_tokens_)
        throw std::invalid_argument("file-backed MoE traversal exceeds its route capacity");
    tokens_ = tokens;
}
void FileSchedule::end_activity() noexcept { tokens_ = 0; }
void FileSchedule::begin_preparation() { preparation_ = true; }
void FileSchedule::end_preparation() noexcept { preparation_ = false; }

} // namespace

std::shared_ptr<const Plan> describe_file_schedule(std::span<const ExpertBankCacheSpec> banks,
                                                   std::span<const LayerSpec> layers) {
#if NINFER_TARGET_JETSON_ORIN
    if (layers.empty()) return {};
    ExpertCacheConfig config;
    config.max_batch_bytes = kUploadBytes;
    auto baseline = describe_expert_cache(banks, config);
    std::size_t fixed_bytes = 0, slot_bytes = 0, dynamic_bytes = 0;
    std::vector<ExpertPlacement> fixed;
    for (const auto& layer : layers) {
        const auto& bank = baseline.banks.at(layer.bank.index);
        if (bank.resident) continue;
        const auto hot = std::min<std::uint32_t>(kHotExpertsPerLayer, bank.count);
        for (std::uint32_t n = 0; n < hot; ++n) {
            fixed.push_back({{layer.bank, layer.priority.at(n)}, fixed_bytes});
            fixed_bytes += bank.entry_bytes;
        }
        slot_bytes += std::min<std::uint32_t>(kDynamicSlotsPerLayer, bank.count) * bank.entry_bytes;
        dynamic_bytes = std::max(dynamic_bytes, bank.entry_bytes * bank.count);
    }
    if (!dynamic_bytes) return {};
    const auto capacity = (fixed_bytes + slot_bytes + dynamic_bytes + 255) / 256 * 256;
    config.fixed_bytes = fixed_bytes;
    return std::make_shared<FilePlan>(std::vector<ExpertBankCacheSpec>(banks.begin(), banks.end()),
                                      std::vector<LayerSpec>(layers.begin(), layers.end()), config,
                                      capacity, std::move(fixed));
#else
    (void)banks;
    (void)layers;
    return {};
#endif
}

} // namespace ninfer::models::moe
