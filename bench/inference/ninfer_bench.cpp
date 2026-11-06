#include "options.h"
#include "report.h"

#include "ninfer/engine.h"
#include "product/logging/configuration.h"
#include "product/logging/startup_log.h"
#include "product/prompt_input/prompt_input.h"

#include <spdlog/logger.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <variant>

namespace {

using namespace ninfer;

using BenchmarkInput = std::variant<PromptInput, std::vector<TokenId>>;

PreparedPrompt prepare_input(Engine& engine, const BenchmarkInput& input) {
    if (const auto* tokens = std::get_if<std::vector<TokenId>>(&input))
        return engine.prepare_tokens(*tokens, false);
    return engine.prepare(std::get<PromptInput>(input));
}

std::vector<TokenId> read_token_ids(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open token-ID input: " + path.string());
    std::vector<TokenId> tokens;
    std::int64_t token = 0;
    while (file >> token) {
        if (token < 0 || token > std::numeric_limits<TokenId>::max())
            throw std::invalid_argument("token-ID input contains an out-of-range ID: " +
                                        path.string());
        tokens.push_back(static_cast<TokenId>(token));
    }
    if (!file.eof()) throw std::invalid_argument("token-ID input is not whitespace-separated IDs: " +
                                                 path.string());
    if (tokens.empty()) throw std::invalid_argument("token-ID input is empty: " + path.string());
    return tokens;
}

bench::Repetition run(Engine& engine, const bench::Options& options,
                      const std::vector<BenchmarkInput>& inputs,
                      std::span<const RequestOptions> requests, bool measured) {
    if (options.launch.generation.max_tokens > 1) {
        for (std::size_t row = 0; row + 1 < requests.size(); ++row) {
            auto prompt  = prepare_input(engine, inputs[inputs.size() == 1 ? 0 : row]);
            auto prefill = requests[row];
            prefill.execution.requested_output_tokens = 1;
            auto handle = engine.submit(std::move(prompt), std::move(prefill));
            (void)handle.wait();
        }
    }
    std::vector<PreparedPrompt> prompts;
    std::vector<GenerationHandle> handles;
    bench::Repetition repetition;
    prompts.reserve(requests.size());
    handles.reserve(requests.size());
    repetition.requests.reserve(requests.size());
    for (std::size_t row = 0; row < requests.size(); ++row)
        prompts.push_back(prepare_input(engine, inputs[inputs.size() == 1 ? 0 : row]));
    if (measured) {
        repetition.before = engine.begin_measurement({.profile = options.profile_measured});
    }
    const auto started = std::chrono::steady_clock::now();
    try {
        for (std::size_t row = 0; row < requests.size(); ++row)
            handles.push_back(engine.submit(std::move(prompts[row]), requests[row],
                                            OutputConsumerMode::Aggregate));
        for (auto& handle : handles) repetition.requests.push_back(handle.wait());
        repetition.wall_seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (measured) repetition.after = engine.end_measurement();
    } catch (...) {
        const auto error = std::current_exception();
        handles.clear();
        if (measured) {
            try {
                (void)engine.end_measurement();
            } catch (...) {}
        }
        std::rethrow_exception(error);
    }
    return repetition;
}

void write_report(const bench::Options& options, const std::string& report) {
    if (options.output_file.empty()) {
        std::cout << report;
        return;
    }
    const auto parent = options.output_file.parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream file(options.output_file);
    if (!file) throw std::runtime_error("cannot open report: " + options.output_file.string());
    file << report;
    file.close();
    if (!file) throw std::runtime_error("cannot write report: " + options.output_file.string());
    std::cerr << "wrote " << options.output_file << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = bench::parse_options(argc, argv);
        if (options.help_requested) {
            std::cout << bench::usage_text(argc ? argv[0] : "ninfer_bench");
            return 0;
        }
        product::LoggingRuntime logging({.logger_name  = "ninfer_bench",
                                         .level        = options.launch.logging.level,
                                         .presentation = product::LogPresentation::Tool});
        const auto logger = logging.logger();
        product::log_configuration(
            *logger, product::configuration_view(options.launch, product::EntryPoint::Bench));
        product::StartupLogRenderer startup_log(logging);
        auto engine_options             = options.launch.engine;
        engine_options.startup_observer = startup_log.observer();

        const auto& defaults = options.launch.generation;
        std::vector<BenchmarkInput> inputs;
        for (const auto& source : options.inputs) {
            if (source.kind == bench::InputKind::TokenIds) {
                inputs.emplace_back(read_token_ids(source.value));
            } else {
                auto input = source.kind == bench::InputKind::Prompt
                                 ? product::prompt_from_text(source.value, defaults.enable_thinking)
                                 : product::prompt_from_messages(source.value,
                                                                 defaults.enable_thinking,
                                                                 engine_options.enable_vision);
                input.options.preserve_thinking = defaults.preserve_thinking;
                inputs.emplace_back(std::move(input));
            }
        }
        std::vector<RequestOptions> requests(options.concurrency);
        std::vector<std::uint64_t> seeds;
        for (std::size_t row = 0; row < requests.size(); ++row) {
            auto& request = requests[row];
            request.execution.sampling = defaults.sampling;
            if (!request.execution.sampling.seed)
                request.execution.sampling.seed = product::random_sampling_seed();
            seeds.push_back(*request.execution.sampling.seed);
            request.execution.requested_output_tokens = defaults.max_tokens;
            request.execution.thinking.budget         = defaults.thinking_budget;
            const auto& source = options.inputs[options.inputs.size() == 1 ? 0 : row];
            if (source.kind == bench::InputKind::TokenIds) {
                request.stop.include_model_defaults    = false;
                request.output.raw                     = true;
                request.output.preserve_special_tokens = true;
            }
        }

        Engine engine(std::move(engine_options));
        startup_log.engine_ready(engine.startup_report().load);
        product::log_resolved_configuration(*logger, engine.description());
        const auto initial = engine.snapshot();
        logger->info("benchmark: {} requests, {} output tokens/request, {} warmup, {} repetitions",
                     options.concurrency, options.launch.generation.max_tokens, options.warmup,
                     options.repetitions);
        for (std::uint32_t i = 0; i < options.warmup; ++i) {
            (void)run(engine, options, inputs, requests, false);
            logger->info("warmup {}/{} complete", i + 1, options.warmup);
        }
        std::vector<bench::Repetition> repetitions;
        repetitions.reserve(options.repetitions);
        for (std::uint32_t i = 0; i < options.repetitions; ++i) {
            repetitions.push_back(run(engine, options, inputs, requests, true));
            logger->info("repetition {}/{} complete", i + 1, options.repetitions);
        }
        const auto report = bench::make_report(
            options, engine.description(), engine.startup_report(), initial, seeds, repetitions);
        write_report(options, bench::format_report(report, options.output));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_bench: " << error.what() << '\n';
        return 1;
    }
}
