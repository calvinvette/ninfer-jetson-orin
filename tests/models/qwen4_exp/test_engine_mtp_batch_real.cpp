#include "ninfer/engine.h"
#include "ninfer/kv_cache_format.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

struct Run {
    std::vector<GenerationResult> requests;
};

PromptInput prompt(std::string text) {
    PromptInput input;
    ChatMessage user;
    user.role = ChatRole::User;
    user.parts.push_back({MessagePartKind::Text, std::move(text), {}});
    input.messages.push_back(std::move(user));
    return input;
}

bool use_cuda_graph() {
    const char* value = std::getenv("NINFER_TEST_CUDA_GRAPH");
    return value && std::string_view(value) == "1";
}

Run run(const std::string& artifact, bool mtp) {
    EngineOptions options;
    options.artifact_path      = artifact;
    options.max_context        = 1024;
    options.prefill_chunk      = 128;
    options.max_concurrency    = 3;
    options.kv_cache           = KvCacheStorage::Int8Group64;
    options.context_capacity   = ContextCapacityPolicy::explicit_capacity(1ULL << 30);
    options.context_cache.enabled = false;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph     = use_cuda_graph();
    if (mtp) options.speculative = {SpeculativeBackend::Mtp, 4};

    Engine engine(std::move(options));
    std::vector<GenerationHandle> handles;
    for (auto text : {"The capital of France is", "The number after 2 is",
                      "The largest ocean on Earth is"}) {
        RequestOptions request;
        request.execution.requested_output_tokens = 8;
        request.execution.sampling.temperature     = 0.0F;
        request.execution.sampling.seed            = 0;
        request.stop.include_model_defaults       = false;
        handles.push_back(engine.submit(engine.prepare(prompt(text)), std::move(request)));
    }

    Run result;
    for (auto& handle : handles) result.requests.push_back(handle.wait());
    return result;
}

GenerationResult run_long(const std::string& artifact, bool mtp, std::string prompt_text,
                          std::uint32_t output_tokens, std::uint32_t draft_tokens = 4) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 1024;
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(1ULL << 30);
    options.context_cache.enabled = false;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph = use_cuda_graph();
    if (mtp) options.speculative = {SpeculativeBackend::Mtp, draft_tokens};
    Engine engine(std::move(options));
    RequestOptions request;
    request.execution.requested_output_tokens = output_tokens;
    request.execution.sampling.temperature = 0.0F;
    request.execution.sampling.seed = 0;
    request.stop.include_model_defaults = false;
    return engine.generate(engine.prepare(prompt(std::move(prompt_text))), std::move(request));
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int main() {
#if !defined(NINFER_TARGET_JETSON_ORIN)
    std::cout << "skip: this is an SM87 real-artifact regression\n";
    return 77;
#else
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        const char* selected_prompt = std::getenv("NINFER_TEST_LONG_PROMPT");
        const char* selected_tokens = std::getenv("NINFER_TEST_LONG_TOKENS");
        const char* selected_drafts = std::getenv("NINFER_TEST_DRAFT_TOKENS");
        const auto output_tokens = selected_tokens && *selected_tokens
                                       ? static_cast<std::uint32_t>(std::stoul(selected_tokens))
                                       : 64U;
        const auto draft_tokens = selected_drafts && *selected_drafts
                                      ? static_cast<std::uint32_t>(std::stoul(selected_drafts))
                                      : 4U;
        std::cout << "cuda_graph=" << (use_cuda_graph() ? "true" : "false") << '\n';
        require(output_tokens > 0, "long MTP output token count must be positive");
        require(draft_tokens > 0, "MTP draft token count must be positive");
        const std::vector<std::string> long_prompts =
            selected_prompt && *selected_prompt
                ? std::vector<std::string>{selected_prompt}
                : std::vector<std::string>{"hi"};
        for (const auto& prompt_text : long_prompts) {
            const auto serial_long = run_long(artifact, false, prompt_text, output_tokens);
            const auto mtp_long = run_long(artifact, true, prompt_text, output_tokens, draft_tokens);
            std::cout << "prompt=" << prompt_text << " mtp drafts="
                      << mtp_long.stats.speculative.drafted_tokens << " accepted="
                      << mtp_long.stats.speculative.accepted_tokens << " fallback_steps="
                      << mtp_long.stats.speculative.fallback_steps << '\n';
            if (serial_long.generated_token_ids != mtp_long.generated_token_ids) {
                std::size_t first_difference = 0;
                while (first_difference < serial_long.generated_token_ids.size() &&
                       first_difference < mtp_long.generated_token_ids.size() &&
                       serial_long.generated_token_ids[first_difference] ==
                           mtp_long.generated_token_ids[first_difference])
                    ++first_difference;
                std::cerr << "MTP prompt=" << prompt_text << " first_difference="
                          << first_difference << " serial=";
                for (const auto token : serial_long.generated_token_ids)
                    std::cerr << token << ' ';
                std::cerr << "\nmtp=";
                for (const auto token : mtp_long.generated_token_ids) std::cerr << token << ' ';
                std::cerr << '\n';
            }
            require(serial_long.generated_token_ids.size() == output_tokens &&
                        mtp_long.generated_token_ids == serial_long.generated_token_ids,
                    "long greedy MTP output differs from non-speculative decode");
            std::cout << "prompt=" << prompt_text << " serial=";
            for (auto token : serial_long.generated_token_ids) std::cout << token << ' ';
            std::cout << "\nprompt=" << prompt_text << " mtp=";
            for (auto token : mtp_long.generated_token_ids) std::cout << token << ' ';
            std::cout << '\n';
            std::cout << "PASS exact " << output_tokens << "-token MTP parity for Flash-Next prompt "
                      << prompt_text << '\n';
        }
        if (selected_prompt && *selected_prompt) return 0;
        const auto reference = run(artifact, false);
        const auto mtp       = run(artifact, true);
        require(reference.requests.size() == 3 && mtp.requests.size() == 3,
                "three-request cohort did not complete");
        std::uint64_t batched_decode_steps = 0;
        for (std::size_t i = 0; i < reference.requests.size(); ++i) {
            const auto& expected = reference.requests[i];
            const auto& actual   = mtp.requests[i];
            require(expected.generated_token_ids.size() == 8 &&
                        actual.generated_token_ids == expected.generated_token_ids,
                    "batched MTP token IDs differ from non-speculative decode");
            require(actual.stats.speculative.backend == SpeculativeBackend::Mtp &&
                        actual.stats.speculative.drafted_tokens > 0,
                    "MTP request did not execute draft work");
            if (actual.stats.decode_by_batch.size() > 2)
                batched_decode_steps += actual.stats.decode_by_batch[2].total.count;
            std::cout << "row=" << i << " ids=";
            for (const auto token : actual.generated_token_ids) std::cout << token << ' ';
            std::cout << "drafted=" << actual.stats.speculative.drafted_tokens
                      << " accepted=" << actual.stats.speculative.accepted_tokens << '\n';
        }
        require(batched_decode_steps > 0,
                "requests did not exercise a three-row decode batch");
        std::cout << "PASS exact MTP parity for a three-request real Flash-Next cohort\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
#endif
}
