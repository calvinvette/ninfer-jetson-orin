#include "ninfer/engine.h"
#include "ninfer/kv_cache_format.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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

Run run(const std::string& artifact, bool mtp) {
    EngineOptions options;
    options.artifact_path      = artifact;
    options.max_context        = 1024;
    options.prefill_chunk      = 128;
    options.max_concurrency    = 2;
    options.kv_cache           = KvCacheStorage::Int8Group64;
    options.context_capacity   = ContextCapacityPolicy::explicit_capacity(1ULL << 30);
    options.context_cache.enabled = false;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph     = false;
    if (mtp) options.speculative = {SpeculativeBackend::Mtp, 4};

    Engine engine(std::move(options));
    std::vector<GenerationHandle> handles;
    for (auto text : {"The capital of France is", "The number after 2 is"}) {
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

GenerationResult run_long(const std::string& artifact, bool mtp) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 1024;
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.context_capacity = ContextCapacityPolicy::explicit_capacity(1ULL << 30);
    options.context_cache.enabled = false;
    options.context_cache.host_capacity_bytes = 0;
    options.use_cuda_graph = false;
    if (mtp) options.speculative = {SpeculativeBackend::Mtp, 4};
    Engine engine(std::move(options));
    RequestOptions request;
    request.execution.requested_output_tokens = 32;
    request.execution.sampling.temperature = 0.0F;
    request.execution.sampling.seed = 0;
    request.stop.include_model_defaults = false;
    return engine.generate(engine.prepare(prompt("hi")), std::move(request));
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
        const auto serial_long = run_long(artifact, false);
        const auto mtp_long = run_long(artifact, true);
        require(serial_long.generated_token_ids.size() == 32 &&
                    mtp_long.generated_token_ids == serial_long.generated_token_ids,
                "32-token greedy MTP output differs from non-speculative decode");
        std::cout << "long serial=";
        for (auto token : serial_long.generated_token_ids) std::cout << token << ' ';
        std::cout << "\nlong mtp=";
        for (auto token : mtp_long.generated_token_ids) std::cout << token << ' ';
        std::cout << '\n';
        std::cout << "PASS exact 32-token MTP parity for Flash-Next\n";
        const auto reference = run(artifact, false);
        const auto mtp       = run(artifact, true);
        require(reference.requests.size() == 2 && mtp.requests.size() == 2,
                "two-request cohort did not complete");
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
            if (actual.stats.decode_by_batch.size() > 1)
                batched_decode_steps += actual.stats.decode_by_batch[1].total.count;
            std::cout << "row=" << i << " ids=";
            for (const auto token : actual.generated_token_ids) std::cout << token << ' ';
            std::cout << "drafted=" << actual.stats.speculative.drafted_tokens
                      << " accepted=" << actual.stats.speculative.accepted_tokens << '\n';
        }
        require(batched_decode_steps > 0,
                "requests did not exercise a two-row decode batch");
        std::cout << "PASS exact MTP parity for a two-request real Flash-Next cohort\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
#endif
}
