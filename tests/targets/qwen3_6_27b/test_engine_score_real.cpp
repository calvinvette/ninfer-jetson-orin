#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_QWEN3_6_27B_WEIGHTS is not set\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.max_context   = 2048;
    const auto score_kv =
#if defined(NINFER_SCORE_KV_BF16)
        ninfer::KvCacheStorage::BFloat16;
#else
        ninfer::KvCacheStorage::Fp8E4M3Row256;
#endif
    options.kv_cache = score_kv;
    std::unique_ptr<ninfer::Engine> engine;
    try {
        engine = std::make_unique<ninfer::Engine>(options);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("FP8 E4M3 causal attention") != std::string::npos) {
            std::cout << "SKIP: FP8 causal scoring is unavailable on this GPU\n";
            return 77;
        }
        throw;
    }
    const auto& effective = engine->options();
    if (effective.max_concurrency != 1 || effective.prefill_chunk != 1024 ||
        effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
        effective.kv_capacity.explicit_tokens != effective.max_context ||
        effective.context_cache.enabled ||
        effective.speculative.backend != ninfer::SpeculativeBackend::None ||
        effective.kv_cache != score_kv) {
        std::cerr << "causal scoring options were not normalized correctly\n";
        return 1;
    }

    std::string text;
    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";
    std::vector<ninfer::TokenId> tokens;
    while (tokens.size() < 1537) {
        text += paragraph;
        tokens = engine->tokenize_text(text);
    }
    tokens.resize(1537);

    const std::vector<float> all      = engine->score_tokens(tokens, 1);
    const std::vector<float> suffix   = engine->score_tokens(tokens, 513);
    const std::vector<float> repeated = engine->score_tokens(tokens, 513);
    if (all.size() != 1536 || suffix.size() != 1024 || repeated.size() != suffix.size()) {
        std::cerr << "causal scoring returned an invalid result shape\n";
        return 1;
    }
    float maximum_overlap_error = 0.0F;
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        if (!std::isfinite(all[i + 512]) || !std::isfinite(suffix[i])) {
            std::cerr << "causal scoring returned a non-finite logprob\n";
            return 1;
        }
        maximum_overlap_error = std::max(maximum_overlap_error, std::abs(all[i + 512] - suffix[i]));
        if (suffix[i] != repeated[i]) {
            std::cerr << "a repeated score window inherited prior State/KV\n";
            return 1;
        }
    }
    if (maximum_overlap_error > 0.25F) {
        std::cerr << "overlapping target suffix changed by " << maximum_overlap_error << '\n';
        return 1;
    }
    std::cout << "OK causal_score_real max_overlap_error=" << maximum_overlap_error << '\n';
    return 0;
}
