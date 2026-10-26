// tests/test_p20_context_validation.cpp
// Phase 20: Context length & generation validation for HydroXL native engine.
// Exits 77 (SKIP) if model file is absent.

#include "mini_test.h"
#include "hydroxl/HydroXLEngine.h"
#include "engine/ExecutionContext.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace agr;

static const char* DEFAULT_MODEL =
    "C:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

static const char* modelPath() {
    const char* e = std::getenv("HETEROACCEL_MODEL_PATH");
    return (e && *e) ? e : DEFAULT_MODEL;
}

static bool modelAvailable() {
    return std::filesystem::exists(modelPath());
}

static bool containsNaNOrInf(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    return (lower.find("nan") != std::string::npos ||
            lower.find("inf") != std::string::npos);
}

int main() {
    if (!modelAvailable()) {
        std::cout << "[SKIP] Model not found: " << modelPath() << "\n";
        std::cout << "       Set HETEROACCEL_MODEL_PATH env var to override.\n";
        return 77;
    }
    std::cout << "[INFO] Validating HydroXL with model: " << modelPath() << "\n";

    HydroXLEngine eng;
    AGR_CHECK(eng.initialize());

    ModelDescriptor desc;
    desc.model_id     = "qwen2.5-0.5b";
    desc.source_path  = modelPath();
    desc.format       = ModelFormat::GGUF;
    desc.architecture = "qwen2";

    bool loaded = eng.loadModel(desc);
    if (!loaded) {
        std::cerr << "[FAIL] loadModel error: " << eng.lastError() << "\n";
    }
    AGR_CHECK(loaded);
    AGR_CHECK(eng.isModelLoaded());

    // =======================================================================
    // 1. Short prompt generation (~5 tokens)
    // =======================================================================
    std::cout << "\n--- Test 1: Short prompt (5 tokens) ---\n";
    std::string short_output_1;
    {
        ExecutionContext ctx;
        ctx.prompt      = "The capital of France is";
        ctx.max_tokens  = 8;
        ctx.temperature = 0.0f; // greedy
        ctx.seed        = 42;

        auto result = eng.execute(ctx);
        std::cout << "[INFO] short prompt tokens=" << result.telemetry.prompt_tokens
                  << " output_tokens=" << result.telemetry.output_tokens
                  << " output='" << result.output << "'\n";

        AGR_CHECK(result.success);
        AGR_CHECK(result.telemetry.prompt_tokens > 0);
        AGR_CHECK(result.telemetry.output_tokens == 8);
        AGR_CHECK(!result.output.empty());
        AGR_CHECK(!containsNaNOrInf(result.output));

        short_output_1 = result.output;
    }

    // =======================================================================
    // 2. Repeated inference calls on the same loaded engine (KV cache reset)
    // =======================================================================
    std::cout << "\n--- Test 2: Repeated inference calls & KV cache reuse ---\n";
    {
        // 2a. Same prompt again -> must reproduce identical output
        ExecutionContext ctx;
        ctx.prompt      = "The capital of France is";
        ctx.max_tokens  = 8;
        ctx.temperature = 0.0f;
        ctx.seed        = 42;

        auto result = eng.execute(ctx);
        AGR_CHECK(result.success);
        AGR_CHECK(result.output == short_output_1);

        // 2b. Different prompt -> must produce different coherent output
        ExecutionContext ctx2;
        ctx2.prompt      = "2 + 2 =";
        ctx2.max_tokens  = 4;
        ctx2.temperature = 0.0f;
        ctx2.seed        = 42;

        auto result2 = eng.execute(ctx2);
        std::cout << "[INFO] math prompt tokens=" << result2.telemetry.prompt_tokens
                  << " output='" << result2.output << "'\n";

        AGR_CHECK(result2.success);
        AGR_CHECK(!result2.output.empty());
        AGR_CHECK(result2.output != short_output_1);
        AGR_CHECK(!containsNaNOrInf(result2.output));
    }

    // =======================================================================
    // 3. Zero-length prompt handling
    // =======================================================================
    std::cout << "\n--- Test 3: Zero-length prompt handling ---\n";
    {
        ExecutionContext ctx_zero;
        ctx_zero.prompt     = "";
        ctx_zero.max_tokens = 8;

        auto result = eng.execute(ctx_zero);
        std::cout << "[INFO] zero-length prompt success=" << result.success
                  << " error='" << result.error << "'\n";

        // Must reject gracefully without crash
        AGR_CHECK(!result.success);
        AGR_CHECK(!result.error.empty());

        // Verify engine is still functional after handling zero-length prompt
        ExecutionContext ctx_rec;
        ctx_rec.prompt     = "Hello";
        ctx_rec.max_tokens = 3;
        ctx_rec.temperature = 0.0f;
        auto result_rec = eng.execute(ctx_rec);
        AGR_CHECK(result_rec.success);
        AGR_CHECK(!result_rec.output.empty());
    }

    // =======================================================================
    // 4. Medium prompt generation (~64 tokens)
    // =======================================================================
    std::cout << "\n--- Test 4: Medium prompt (~64 tokens) ---\n";
    {
        ExecutionContext ctx_med;
        ctx_med.prompt =
            "Artificial intelligence and machine learning architectures have evolved "
            "rapidly over the past decade. Modern large language models use multi-head "
            "self-attention and deep transformer layers to process textual representations, "
            "predict upcoming tokens in sequence, and generate coherent human language outputs.";
        ctx_med.max_tokens  = 6;
        ctx_med.temperature = 0.0f;
        ctx_med.seed        = 42;

        auto result = eng.execute(ctx_med);
        std::cout << "[INFO] medium prompt tokens=" << result.telemetry.prompt_tokens
                  << " gen_tokens=" << result.telemetry.output_tokens
                  << " output='" << result.output << "'\n";

        AGR_CHECK(result.success);
        AGR_CHECK(result.telemetry.prompt_tokens >= 30);
        AGR_CHECK(result.telemetry.output_tokens == 6);
        AGR_CHECK(!result.output.empty());
        AGR_CHECK(!containsNaNOrInf(result.output));
    }

    // =======================================================================
    // 5. Boundary context prompts (near and exceeding context limit)
    // =======================================================================
    std::cout << "\n--- Test 5: Boundary context prompt handling ---\n";
    {
        // 5a. Prompt exceeding specified context limit -> reject upfront
        ExecutionContext ctx_over;
        ctx_over.prompt     = "This is a prompt that has several tokens in it.";
        ctx_over.n_ctx      = 4; // limit context to 4 tokens
        ctx_over.max_tokens = 4;

        auto result_over = eng.execute(ctx_over);
        std::cout << "[INFO] ctx overflow prompt tokens=" << result_over.telemetry.prompt_tokens
                  << " success=" << result_over.success
                  << " error='" << result_over.error << "'\n";

        AGR_CHECK(!result_over.success);
        AGR_CHECK(result_over.error.find("max_seq_len") != std::string::npos);

        // 5b. Prompt reaching context limit during generation -> truncate gracefully
        ExecutionContext ctx_boundary;
        ctx_boundary.prompt      = "The sky is"; // ~3-4 tokens
        ctx_boundary.n_ctx       = 6;            // total budget of 6 tokens
        ctx_boundary.max_tokens  = 10;           // requested 10 tokens
        ctx_boundary.temperature = 0.0f;
        ctx_boundary.seed        = 42;

        auto result_bound = eng.execute(ctx_boundary);
        std::cout << "[INFO] boundary prompt tokens=" << result_bound.telemetry.prompt_tokens
                  << " gen_tokens=" << result_bound.telemetry.output_tokens
                  << " success=" << result_bound.success
                  << " output='" << result_bound.output << "'\n";

        AGR_CHECK(result_bound.success);
        // Total tokens processed must not exceed n_ctx
        int total_tokens = result_bound.telemetry.prompt_tokens + result_bound.telemetry.output_tokens;
        AGR_CHECK(total_tokens <= 6);
        AGR_CHECK(!result_bound.output.empty());
        AGR_CHECK(!containsNaNOrInf(result_bound.output));

        // 5c. Engine recovery check: ensure engine operates normally after boundary hits
        ExecutionContext ctx_clean;
        ctx_clean.prompt      = "Sun is";
        ctx_clean.max_tokens  = 3;
        ctx_clean.temperature = 0.0f;
        ctx_clean.seed        = 42;

        auto result_clean = eng.execute(ctx_clean);
        AGR_CHECK(result_clean.success);
        AGR_CHECK(!result_clean.output.empty());
    }

    std::cout << "\n[INFO] All context validation tests passed!\n";
    AGR_TEST_MAIN_END()
}
