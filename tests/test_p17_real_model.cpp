// tests/test_p17_real_model.cpp
// Phase 17: Full HydroXL pipeline integration test using the real Qwen2.5-0.5B model.
// Exits 77 (SKIP) if model file is absent.

#include "mini_test.h"
#include "hydroxl/HydroXLEngine.h"
#include "engine/ExecutionContext.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

using namespace agr;

static const char* DEFAULT_MODEL =
    "C:\\HeteroAccel\\Models\\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";

static const char* modelPath() {
    const char* e = std::getenv("HETEROACCEL_MODEL_PATH");
    return (e && *e) ? e : DEFAULT_MODEL;
}
static bool modelAvailable() { return std::filesystem::exists(modelPath()); }

int main() {
    if (!modelAvailable()) {
        std::cout << "[SKIP] Model not found: " << modelPath() << "\n";
        std::cout << "       Set HETEROACCEL_MODEL_PATH env var to override.\n";
        return 77;
    }
    std::cout << "[INFO] Model: " << modelPath() << "\n";

    // --- Load ---
    HydroXLEngine eng;
    AGR_CHECK(eng.initialize());

    ModelDescriptor desc;
    desc.model_id    = "qwen2.5-0.5b";
    desc.source_path = modelPath();
    desc.format      = ModelFormat::GGUF;
    desc.architecture = "qwen2";

    {   auto t0 = std::chrono::steady_clock::now();
        bool ok = eng.loadModel(desc);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        if (!ok) std::cerr << "[FAIL] loadModel: " << eng.lastError() << "\n";
        AGR_CHECK(ok);
        AGR_CHECK(eng.isModelLoaded());
        std::cout << "[INFO] Loaded in " << ms << " ms\n";
    }

    // --- Short greedy inference ---
    {   ExecutionContext ctx;
        ctx.prompt      = "The capital of France is";
        ctx.max_tokens  = 16;
        ctx.temperature = 0.0f;  // greedy
        ctx.seed        = 42;

        auto t0 = std::chrono::steady_clock::now();
        auto result = eng.execute(ctx);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

        std::cout << "[INFO] success=" << result.success << "\n";
        std::cout << "[INFO] output='" << result.output << "'\n";
        std::cout << "[INFO] prompt_tokens=" << result.telemetry.prompt_tokens
                  << " output_tokens=" << result.telemetry.output_tokens << "\n";
        std::cout << "[INFO] prompt_ms=" << result.telemetry.prompt_eval_ms
                  << " gen_ms=" << result.telemetry.generation_ms
                  << " tok/s=" << result.telemetry.tokens_per_sec << "\n";
        std::cout << "[INFO] backend=" << result.telemetry.backend_name
                  << " gpu_used=" << result.telemetry.gpu_used << "\n";
        std::cout << "[INFO] wall_ms=" << ms << "\n";

        AGR_CHECK(result.success);
        AGR_CHECK(!result.output.empty());
        AGR_CHECK(result.telemetry.output_tokens > 0);
        AGR_CHECK(!result.telemetry.gpu_used);          // CPU only
        AGR_CHECK(result.telemetry.backend_name == "HydroXL");
    }

    // --- Streaming ---
    {   ExecutionContext ctx;
        ctx.prompt      = "Hello";
        ctx.max_tokens  = 8;
        ctx.temperature = 0.0f;
        ctx.seed        = 42;

        int cb_count = 0;
        std::string streamed;
        auto result = eng.executeStreaming(ctx, [&](const std::string& piece){
            streamed += piece;
            ++cb_count;
        });

        std::cout << "[INFO] Streamed " << cb_count << " pieces: '" << streamed << "'\n";
        AGR_CHECK(result.success);
        AGR_CHECK(streamed == result.output);
    }

    // --- Unload + reload ---
    {   eng.unloadModel();
        AGR_CHECK(!eng.isModelLoaded());
        AGR_CHECK(eng.loadModel(desc));
        AGR_CHECK(eng.isModelLoaded());
    }

    AGR_TEST_MAIN_END()
}
