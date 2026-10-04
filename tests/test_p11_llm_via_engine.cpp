// tests/test_p11_llm_via_engine.cpp
// Phase 11: Real-model inference through the IExecutionEngine interface.
//
// Requires:
//   HETEROACCEL_MODEL_PATH=<path-to-valid-gguf>
// Skips (exit 77) if not set.
//
// Validates:
//  1. CPU inference through IExecutionEngine
//  2. Vulkan inference through IExecutionEngine (if Vulkan available)
//  3. ExecutionResult carries correct backend / telemetry
//  4. vulkan_confirmed semantics preserved through the new interface
//  5. Cancellation through IExecutionEngine
//  6. Streaming callback through IExecutionEngine::executeStreaming

#include "mini_test.h"
#include "engine/ExecutionEngineRegistry.h"
#include "engine/IExecutionEngine.h"
#include "engine/LlamaCppExecutionEngine.h"
#include "engine/ExecutionContext.h"
#include "engine/ModelDescriptor.h"
#include "hardware/HardwareDetector.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace agr;

// ---------------------------------------------------------------------------
// Helper: build a ModelDescriptor for the test GGUF
// ---------------------------------------------------------------------------
static ModelDescriptor makeDescriptor(const std::string& path, bool cpu_only,
                                      int n_gpu_layers = 99)
{
    ModelDescriptor d;
    d.model_id    = path;
    d.source_path = path;
    d.format      = ModelFormat::GGUF;
    if (cpu_only) {
        d.metadata["cpu_only"]     = "1";
        d.metadata["n_gpu_layers"] = "0";
    } else {
        d.metadata["n_gpu_layers"] = std::to_string(n_gpu_layers);
    }
    // Small context for fast test runs
    d.metadata["n_ctx"] = "512";
    return d;
}

// ---------------------------------------------------------------------------
// Helper: build ExecutionContext
// ---------------------------------------------------------------------------
static ExecutionContext makeCtx(bool cpu_only, int n_gpu_layers = 99) {
    ExecutionContext ctx;
    ctx.prompt       = "Hello! Please answer in one word: what is 2+2?";
    ctx.max_tokens   = 32;
    ctx.temperature  = 0.1f;
    ctx.n_ctx        = 512;
    ctx.n_batch      = 128;
    ctx.cpu_only     = cpu_only;
    ctx.n_gpu_layers = cpu_only ? 0 : n_gpu_layers;
    return ctx;
}

int main() {
    std::cout << "== test_p11_llm_via_engine ==\n";

    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    if (!model_env || std::string(model_env).empty()) {
        std::cout << "  SKIP: HETEROACCEL_MODEL_PATH not set.\n";
        return 77;
    }
    std::string model_path(model_env);

    // ----------------------------------------------------------------
    // 1. Obtain engine from registry
    // ----------------------------------------------------------------
    auto& registry = ExecutionEngineRegistry::instance();
    AGR_CHECK(registry.isRegistered("llama.cpp"));
    auto engine = registry.create("llama.cpp");
    AGR_CHECK(engine != nullptr);
    AGR_CHECK(engine->initialize());

    // ----------------------------------------------------------------
    // 2. CPU inference
    // ----------------------------------------------------------------
    {
        ModelDescriptor desc = makeDescriptor(model_path, /*cpu_only=*/true);
        AGR_CHECK(engine->loadModel(desc));
        AGR_CHECK(engine->isModelLoaded());

        ExecutionContext ctx = makeCtx(/*cpu_only=*/true);
        ExecutionResult r = engine->execute(ctx);

        std::cout << "  CPU output: " << r.output << "\n";
        std::cout << "  CPU backend_type: "
                  << toString(r.telemetry.backend_type) << "\n";
        std::cout << "  CPU gpu_layers: " << r.telemetry.gpu_layers << "\n";
        std::cout << "  CPU total_ms: " << r.telemetry.total_ms << "\n";

        AGR_CHECK(r.success);
        AGR_CHECK(!r.output.empty());
        AGR_CHECK(r.telemetry.backend_type == ExecutionBackendType::CPU);
        AGR_CHECK(r.telemetry.gpu_layers == 0);
        AGR_CHECK(!r.telemetry.gpu_used);
        AGR_CHECK(r.telemetry.output_tokens > 0);
        AGR_CHECK(r.telemetry.total_ms > 0.0);

        std::cout << "  ok: CPU inference through IExecutionEngine\n";
    }

    // ----------------------------------------------------------------
    // 3. Vulkan inference (skip if no Vulkan)
    // ----------------------------------------------------------------
    {
        HardwareInfo hw = HardwareDetector::detectAll();
        if (!hw.vulkan.available) {
            std::cout << "  SKIP: Vulkan not available, skipping GPU test.\n";
        } else {
            engine->unloadModel();

            ModelDescriptor desc = makeDescriptor(model_path, /*cpu_only=*/false, 99);
            AGR_CHECK(engine->loadModel(desc));
            AGR_CHECK(engine->isModelLoaded());

            ExecutionContext ctx = makeCtx(/*cpu_only=*/false, 99);
            ExecutionResult r = engine->execute(ctx);

            std::cout << "  Vulkan output: " << r.output << "\n";
            std::cout << "  Vulkan backend_type: "
                      << toString(r.telemetry.backend_type) << "\n";
            std::cout << "  Vulkan gpu_layers: " << r.telemetry.gpu_layers << "\n";

            AGR_CHECK(r.success);
            AGR_CHECK(!r.output.empty());
            AGR_CHECK(r.telemetry.backend_type == ExecutionBackendType::VULKAN);
            AGR_CHECK(r.telemetry.gpu_layers > 0);
            AGR_CHECK(r.telemetry.gpu_used);
            AGR_CHECK(r.telemetry.output_tokens > 0);

            std::cout << "  ok: Vulkan inference through IExecutionEngine\n";
        }
    }

    // ----------------------------------------------------------------
    // 4. Mode distinction: CPU backend_type != Vulkan backend_type
    //    (runs both; ensures vulkan_confirmed semantics are preserved)
    // ----------------------------------------------------------------
    {
        // CPU run
        engine->unloadModel();
        ModelDescriptor cpuDesc = makeDescriptor(model_path, true);
        AGR_CHECK(engine->loadModel(cpuDesc));
        ExecutionContext cpuCtx = makeCtx(true);
        ExecutionResult cpuR = engine->execute(cpuCtx);
        AGR_CHECK(cpuR.success);
        AGR_CHECK(cpuR.telemetry.backend_type == ExecutionBackendType::CPU);
        AGR_CHECK(!cpuR.telemetry.gpu_used);

        HardwareInfo hw = HardwareDetector::detectAll();
        if (hw.vulkan.available) {
            // GPU run
            engine->unloadModel();
            ModelDescriptor gpuDesc = makeDescriptor(model_path, false);
            AGR_CHECK(engine->loadModel(gpuDesc));
            ExecutionContext gpuCtx = makeCtx(false);
            ExecutionResult gpuR = engine->execute(gpuCtx);
            AGR_CHECK(gpuR.success);
            AGR_CHECK(gpuR.telemetry.backend_type == ExecutionBackendType::VULKAN);
            AGR_CHECK(gpuR.telemetry.gpu_used);

            // cpu backend_type != gpu backend_type
            AGR_CHECK(cpuR.telemetry.backend_type != gpuR.telemetry.backend_type);
            std::cout << "  ok: mode distinction preserved through IExecutionEngine\n";
        }
    }

    // ----------------------------------------------------------------
    // 5. Cancellation
    // ----------------------------------------------------------------
    {
        engine->unloadModel();
        ModelDescriptor desc = makeDescriptor(model_path, true);
        AGR_CHECK(engine->loadModel(desc));

        std::atomic<bool> cancel{false};
        ExecutionContext ctx = makeCtx(true);
        ctx.max_tokens  = 256;
        ctx.cancel_flag = &cancel;

        // Cancel immediately
        cancel.store(true);
        ExecutionResult r = engine->execute(ctx);
        // Result may succeed with 0 tokens or fail with cancellation error;
        // the important thing is it does NOT hang.
        // (Engine may generate a couple tokens before observing the flag.)
        std::cout << "  Cancelled result: success=" << r.success
                  << " output='" << r.output << "' err='" << r.error << "'\n";
        std::cout << "  ok: cancellation did not hang\n";
    }

    // ----------------------------------------------------------------
    // 6. Streaming callback
    // ----------------------------------------------------------------
    {
        engine->unloadModel();
        ModelDescriptor desc = makeDescriptor(model_path, true);
        AGR_CHECK(engine->loadModel(desc));

        ExecutionContext ctx = makeCtx(true);
        ctx.max_tokens = 16;
        ctx.streaming  = true;

        int token_callbacks = 0;
        ExecutionResult r = engine->executeStreaming(ctx, [&](const std::string& piece) {
            ++token_callbacks;
            (void)piece;
        });

        std::cout << "  Streaming tokens received: " << token_callbacks << "\n";
        AGR_CHECK(r.success);
        AGR_CHECK(!r.output.empty());
        // Streaming simulates per-token via generateStreaming; at least 1 callback
        AGR_CHECK(token_callbacks > 0);
        std::cout << "  ok: streaming callback through IExecutionEngine\n";
    }

    engine->shutdown();
    std::cout << "PASS\n";
    return 0;
}
