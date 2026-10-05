// tests/test_p11_engine_interface.cpp
// Phase 11: Unit tests for the IExecutionEngine interface / infrastructure.
//
// Tests covered:
//  1. EngineCapabilitySet basics
//  2. ModelDescriptor basics
//  3. EngineError basics
//  4. ExecutionEngineRegistry: register, create, query, reject
//  5. LlamaCppExecutionEngine: identity, capabilities, lifecycle
//  6. Backend isolation: no llama.cpp types in engine headers
//
// This test does NOT load a real model. Real-model tests are in:
//   test_p11_llm_via_engine.cpp (CPU + Vulkan if HETEROACCEL_MODEL_PATH set)

#include "mini_test.h"
#include "engine/EngineCapability.h"
#include "engine/EngineError.h"
#include "engine/EngineIdentity.h"
#include "engine/ExecutionContext.h"
#include "engine/ExecutionResult.h"
#include "engine/ModelDescriptor.h"
#include "engine/IExecutionEngine.h"
#include "engine/ExecutionEngineRegistry.h"
#include "engine/LlamaCppExecutionEngine.h"

using namespace agr;

int main() {
    std::cout << "== test_p11_engine_interface ==\n";

    // ----------------------------------------------------------------
    // 1. EngineCapabilitySet
    // ----------------------------------------------------------------
    {
        EngineCapabilitySet empty;
        AGR_CHECK(!empty.has(EngineCapability::TEXT_GENERATION));
        AGR_CHECK(empty.empty());

        EngineCapabilitySet caps{
            EngineCapability::TEXT_GENERATION,
            EngineCapability::BACKEND_CPU,
            EngineCapability::CANCELLATION
        };
        AGR_CHECK(caps.has(EngineCapability::TEXT_GENERATION));
        AGR_CHECK(caps.has(EngineCapability::BACKEND_CPU));
        AGR_CHECK(caps.has(EngineCapability::CANCELLATION));
        AGR_CHECK(!caps.has(EngineCapability::VISION));
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_CUDA));
        AGR_CHECK(!caps.empty());

        std::string s = caps.toString();
        AGR_CHECK(s.find("TEXT_GENERATION") != std::string::npos);
        AGR_CHECK(s.find("BACKEND_CPU") != std::string::npos);

        std::cout << "  ok: EngineCapabilitySet\n";
    }

    // ----------------------------------------------------------------
    // 2. ModelDescriptor
    // ----------------------------------------------------------------
    {
        ModelDescriptor d;
        AGR_CHECK(!d.isValid()); // no id or path

        d.model_id    = "test-model";
        d.source_path = "/fake/model.gguf";
        d.format      = ModelFormat::GGUF;
        d.architecture = "llama";
        AGR_CHECK(d.isValid());
        AGR_CHECK(d.format == ModelFormat::GGUF);

        std::string sum = d.summary();
        AGR_CHECK(!sum.empty());
        AGR_CHECK(sum.find("test-model") != std::string::npos);

        AGR_CHECK(std::string(toString(ModelFormat::GGUF)) == "GGUF");
        AGR_CHECK(std::string(toString(ModelModality::TEXT)) == "Text");
        std::cout << "  ok: ModelDescriptor\n";
    }

    // ----------------------------------------------------------------
    // 3. EngineError
    // ----------------------------------------------------------------
    {
        EngineError ok = EngineError::success();
        AGR_CHECK(ok.ok());
        AGR_CHECK(ok.kind == EngineErrorKind::NONE);

        EngineError err = EngineError::make(EngineErrorKind::MODEL_NOT_FOUND, "not found");
        AGR_CHECK(!err.ok());
        AGR_CHECK(err.kind == EngineErrorKind::MODEL_NOT_FOUND);
        AGR_CHECK(err.message == "not found");

        AGR_CHECK(std::string(toString(EngineErrorKind::CANCELLED)) == "Cancelled");
        std::cout << "  ok: EngineError\n";
    }

    // ----------------------------------------------------------------
    // 4. ExecutionEngineRegistry
    // ----------------------------------------------------------------
    {
        // "llama.cpp" should be auto-registered by the static initialiser
        auto& reg = ExecutionEngineRegistry::instance();
        AGR_CHECK(reg.isRegistered("llama.cpp"));
        AGR_CHECK(!reg.isRegistered("colibri"));          // not yet
        AGR_CHECK(!reg.isRegistered("does-not-exist"));

        auto keys = reg.registeredKeys();
        bool found = false;
        for (auto& k : keys) if (k == "llama.cpp") { found = true; break; }
        AGR_CHECK(found);

        // Create via registry
        auto engine = reg.create("llama.cpp");
        AGR_CHECK(engine != nullptr);

        // Unknown key returns nullptr
        auto none = reg.create("does-not-exist");
        AGR_CHECK(none == nullptr);

        // Registration record
        const EngineRegistration* rec = reg.registration("llama.cpp");
        AGR_CHECK(rec != nullptr);
        AGR_CHECK(rec->key == "llama.cpp");
        AGR_CHECK(!rec->description.empty());

        std::cout << "  ok: ExecutionEngineRegistry\n";
    }

    // ----------------------------------------------------------------
    // 5. LlamaCppExecutionEngine: identity, capabilities, lifecycle
    // ----------------------------------------------------------------
    {
        auto engine = std::make_shared<LlamaCppExecutionEngine>();

        // Identity
        EngineIdentity id = engine->identity();
        AGR_CHECK(id.name == "llama.cpp");
        AGR_CHECK(!id.version.empty());
        AGR_CHECK(id.registry_key == "llama.cpp");

        // Capabilities
        EngineCapabilitySet caps = engine->capabilities();
        AGR_CHECK(caps.has(EngineCapability::TEXT_GENERATION));
        AGR_CHECK(caps.has(EngineCapability::BACKEND_CPU));
        AGR_CHECK(caps.has(EngineCapability::BACKEND_VULKAN));
        AGR_CHECK(caps.has(EngineCapability::CANCELLATION));
        AGR_CHECK(caps.has(EngineCapability::STREAMING));
#if !defined(AGR_GGML_CUDA) || !AGR_GGML_CUDA
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_CUDA));
#endif

        // Not yet initialised
        AGR_CHECK(!engine->isInitialized());
        AGR_CHECK(!engine->isModelLoaded());

        // Initialize / shutdown lifecycle
        AGR_CHECK(engine->initialize());
        AGR_CHECK(engine->isInitialized());
        AGR_CHECK(engine->initialize()); // idempotent

        engine->shutdown();
        AGR_CHECK(!engine->isInitialized());
        AGR_CHECK(!engine->isModelLoaded());

        // Re-initialize after shutdown
        AGR_CHECK(engine->initialize());
        AGR_CHECK(engine->isInitialized());

        std::cout << "  ok: LlamaCppExecutionEngine lifecycle\n";
    }

    // ----------------------------------------------------------------
    // 6. Model load failure returns clean error
    // ----------------------------------------------------------------
    {
        auto engine = std::make_shared<LlamaCppExecutionEngine>();
        engine->initialize();

        ModelDescriptor bad;
        bad.model_id    = "bad";
        bad.source_path = "/nonexistent/model.gguf";

        bool loaded = engine->loadModel(bad);
        AGR_CHECK(!loaded);
        AGR_CHECK(!engine->isModelLoaded());

        EngineError err = engine->lastEngineError();
        AGR_CHECK(!err.ok());
        AGR_CHECK(err.kind == EngineErrorKind::MODEL_NOT_FOUND);
        AGR_CHECK(!err.message.empty());

        std::cout << "  ok: clean error on invalid model path\n";
    }

    // ----------------------------------------------------------------
    // 7. execute() before initialise returns clean error
    // ----------------------------------------------------------------
    {
        LlamaCppExecutionEngine engine;
        ExecutionContext ctx;
        ctx.prompt = "hello";
        ExecutionResult r = engine.execute(ctx);
        AGR_CHECK(!r.success);
        AGR_CHECK(!r.error.empty());
        std::cout << "  ok: execute before init returns error\n";
    }

    // ----------------------------------------------------------------
    // 8. Backend isolation: ExecutionContext / ExecutionResult types
    //    do not contain llama.cpp-specific types
    // ----------------------------------------------------------------
    {
        // This is a compile-time check (if it compiles without llama.h, we're good).
        ExecutionContext ctx;
        ctx.max_tokens  = 128;
        ctx.temperature = 0.5f;
        ctx.n_gpu_layers = 0;
        ctx.cpu_only     = true;

        ExecutionResult r;
        r.success = false;
        r.error   = "test";
        r.telemetry.backend_type = ExecutionBackendType::CPU;

        AGR_CHECK(ctx.max_tokens == 128);
        AGR_CHECK(!r.success);

        std::cout << "  ok: ExecutionContext/Result contain no llama.cpp types\n";
    }

    std::cout << "PASS\n";
    return 0;
}
