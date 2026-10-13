// tests/test_p17_engine_lifecycle.cpp
// Phase 17: HydroXLEngine IExecutionEngine lifecycle tests.
// Does NOT require the real model file.

#include "mini_test.h"
#include "hydroxl/HydroXLEngine.h"
#include "engine/ExecutionEngineRegistry.h"
#include "engine/ExecutionContext.h"
#include <iostream>

using namespace agr;

int main() {
    // Identity
    {   HydroXLEngine eng;
        auto id = eng.identity();
        AGR_CHECK(id.name         == "HydroXL");
        AGR_CHECK(id.registry_key == "hydroxl");
        AGR_CHECK(!id.version.empty());
    }

    // Capabilities: claims CPU, streaming, cancellation; NOT Vulkan/CUDA
    {   HydroXLEngine eng;
        auto caps = eng.capabilities();
        AGR_CHECK( caps.has(EngineCapability::TEXT_GENERATION));
        AGR_CHECK( caps.has(EngineCapability::BACKEND_CPU));
        AGR_CHECK( caps.has(EngineCapability::STREAMING));
        AGR_CHECK( caps.has(EngineCapability::CANCELLATION));
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_CUDA));
        AGR_CHECK(!caps.has(EngineCapability::BACKEND_VULKAN));
    }

    // initialize/shutdown lifecycle
    {   HydroXLEngine eng;
        AGR_CHECK(!eng.isInitialized());
        AGR_CHECK(eng.initialize());
        AGR_CHECK(eng.isInitialized());
        AGR_CHECK(eng.initialize()); // idempotent
        eng.shutdown();
        AGR_CHECK(!eng.isInitialized());
    }

    // Model not loaded initially
    {   HydroXLEngine eng;
        eng.initialize();
        AGR_CHECK(!eng.isModelLoaded());
    }

    // Load nonexistent model -> failure
    {   HydroXLEngine eng;
        eng.initialize();
        ModelDescriptor desc;
        desc.model_id    = "x";
        desc.source_path = "no_such_file.gguf";
        AGR_CHECK(!eng.loadModel(desc));
        AGR_CHECK(!eng.isModelLoaded());
        auto err = eng.lastEngineError();
        AGR_CHECK(!err.ok());
        AGR_CHECK(err.kind == EngineErrorKind::MODEL_LOAD_FAILED ||
                  err.kind == EngineErrorKind::MODEL_NOT_FOUND);
    }

    // Load without init -> ENGINE_UNAVAILABLE
    {   HydroXLEngine eng;
        ModelDescriptor desc; desc.model_id = "x"; desc.source_path = "x.gguf";
        AGR_CHECK(!eng.loadModel(desc));
        AGR_CHECK(eng.lastEngineError().kind == EngineErrorKind::ENGINE_UNAVAILABLE);
    }

    // Execute without model -> failure
    {   HydroXLEngine eng;
        eng.initialize();
        ExecutionContext ctx; ctx.prompt = "Hello";
        auto result = eng.execute(ctx);
        AGR_CHECK(!result.success);
        AGR_CHECK(!result.error.empty());
    }

    // Invalid descriptor -> INVALID_REQUEST
    {   HydroXLEngine eng;
        eng.initialize();
        ModelDescriptor desc; // empty
        AGR_CHECK(!eng.loadModel(desc));
        AGR_CHECK(eng.lastEngineError().kind == EngineErrorKind::INVALID_REQUEST);
    }

    // Registry registration and create()
    {   HydroXLEngine::registerInRegistry();
        auto& reg = ExecutionEngineRegistry::instance();
        AGR_CHECK(reg.isRegistered("hydroxl"));
        auto eng = reg.create("hydroxl");
        AGR_CHECK(eng != nullptr);
        AGR_CHECK(eng->identity().name == "HydroXL");
    }

    // unload without model is safe
    {   HydroXLEngine eng;
        eng.initialize();
        eng.unloadModel();
        AGR_CHECK(!eng.isModelLoaded());
    }

    // currentModel() empty before load
    {   HydroXLEngine eng;
        eng.initialize();
        AGR_CHECK(!eng.currentModel().isValid());
    }

    AGR_TEST_MAIN_END()
}
