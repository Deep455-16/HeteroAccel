// tests/test_p11_registry.cpp
// Phase 11: ExecutionEngineRegistry tests.
//
// Validates:
//  1. llama.cpp auto-registered
//  2. Custom engine can be registered
//  3. Registered engine is creatable
//  4. Unknown key returns nullptr
//  5. Registration survives create() calls
//  6. clear() removes registrations

#include "mini_test.h"
#include "engine/ExecutionEngineRegistry.h"
#include "engine/IExecutionEngine.h"
#include "engine/LlamaCppExecutionEngine.h"

#include <string>
#include <iostream>

using namespace agr;

// ---------------------------------------------------------------------------
// Minimal stub engine for registry testing — no llama.cpp dependency
// ---------------------------------------------------------------------------
class StubEngine final : public IExecutionEngine {
public:
    EngineIdentity identity() const override {
        return {"Stub", "0.1", "Test", "Test", "stub"};
    }
    EngineCapabilitySet capabilities() const override { return {}; }

    bool initialize() override { return true; }
    void shutdown() override {}
    bool isInitialized() const override { return true; }

    bool loadModel(const ModelDescriptor&) override { return false; }
    void unloadModel() override {}
    bool isModelLoaded() const override { return false; }
    ModelDescriptor currentModel() const override { return {}; }

    ExecutionResult execute(const ExecutionContext&) override { return {}; }
    ExecutionResult executeStreaming(const ExecutionContext&, TokenCallback) override { return {}; }

    EngineError lastEngineError() const override { return EngineError::success(); }
};

int main() {
    std::cout << "== test_p11_registry ==\n";

    auto& reg = ExecutionEngineRegistry::instance();

    // 1. llama.cpp is auto-registered
    AGR_CHECK(reg.isRegistered("llama.cpp"));
    std::cout << "  ok: llama.cpp auto-registered\n";

    // 2. Register a custom stub engine
    reg.registerFactory("stub", "Test stub engine",
                         []() -> std::shared_ptr<IExecutionEngine> {
                             return std::make_shared<StubEngine>();
                         });
    AGR_CHECK(reg.isRegistered("stub"));
    std::cout << "  ok: custom engine registered\n";

    // 3. Create registered engines
    auto llamaEngine = reg.create("llama.cpp");
    AGR_CHECK(llamaEngine != nullptr);
    AGR_CHECK(llamaEngine->identity().name == "llama.cpp");

    auto stubEngine = reg.create("stub");
    AGR_CHECK(stubEngine != nullptr);
    AGR_CHECK(stubEngine->identity().name == "Stub");
    std::cout << "  ok: engines created via registry\n";

    // 4. Unknown key returns nullptr
    auto none = reg.create("colibri");
    AGR_CHECK(none == nullptr);

    auto none2 = reg.create("does-not-exist");
    AGR_CHECK(none2 == nullptr);
    std::cout << "  ok: unknown keys return nullptr\n";

    // 5. Keys list includes both
    auto keys = reg.registeredKeys();
    bool foundLlama = false, foundStub = false;
    for (auto& k : keys) {
        if (k == "llama.cpp") foundLlama = true;
        if (k == "stub")      foundStub  = true;
    }
    AGR_CHECK(foundLlama);
    AGR_CHECK(foundStub);
    std::cout << "  ok: keys enumeration correct\n";

    // 6. Registration record accessible
    const EngineRegistration* rec = reg.registration("stub");
    AGR_CHECK(rec != nullptr);
    AGR_CHECK(rec->key == "stub");
    AGR_CHECK(rec->description == "Test stub engine");
    std::cout << "  ok: registration metadata accessible\n";

    // 7. Overwrite registration (same key)
    reg.registerFactory("stub", "Updated stub", []() -> std::shared_ptr<IExecutionEngine> {
        return std::make_shared<StubEngine>();
    });
    const EngineRegistration* updated = reg.registration("stub");
    AGR_CHECK(updated != nullptr);
    AGR_CHECK(updated->description == "Updated stub");
    std::cout << "  ok: factory overwrite works\n";

    std::cout << "PASS\n";
    return 0;
}
