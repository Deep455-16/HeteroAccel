// test_llm_model_path.cpp
//
// Verifies that LlamaCppEngine handles invalid model paths cleanly:
//   - empty path     -> initialize() returns false, error non-empty
//   - nonexistent    -> initialize() returns false, error non-empty
//   - no crash in any case
//
// Does NOT require a real GGUF model file. Always runs.

#include "mini_test.h"
#include "llm/LlamaCppEngine.h"
#include "llm/LlmConfig.h"

int main() {
    std::cout << "== test_llm_model_path ==\n";

    // --- Test 1: empty model path ---
    {
        agr::LlmConfig cfg;
        cfg.model_path = "";

        agr::LlamaCppEngine engine;
        bool ok = engine.initialize(cfg);
        std::cout << "  empty path -> initialize()=" << (ok ? "true (BUG)" : "false (correct)") << "\n";
        std::cout << "  error: " << engine.lastError() << "\n";
        AGR_CHECK(!ok);
        AGR_CHECK(!engine.lastError().empty());
        engine.shutdown();
    }

    // --- Test 2: nonexistent file ---
    {
        agr::LlmConfig cfg;
        cfg.model_path = "C:/nonexistent/path/totally_fake_model_xyz_12345.gguf";

        agr::LlamaCppEngine engine;
        bool ok = engine.initialize(cfg);
        std::cout << "  nonexistent -> initialize()=" << (ok ? "true (BUG)" : "false (correct)") << "\n";
        std::cout << "  error: " << engine.lastError() << "\n";
        AGR_CHECK(!ok);
        AGR_CHECK(!engine.lastError().empty());
        engine.shutdown();
    }

    // --- Test 3: calling infer() without initialize() ---
    {
        agr::LlamaCppEngine engine;
        agr::LlmResult r = engine.infer("Hello");
        std::cout << "  infer() without init -> success=" << (r.success ? "true (BUG)" : "false (correct)") << "\n";
        AGR_CHECK(!r.success);
        AGR_CHECK(!r.error.empty());
    }

    // --- Test 4: shutdown() is safe to call without initialize() ---
    {
        agr::LlamaCppEngine engine;
        engine.shutdown(); // must not crash
        engine.shutdown(); // must not crash on second call either
        AGR_CHECK(true);
        std::cout << "  double shutdown() without init: OK\n";
    }

    AGR_TEST_MAIN_END();
}
