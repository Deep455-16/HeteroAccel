// test_inference_invalid_model.cpp - Phase 7: Graceful handling of missing/invalid model
#include "mini_test.h"
#include "inference/HeteroRuntime.h"

int main() {
    std::cout << "== test_inference_invalid_model ==\n";

    agr::HeteroRuntime runtime;
    bool ok = runtime.initialize();
    AGR_CHECK(ok);

    // Try to generate with non-existent model
    agr::GenerationOptions opts;
    opts.max_tokens = 1;
    agr::InferenceResult result = runtime.generate(
        "nonexistent/model.gguf",
        "Hello",
        opts
    );

    // Must fail gracefully — no crash, no exception
    AGR_CHECK(!result.success);
    AGR_CHECK(!result.error.empty());

    // Empty prompt must also fail gracefully
    agr::InferenceResult result2 = runtime.generate(
        "some/model.gguf",
        "",
        opts
    );
    AGR_CHECK(!result2.success);
    AGR_CHECK(!result2.error.empty());

    runtime.shutdown();
    AGR_TEST_MAIN_END();
}
