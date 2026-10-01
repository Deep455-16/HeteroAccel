// test_inference_runtime_init.cpp - Phase 7: HeteroRuntime initialization
#include "mini_test.h"
#include "inference/HeteroRuntime.h"

int main() {
    std::cout << "== test_inference_runtime_init ==\n";

    agr::HeteroRuntime runtime;

    AGR_CHECK(!runtime.isInitialized());

    bool ok = runtime.initialize();
    AGR_CHECK(ok);
    AGR_CHECK(runtime.isInitialized());

    // Hardware must be detected
    AGR_CHECK(!runtime.hardware().cpu.model_name.empty());

    // Diagnostics must return non-empty string
    std::string diag = runtime.diagnosticsReport();
    AGR_CHECK(!diag.empty());
    AGR_CHECK(diag.find("HeteroAccel") != std::string::npos);

    runtime.shutdown();
    AGR_CHECK(!runtime.isInitialized());

    AGR_TEST_MAIN_END();
}
