// test_inference_backend_init.cpp - Phase 7: LlamaCppBackend lifecycle without a model
#include "mini_test.h"
#include "inference/LlamaCppBackend.h"

int main() {
    std::cout << "== test_inference_backend_init ==\n";

    agr::LlamaCppBackend backend;

    // Fresh backend must be DISCOVERED
    AGR_CHECK(backend.state() == agr::ModelState::DISCOVERED);
    AGR_CHECK(backend.backendName() == "LlamaCppBackend");

    // loadModel with empty path must fail gracefully (not crash)
    bool ok = backend.loadModel("", 0, true);
    AGR_CHECK(!ok);
    AGR_CHECK(backend.state() == agr::ModelState::FAILED);
    AGR_CHECK(!backend.lastError().empty());

    // unload on FAILED state is safe
    backend.unload();
    AGR_CHECK(backend.state() == agr::ModelState::RELEASED);

    AGR_TEST_MAIN_END();
}
