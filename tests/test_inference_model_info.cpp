// test_inference_model_info.cpp - Phase 7: ModelInfo from backend (no real model needed)
#include "mini_test.h"
#include "inference/LlamaCppBackend.h"
#include "inference/InferenceTypes.h"

int main() {
    std::cout << "== test_inference_model_info ==\n";

    agr::LlamaCppBackend backend;

    // modelInfo() on unloaded backend should return empty/default struct
    agr::ModelInfo info = backend.modelInfo();
    AGR_CHECK(info.path.empty());
    AGR_CHECK(info.size_bytes == 0);

    // Attempt load with bad path — returns FAILED state
    bool ok = backend.loadModel("/nonexistent/model.gguf", 0, true);
    AGR_CHECK(!ok);
    AGR_CHECK(backend.state() == agr::ModelState::FAILED);

    // generate() on FAILED state must return error, not crash
    agr::InferenceRequest req;
    req.prompt = "test";
    req.options.max_tokens = 1;
    agr::InferenceResult res = backend.generate(req);
    AGR_CHECK(!res.success);
    AGR_CHECK(!res.error.empty());

    AGR_TEST_MAIN_END();
}
