// tests/test_p23_sdk_concurrent.cpp
// Phase 23: Concurrent Multi-Session SDK Tests

#define _CRT_SECURE_NO_WARNINGS
#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
    std::cout << "--- Phase 23: Concurrent SDK Sessions Test ---\n";

    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cout << "[SKIP] HETEROACCEL_MODEL_PATH not set or file does not exist: " << model_path << "\n";
        return 77;
    }

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    HaError err = HA_OK;
    HaModel model = ha_model_load(rt, model_path.c_str(), &err);
    AGR_CHECK(model != nullptr);
    AGR_CHECK(err == HA_OK);

    // Create two independent sessions
    HaSession session1 = ha_session_create(model, &err);
    AGR_CHECK(session1 != nullptr);
    AGR_CHECK(err == HA_OK);

    HaSession session2 = ha_session_create(model, &err);
    AGR_CHECK(session2 != nullptr);
    AGR_CHECK(err == HA_OK);

    // Run inference concurrently on separate threads
    HaResult res1 = {};
    HaResult res2 = {};

    std::thread t1([session1, &res1]() {
        HaInferParams params = {};
        params.prompt = "The sun rises in the";
        params.max_new_tokens = 4;
        params.temperature = 0.0f;
        res1 = ha_infer(session1, &params);
    });

    std::thread t2([session2, &res2]() {
        HaInferParams params = {};
        params.prompt = "Water boils at";
        params.max_new_tokens = 4;
        params.temperature = 0.0f;
        res2 = ha_infer(session2, &params);
    });

    t1.join();
    t2.join();

    std::cout << "  Thread 1 output: '" << res1.output_text << "' (backend=" << res1.backend_used << ")\n";
    std::cout << "  Thread 2 output: '" << res2.output_text << "' (backend=" << res2.backend_used << ")\n";

    AGR_CHECK(res1.error == HA_OK);
    AGR_CHECK(res1.n_output_tokens > 0);
    AGR_CHECK(res1.output_text != nullptr);

    AGR_CHECK(res2.error == HA_OK);
    AGR_CHECK(res2.n_output_tokens > 0);
    AGR_CHECK(res2.output_text != nullptr);

    ha_session_destroy(session1);
    ha_session_destroy(session2);
    ha_model_unload(model);
    ha_runtime_destroy(rt);

    AGR_TEST_MAIN_END();
}
