// tests/test_p24_streaming.cpp
// Phase 24: Native Token Streaming and Mid-Generation Cancellation Tests

#define _CRT_SECURE_NO_WARNINGS
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

struct StreamUserData {
    int callback_count = 0;
    std::string accumulated;
    std::vector<int> token_ids;
    int max_tokens_to_allow = 999;
};

// Return 0 to continue, non-zero to cancel
static int stream_callback(const char* piece, int token_id, void* user_data) {
    if (!user_data) return 0;
    auto* d = static_cast<StreamUserData*>(user_data);
    d->callback_count++;
    if (piece) {
        d->accumulated += piece;
    }
    d->token_ids.push_back(token_id);

    if (d->callback_count >= d->max_tokens_to_allow) {
        return 1; // Signal cancellation
    }
    return 0; // Continue streaming
}

int main() {
    std::cout << "--- Phase 24: Native Streaming & Cancellation Tests ---\n";

    // Test 1: Null safety and invalid parameter checks
    HaInferParams params = {};
    params.prompt = "Test prompt";
    params.max_new_tokens = 5;

    AGR_CHECK(ha_infer_stream(nullptr, &params, stream_callback, nullptr) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_session_cancel(nullptr) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_infer_cancel(nullptr) == HA_ERR_INVALID_ARG);

    // Locate test model
    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::string fallback = "D:/HeteroAccel/Models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf";
        if (std::filesystem::exists(fallback)) {
            model_path = fallback;
        }
    }

    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cout << "[SKIP] No model found at HETEROACCEL_MODEL_PATH or default path: " << model_path << "\n";
        AGR_TEST_MAIN_END();
    }

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    HaError err = HA_OK;
    HaModel model = ha_model_load(rt, model_path.c_str(), &err);
    AGR_CHECK(model != nullptr);
    AGR_CHECK(err == HA_OK);

    HaSession session = ha_session_create(model, &err);
    AGR_CHECK(session != nullptr);
    AGR_CHECK(err == HA_OK);

    // More invalid arg checks with valid session
    AGR_CHECK(ha_infer_stream(session, nullptr, stream_callback, nullptr) == HA_ERR_INVALID_ARG);
    AGR_CHECK(ha_infer_stream(session, &params, nullptr, nullptr) == HA_ERR_INVALID_ARG);

    // Test 2: Full streaming inference (no premature cancellation)
    {
        StreamUserData data;
        data.max_tokens_to_allow = 100; // Allow all tokens up to max_new_tokens

        HaInferParams st_params = {};
        st_params.prompt = "Write a haiku about code";
        st_params.max_new_tokens = 8;
        st_params.temperature = 0.7f;
        st_params.top_p = 0.9f;

        auto t0 = std::chrono::steady_clock::now();
        HaError res_err = ha_infer_stream(session, &st_params, stream_callback, &data);
        auto t1 = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        AGR_CHECK(res_err == HA_OK);
        AGR_CHECK(data.callback_count > 0);
        AGR_CHECK(data.callback_count == static_cast<int>(data.token_ids.size()));
        AGR_CHECK(!data.accumulated.empty());

        std::cout << "  Streaming completed: " << data.callback_count << " tokens in "
                  << elapsed_ms << " ms\n";
        std::cout << "  Generated: \"" << data.accumulated << "\"\n";
    }

    // Test 3: Early cancellation via callback return value
    {
        StreamUserData cancel_data;
        cancel_data.max_tokens_to_allow = 3; // Cancel at token 3

        HaInferParams cancel_params = {};
        cancel_params.prompt = "Explain quantum computing in detail";
        cancel_params.max_new_tokens = 20;

        HaError res_err = ha_infer_stream(session, &cancel_params, stream_callback, &cancel_data);
        // Returns HA_ERR_CANCELLED when aborted early by callback
        AGR_CHECK(res_err == HA_ERR_CANCELLED);
        AGR_CHECK(cancel_data.callback_count == 3);
        std::cout << "  Early callback cancellation verified: stopped at "
                  << cancel_data.callback_count << " tokens (requested 20)\n";
    }

    // Test 4: Pre-cancellation via ha_session_cancel
    {
        AGR_CHECK(ha_session_cancel(session) == HA_OK);

        StreamUserData pre_data;
        pre_data.max_tokens_to_allow = 10;
        HaInferParams pre_params = {};
        pre_params.prompt = "Another test";
        pre_params.max_new_tokens = 10;

        HaError res_err = ha_infer_stream(session, &pre_params, stream_callback, &pre_data);
        AGR_CHECK(res_err == HA_ERR_CANCELLED);
        AGR_CHECK(pre_data.callback_count == 0);
        std::cout << "  Pre-cancellation verified: immediately halted\n";
    }

    // Test 5: C++ wrapper streaming and lambda callback
    {
        ha::Session cpp_session(session);

        int count = 0;
        std::string text;
        auto cpp_res = cpp_session.infer_stream(
            "Counting: 1 2 3",
            [&count, &text](const std::string& piece, int /*tid*/) -> bool {
                count++;
                text += piece;
                return true;
            },
            6
        );

        AGR_CHECK(cpp_res.error == HA_OK);
        AGR_CHECK(count > 0);
        AGR_CHECK(!text.empty());
        std::cout << "  C++ streaming verified: " << count << " tokens\n";

        // C++ cancellation test
        int cancel_count = 0;
        auto cancel_res = cpp_session.infer_stream(
            "Long prompt for cancellation test",
            [&cancel_count](const std::string& /*piece*/, int /*tid*/) -> bool {
                cancel_count++;
                return cancel_count < 2; // Cancel after token 2
            },
            15
        );
        AGR_CHECK(cancel_res.error == HA_ERR_CANCELLED);
        AGR_CHECK(cancel_count == 2);
        std::cout << "  C++ lambda cancellation verified: stopped at " << cancel_count << " tokens\n";

        cpp_session.release(); // Keep underlying C session alive for ha_session_destroy
    }

    // Test 6: ha_infer_cancel compatibility function
    {
        AGR_CHECK(ha_infer_cancel(session) == HA_OK);
    }

    ha_session_destroy(session);
    ha_model_unload(model);
    ha_runtime_destroy(rt);

    AGR_TEST_MAIN_END();
}
