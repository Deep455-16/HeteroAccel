// tests/test_p23_sdk_infer.cpp
// Phase 23: Inference Execution, Result Struct, and Streaming Tests

#define _CRT_SECURE_NO_WARNINGS
#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <vector>

struct StreamData {
    int token_count = 0;
    std::string collected_text;
    bool should_cancel_at_3 = false;
};

static int stream_callback(const char* token_text, int token_id, void* user_data) {
    (void)token_id;
    auto* data = static_cast<StreamData*>(user_data);
    if (!data) return 0;

    data->token_count++;
    if (token_text) {
        data->collected_text += token_text;
    }
    if (data->should_cancel_at_3 && data->token_count >= 3) {
        return 1; // cancel
    }
    return 0; // continue
}

int main() {
    std::cout << "--- Phase 23: SDK Inference & Streaming Tests ---\n";

    // Test 1: Null safety and invalid parameter checks
    HaError err = HA_OK;
    HaSession s_null = ha_session_create(nullptr, &err);
    AGR_CHECK(s_null == nullptr);
    AGR_CHECK(err == HA_ERR_INVALID_ARG);

    ha_session_destroy(nullptr);

    HaInferParams params = {};
    params.max_new_tokens = 4;
    HaResult res_null_sess = ha_infer(nullptr, &params);
    AGR_CHECK(res_null_sess.error == HA_ERR_INVALID_ARG);

    // Requires model
    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cout << "[SKIP] HETEROACCEL_MODEL_PATH not set or file does not exist: " << model_path << "\n";
        return 77;
    }

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    HaModel model = ha_model_load(rt, model_path.c_str(), &err);
    AGR_CHECK(model != nullptr);
    AGR_CHECK(err == HA_OK);

    // Test 2: Create session
    HaSession session = ha_session_create(model, &err);
    AGR_CHECK(session != nullptr);
    AGR_CHECK(err == HA_OK);

    // Test 3: Infer with NULL params on valid session
    HaResult res_null_params = ha_infer(session, nullptr);
    AGR_CHECK(res_null_params.error == HA_ERR_INVALID_ARG);

    // Test 4: Synchronous inference with text prompt
    params.prompt = "The capital of France is";
    params.token_ids = nullptr;
    params.n_token_ids = 0;
    params.max_new_tokens = 6;
    params.temperature = 0.0f;
    params.top_p = 0.95f;

    HaResult res1 = ha_infer(session, &params);
    AGR_CHECK(res1.error == HA_OK);
    AGR_CHECK(res1.output_text != nullptr);
    AGR_CHECK(std::strlen(res1.output_text) > 0);
    AGR_CHECK(res1.n_output_tokens > 0);
    AGR_CHECK(res1.output_token_ids != nullptr);
    AGR_CHECK(res1.n_prompt_tokens > 0);
    AGR_CHECK(res1.tokens_per_second >= 0.0f);
    AGR_CHECK(res1.backend_used != nullptr && std::strlen(res1.backend_used) > 0);

    std::cout << "  Prompt output: '" << res1.output_text << "' ("
              << res1.n_output_tokens << " tokens, "
              << res1.tokens_per_second << " tok/s, backend="
              << res1.backend_used << ")\n";

    // Test 5: Synchronous inference with token IDs
    int sample_tokens[] = { 151644, 872, 198 }; // sample tokens
    HaInferParams token_params = {};
    token_params.prompt = nullptr;
    token_params.token_ids = sample_tokens;
    token_params.n_token_ids = 3;
    token_params.max_new_tokens = 4;
    token_params.temperature = 0.0f;

    HaResult res2 = ha_infer(session, &token_params);
    AGR_CHECK(res2.error == HA_OK);
    AGR_CHECK(res2.n_output_tokens > 0);
    AGR_CHECK(res2.output_text != nullptr);

    // Test 6: Streaming simulation callback
    StreamData stream_data;
    HaError stream_err = ha_infer_stream(session, &params, stream_callback, &stream_data);
    AGR_CHECK(stream_err == HA_OK);
    AGR_CHECK(stream_data.token_count > 0);
    AGR_CHECK(!stream_data.collected_text.empty());

    // Test 7: Streaming cancellation
    StreamData cancel_data;
    cancel_data.should_cancel_at_3 = true;
    HaError cancel_err = ha_infer_stream(session, &params, stream_callback, &cancel_data);
    AGR_CHECK(cancel_err == HA_ERR_CANCELLED);
    AGR_CHECK(cancel_data.token_count >= 3);

    // Test 8: C++ SDK Session inference and streaming
    {
        ha::Runtime cpp_rt;
        ha::Model cpp_model = cpp_rt.load_model(model_path);
        ha::Session cpp_session = cpp_model.create_session();

        ha::InferParams cpp_params;
        cpp_params.prompt = "Paris is";
        cpp_params.max_new_tokens = 4;

        ha::Result cpp_res = cpp_session.infer(cpp_params);
        AGR_CHECK(cpp_res.ok());
        AGR_CHECK(!cpp_res.output_text.empty());
        AGR_CHECK(cpp_res.n_output_tokens > 0);
        AGR_CHECK(cpp_res.output_token_ids.size() == static_cast<size_t>(cpp_res.n_output_tokens));

        int cpp_stream_count = 0;
        HaError c_err = cpp_session.infer_stream(cpp_params, [&](const std::string& txt, int tid) {
            (void)txt;
            (void)tid;
            cpp_stream_count++;
            return true;
        });
        AGR_CHECK(c_err == HA_OK);
        AGR_CHECK(cpp_stream_count > 0);
    }

    ha_session_destroy(session);
    ha_model_unload(model);
    ha_runtime_destroy(rt);

    AGR_TEST_MAIN_END();
}
