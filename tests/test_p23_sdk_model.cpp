// tests/test_p23_sdk_model.cpp
// Phase 23: Model Loading, Metadata Inspection, and Unload Tests

#define _CRT_SECURE_NO_WARNINGS
#include "mini_test.h"
#include <heteroaccel/heteroaccel.h>
#include <heteroaccel/HeteroAccel.hpp>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

int main() {
    std::cout << "--- Phase 23: SDK Model Lifecycle & Metadata Tests ---\n";

    HaRuntime rt = ha_runtime_create();
    AGR_CHECK(rt != nullptr);

    // Test 1: Invalid arguments
    HaError err = HA_OK;
    HaModel m_null_rt = ha_model_load(nullptr, "dummy.gguf", &err);
    AGR_CHECK(m_null_rt == nullptr);
    AGR_CHECK(err == HA_ERR_INVALID_ARG);

    HaModel m_null_path = ha_model_load(rt, nullptr, &err);
    AGR_CHECK(m_null_path == nullptr);
    AGR_CHECK(err == HA_ERR_INVALID_ARG);

    HaModel m_empty_path = ha_model_load(rt, "", &err);
    AGR_CHECK(m_empty_path == nullptr);
    AGR_CHECK(err == HA_ERR_INVALID_ARG);

    // Test 2: Non-existent file
    HaModel m_missing = ha_model_load(rt, "this_file_does_not_exist_at_all.gguf", &err);
    AGR_CHECK(m_missing == nullptr);
    AGR_CHECK(err == HA_ERR_MODEL_LOAD);

    // Test 3: Null safety for metadata accessors and unload
    ha_model_unload(nullptr);
    AGR_CHECK(std::strlen(ha_model_arch(nullptr)) == 0);
    AGR_CHECK(ha_model_context_length(nullptr) == 0);
    AGR_CHECK(ha_model_vocab_size(nullptr) == 0);

    // Model-dependent tests require HETEROACCEL_MODEL_PATH
    const char* model_env = std::getenv("HETEROACCEL_MODEL_PATH");
    std::string model_path = model_env ? model_env : "";
    if (model_path.empty() || !std::filesystem::exists(model_path)) {
        std::cout << "[SKIP] HETEROACCEL_MODEL_PATH not set or file does not exist: " << model_path << "\n";
        ha_runtime_destroy(rt);
        return 77;
    }

    std::cout << "Loading real model: " << model_path << "\n";

    // Test 4: Real model load & metadata verification (C API)
    HaModel model = ha_model_load(rt, model_path.c_str(), &err);
    AGR_CHECK(err == HA_OK);
    AGR_CHECK(model != nullptr);

    const char* arch = ha_model_arch(model);
    AGR_CHECK(arch != nullptr);
    AGR_CHECK(std::strcmp(arch, "qwen2") == 0);

    uint32_t ctx_len = ha_model_context_length(model);
    AGR_CHECK(ctx_len > 0);
    std::cout << "  Model arch: " << arch << ", context_length: " << ctx_len << "\n";

    uint32_t vocab = ha_model_vocab_size(model);
    AGR_CHECK(vocab > 0);
    std::cout << "  Vocab size: " << vocab << "\n";

    ha_model_unload(model);

    // Test 5: C++ SDK Model wrapper
    {
        ha::Runtime cpp_rt;
        ha::Model cpp_model = cpp_rt.load_model(model_path);
        AGR_CHECK(cpp_model.handle() != nullptr);
        AGR_CHECK(cpp_model.arch() == "qwen2");
        AGR_CHECK(cpp_model.context_length() > 0);
        AGR_CHECK(cpp_model.vocab_size() > 0);

        // Test move semantics
        ha::Model moved_model(std::move(cpp_model));
        AGR_CHECK(moved_model.handle() != nullptr);
        AGR_CHECK(cpp_model.handle() == nullptr);
    }

    ha_runtime_destroy(rt);
    AGR_TEST_MAIN_END();
}
