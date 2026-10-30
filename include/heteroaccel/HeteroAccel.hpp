// include/heteroaccel/HeteroAccel.hpp
// HeteroAccel Phase 23: Header-only Modern C++ RAII SDK Wrapper
//
// Provides safe, idiomatic C++ RAII classes wrapping the C API.

#ifndef HETEROACCEL_HPP
#define HETEROACCEL_HPP

#include "heteroaccel/heteroaccel.h"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ha {

class Exception : public std::runtime_error {
public:
    Exception(HaError code, const std::string& msg)
        : std::runtime_error(msg.empty() ? ha_error_string(code) : msg),
          code_(code) {}

    HaError code() const noexcept { return code_; }

private:
    HaError code_;
};

struct BackendInfo {
    std::string backend_id;
    bool available = false;
    bool requires_vulkan = false;
    bool requires_cuda = false;
};

struct InferParams {
    std::string prompt;
    std::vector<int> token_ids;
    int max_new_tokens = 16;
    float temperature = 0.0f;
    float top_p = 0.95f;

    HaInferParams to_c_params() const {
        HaInferParams p;
        p.prompt = prompt.empty() ? nullptr : prompt.c_str();
        p.token_ids = token_ids.empty() ? nullptr : token_ids.data();
        p.n_token_ids = static_cast<int>(token_ids.size());
        p.max_new_tokens = max_new_tokens;
        p.temperature = temperature;
        p.top_p = top_p;
        return p;
    }
};

struct Result {
    std::string output_text;
    std::vector<int> output_token_ids;
    int n_output_tokens = 0;
    int n_prompt_tokens = 0;
    float tokens_per_second = 0.0f;
    bool gpu_used = false;
    std::string backend_used;
    HaError error = HA_OK;
    std::string error_message;

    bool ok() const noexcept { return error == HA_OK; }
};

class Session {
public:
    explicit Session(HaSession handle) : handle_(handle) {
        if (!handle_) {
            throw Exception(HA_ERR_INVALID_ARG, "Cannot wrap null HaSession handle");
        }
    }

    ~Session() {
        if (handle_) {
            ha_session_destroy(handle_);
            handle_ = nullptr;
        }
    }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    Session(Session&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Session& operator=(Session&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                ha_session_destroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    HaSession handle() const noexcept { return handle_; }

    Result infer(const InferParams& params) {
        if (!handle_) {
            throw Exception(HA_ERR_INVALID_ARG, "Session is not open");
        }
        HaInferParams c_params = params.to_c_params();
        HaResult c_res = ha_infer(handle_, &c_params);

        Result r;
        if (c_res.output_text) {
            r.output_text = c_res.output_text;
        }
        if (c_res.output_token_ids && c_res.n_output_tokens > 0) {
            r.output_token_ids.assign(c_res.output_token_ids,
                                      c_res.output_token_ids + c_res.n_output_tokens);
        }
        r.n_output_tokens = c_res.n_output_tokens;
        r.n_prompt_tokens = c_res.n_prompt_tokens;
        r.tokens_per_second = c_res.tokens_per_second;
        r.gpu_used = (c_res.gpu_used != 0);
        if (c_res.backend_used) {
            r.backend_used = c_res.backend_used;
        }
        r.error = c_res.error;
        if (c_res.error_message) {
            r.error_message = c_res.error_message;
        }
        return r;
    }

    using TokenCallback = std::function<bool(const std::string& token_text, int token_id)>;

    HaError infer_stream(const InferParams& params, const TokenCallback& callback) {
        if (!handle_) {
            return HA_ERR_INVALID_ARG;
        }
        HaInferParams c_params = params.to_c_params();

        auto trampoline = [](const char* text, int tid, void* user_data) -> int {
            auto* cb = static_cast<const TokenCallback*>(user_data);
            if (!cb || !(*cb)) return 0;
            bool continue_stream = (*cb)(text ? text : "", tid);
            return continue_stream ? 0 : 1;
        };

        return ha_infer_stream(handle_, &c_params, trampoline, const_cast<void*>(static_cast<const void*>(&callback)));
    }

private:
    HaSession handle_ = nullptr;
};

class Model {
public:
    explicit Model(HaModel handle) : handle_(handle) {
        if (!handle_) {
            throw Exception(HA_ERR_INVALID_ARG, "Cannot wrap null HaModel handle");
        }
    }

    ~Model() {
        if (handle_) {
            ha_model_unload(handle_);
            handle_ = nullptr;
        }
    }

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                ha_model_unload(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    HaModel handle() const noexcept { return handle_; }

    std::string arch() const {
        const char* a = ha_model_arch(handle_);
        return a ? a : "";
    }

    uint32_t context_length() const {
        return ha_model_context_length(handle_);
    }

    uint32_t vocab_size() const {
        return ha_model_vocab_size(handle_);
    }

    Session create_session() {
        HaError err = HA_OK;
        HaSession s = ha_session_create(handle_, &err);
        if (!s || err != HA_OK) {
            throw Exception(err, std::string("Failed to create session: ") + ha_error_string(err));
        }
        return Session(s);
    }

private:
    HaModel handle_ = nullptr;
};

class Runtime {
public:
    Runtime() {
        handle_ = ha_runtime_create();
        if (!handle_) {
            throw Exception(HA_ERR_INTERNAL, "Failed to initialize HeteroAccel runtime");
        }
    }

    explicit Runtime(HaRuntime handle) : handle_(handle) {
        if (!handle_) {
            throw Exception(HA_ERR_INVALID_ARG, "Cannot wrap null HaRuntime handle");
        }
    }

    ~Runtime() {
        if (handle_) {
            ha_runtime_destroy(handle_);
            handle_ = nullptr;
        }
    }

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    Runtime(Runtime&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Runtime& operator=(Runtime&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                ha_runtime_destroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    HaRuntime handle() const noexcept { return handle_; }

    static std::string version() {
        return ha_version_string();
    }

    int backend_count() const {
        return ha_backend_count(handle_);
    }

    BackendInfo backend_info(int index) const {
        HaBackendInfo c_info = {};
        HaError err = ha_backend_get(handle_, index, &c_info);
        if (err != HA_OK) {
            throw Exception(err, std::string("Failed to get backend info: ") + ha_error_string(err));
        }
        BackendInfo bi;
        bi.backend_id = c_info.backend_id ? c_info.backend_id : "";
        bi.available = (c_info.available != 0);
        bi.requires_vulkan = (c_info.requires_vulkan != 0);
        bi.requires_cuda = (c_info.requires_cuda != 0);
        return bi;
    }

    std::vector<BackendInfo> backends() const {
        std::vector<BackendInfo> list;
        int count = backend_count();
        list.reserve(count);
        for (int i = 0; i < count; ++i) {
            list.push_back(backend_info(i));
        }
        return list;
    }

    Model load_model(const std::string& path) {
        HaError err = HA_OK;
        HaModel m = ha_model_load(handle_, path.c_str(), &err);
        if (!m || err != HA_OK) {
            throw Exception(err, std::string("Failed to load model '") + path + "': " + ha_error_string(err));
        }
        return Model(m);
    }

private:
    HaRuntime handle_ = nullptr;
};

} // namespace ha

#endif // HETEROACCEL_HPP
