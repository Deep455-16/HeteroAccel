// include/heteroaccel/HeteroAccel.hpp
// HeteroAccel Phase 24: Header-only Modern C++ RAII SDK Wrapper
//
// Provides safe, idiomatic C++ RAII classes wrapping the C API.

#ifndef HETEROACCEL_HPP
#define HETEROACCEL_HPP

#include "heteroaccel.h"

#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ha {

class Exception : public std::exception {
public:
    Exception(HaError code, const std::string& message)
        : code_(code), msg_(message) {}

    const char* what() const noexcept override { return msg_.c_str(); }

    HaError code() const noexcept { return code_; }

private:
    HaError code_;
    std::string msg_;
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

struct MemoryEstimate {
    size_t weights_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t activations_bytes = 0;
    size_t overhead_bytes = 0;
    size_t total_estimated = 0;
    bool unified_memory = false;
    std::string notes;
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
            throw Exception(HA_ERR_INVALID_ARG, "Session is not initialized");
        }
        HaInferParams c_params = params.to_c_params();
        HaResult res = ha_infer(handle_, &c_params);

        Result r;
        r.error = res.error;
        if (res.error_message) r.error_message = res.error_message;
        if (res.error != HA_OK) {
            return r;
        }

        if (res.output_text) r.output_text = res.output_text;
        if (res.output_token_ids && res.n_output_tokens > 0) {
            r.output_token_ids.assign(res.output_token_ids, res.output_token_ids + res.n_output_tokens);
        }
        r.n_output_tokens = res.n_output_tokens;
        r.n_prompt_tokens = res.n_prompt_tokens;
        r.tokens_per_second = res.tokens_per_second;
        r.gpu_used = (res.gpu_used != 0);
        if (res.backend_used) r.backend_used = res.backend_used;
        return r;
    }

    Result infer(const std::string& prompt, int max_new_tokens = 16) {
        InferParams params;
        params.prompt = prompt;
        params.max_new_tokens = max_new_tokens;
        return infer(params);
    }

    using TokenCallback = std::function<bool(const std::string& token_text, int token_id)>;

    /// Runs native incremental token streaming.
    /// The callback is invoked as each token is generated.
    HaError infer_stream(const InferParams& params, const TokenCallback& callback) {
        if (!handle_) {
            return HA_ERR_INVALID_ARG;
        }
        HaInferParams c_params = params.to_c_params();

        auto trampoline = [](const char* text, int tid, void* user_data) -> int {
            auto* cb = static_cast<const TokenCallback*>(user_data);
            if (!cb || !(*cb)) return 0;
            bool continue_stream = (*cb)(text ? text : "", tid);
            return continue_stream ? 0 : 1; // 0 to continue, 1 to cancel
        };

        return ha_infer_stream(handle_, &c_params, trampoline, const_cast<void*>(static_cast<const void*>(&callback)));
    }

    Result infer_stream(const std::string& prompt, const TokenCallback& callback, int max_new_tokens = 16) {
        InferParams params;
        params.prompt = prompt;
        params.max_new_tokens = max_new_tokens;
        HaError err = infer_stream(params, callback);
        Result r;
        r.error = err;
        r.error_message = ha_error_string(err);
        return r;
    }

    void cancel() {
        if (handle_) {
            ha_session_cancel(handle_);
        }
    }

    HaSession release() noexcept {
        HaSession h = handle_;
        handle_ = nullptr;
        return h;
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

    MemoryEstimate memory_estimate(uint32_t max_seq_len = 2048) const {
        HaMemoryEstimate est = {};
        HaError err = ha_model_memory_estimate(handle_, max_seq_len, &est);
        if (err != HA_OK) {
            throw Exception(err, "Failed to compute model memory estimate");
        }
        MemoryEstimate me;
        me.weights_bytes = est.weights_bytes;
        me.kv_cache_bytes = est.kv_cache_bytes;
        me.activations_bytes = est.activations_bytes;
        me.overhead_bytes = est.overhead_bytes;
        me.total_estimated = est.total_estimated;
        me.unified_memory = (est.unified_memory != 0);
        if (est.notes) me.notes = est.notes;
        return me;
    }

    Session create_session() {
        HaError err = HA_OK;
        HaSession s = ha_session_create(handle_, &err);
        if (!s || err != HA_OK) {
            throw Exception(err, std::string("Failed to create session: ") + ha_error_string(err));
        }
        return Session(s);
    }

    HaModel release() noexcept {
        HaModel h = handle_;
        handle_ = nullptr;
        return h;
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
            throw Exception(err, std::string("Failed to load model: ") + ha_error_string(err));
        }
        return Model(m);
    }

    void perf_reset() {
        if (handle_) {
            ha_runtime_perf_reset(handle_);
        }
    }

    std::string perf_path() const {
        if (!handle_) return "";
        char buf[512] = {0};
        ha_runtime_perf_path(handle_, buf, sizeof(buf));
        return std::string(buf);
    }

    HaRuntime release() noexcept {
        HaRuntime h = handle_;
        handle_ = nullptr;
        return h;
    }

private:
    HaRuntime handle_ = nullptr;
};

} // namespace ha

#endif // HETEROACCEL_HPP
