// src/sdk/HeteroAccelSDK.cpp
// HeteroAccel Phase 23: Stable Public C API and Developer SDK Implementation

#include "heteroaccel/heteroaccel.h"

#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/GGUFLoader.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// Internal implementation structures backing opaque C handles

struct BackendEntry {
    std::string id;
    int available = 0;
    int requires_vulkan = 0;
    int requires_cuda = 0;
};

struct HaRuntime_ {
    agr::orchestration::HardwareDetector* hw = nullptr;
    agr::orchestration::EngineOrchestrator* orch = nullptr;
    std::vector<BackendEntry> backends;
};

struct HaModel_ {
    HaRuntime rt = nullptr;
    std::string path;
    std::string arch;
    uint32_t context_length = 0;
    uint32_t vocab_size = 0;
    size_t model_size_bytes = 0;
    int quant_type = 12; // default GGML_TYPE_Q4_K
    std::unique_ptr<agr::hydroxl::HydroXLTokenizer> tokenizer;
};

struct HaSession_ {
    std::mutex mutex;
    HaModel model = nullptr;
    std::shared_ptr<agr::orchestration::IInferenceEngine> engine;

    // Output memory buffers owned by session (valid until next call or destroy)
    std::string last_output_text;
    std::vector<int> last_output_token_ids;
    std::string last_backend_used;
    std::string last_error_message;
};

// ---------------------------------------------------------------------------
// Runtime Lifecycle
// ---------------------------------------------------------------------------

HaRuntime ha_runtime_create(void) {
    try {
        auto* rt = new HaRuntime_();
        rt->hw = &agr::orchestration::HardwareDetector::instance();
        rt->hw->detect();

        rt->orch = &agr::orchestration::EngineOrchestrator::instance();
        agr::orchestration::register_default_backends(*rt->orch, *rt->hw);

        auto registered = rt->orch->backends();
        rt->backends.clear();
        rt->backends.reserve(registered.size());
        for (const auto& b : registered) {
            BackendEntry entry;
            entry.id = b.backend_id;
            entry.available = b.is_available(*rt->hw) ? 1 : 0;
            entry.requires_vulkan = b.requires_vulkan ? 1 : 0;
            entry.requires_cuda = b.requires_cuda ? 1 : 0;
            rt->backends.push_back(std::move(entry));
        }

        return rt;
    } catch (...) {
        return nullptr;
    }
}

void ha_runtime_destroy(HaRuntime rt) {
    if (!rt) return;
    try {
        delete rt;
    } catch (...) {}
}

const char* ha_version_string(void) {
    return "0.23.0";
}

// ---------------------------------------------------------------------------
// Model Loading & Metadata
// ---------------------------------------------------------------------------

HaModel ha_model_load(HaRuntime rt, const char* path, HaError* err_out) {
    if (err_out) *err_out = HA_OK;

    if (!rt || !path || path[0] == '\0') {
        if (err_out) *err_out = HA_ERR_INVALID_ARG;
        return nullptr;
    }

    try {
        std::filesystem::path fs_path(path);
        if (!std::filesystem::exists(fs_path) || !std::filesystem::is_regular_file(fs_path)) {
            if (err_out) *err_out = HA_ERR_MODEL_LOAD;
            return nullptr;
        }

        agr::hydroxl::GGUFLoader loader;
        if (!loader.open(path)) {
            if (err_out) *err_out = HA_ERR_MODEL_LOAD;
            return nullptr;
        }

        std::string arch = loader.kvString("general.architecture", "qwen2");
        if (arch.empty()) arch = "qwen2";

        uint32_t ctx = loader.kvUInt32(arch + ".context_length", 0);
        if (ctx == 0) ctx = loader.kvUInt32("qwen2.context_length", 32768);

        uint32_t vocab = loader.kvUInt32("llm.vocab_size", 0);
        if (vocab == 0) vocab = loader.kvUInt32(arch + ".vocab_size", 0);
        if (vocab == 0) vocab = loader.kvUInt32("qwen2.vocab_size", 151936);

        auto tokenizer = std::make_unique<agr::hydroxl::HydroXLTokenizer>();
        if (tokenizer->loadFromGGUF(loader)) {
            if (vocab == 0 || vocab == 151936) {
                if (tokenizer->vocabSize() > 0) {
                    vocab = tokenizer->vocabSize();
                }
            }
        }

        int quant_type = 12; // default Q4_K
        for (const auto& t : loader.tensors()) {
            if (t.type != agr::hydroxl::GGMLType::UNKNOWN) {
                quant_type = static_cast<int>(t.type);
                break;
            }
        }

        auto* model = new HaModel_();
        model->rt = rt;
        model->path = path;
        model->arch = arch;
        model->context_length = ctx;
        model->vocab_size = vocab;
        model->model_size_bytes = static_cast<size_t>(loader.fileSize());
        model->quant_type = quant_type;
        model->tokenizer = std::move(tokenizer);

        return model;
    } catch (const std::bad_alloc&) {
        if (err_out) *err_out = HA_ERR_OOM;
        return nullptr;
    } catch (...) {
        if (err_out) *err_out = HA_ERR_INTERNAL;
        return nullptr;
    }
}

void ha_model_unload(HaModel model) {
    if (!model) return;
    try {
        delete model;
    } catch (...) {}
}

const char* ha_model_arch(HaModel model) {
    if (!model) return "";
    return model->arch.c_str();
}

uint32_t ha_model_context_length(HaModel model) {
    if (!model) return 0;
    return model->context_length;
}

uint32_t ha_model_vocab_size(HaModel model) {
    if (!model) return 0;
    return model->vocab_size;
}

// ---------------------------------------------------------------------------
// Inference Session
// ---------------------------------------------------------------------------

HaSession ha_session_create(HaModel model, HaError* err_out) {
    if (err_out) *err_out = HA_OK;

    if (!model || model->path.empty()) {
        if (err_out) *err_out = HA_ERR_INVALID_ARG;
        return nullptr;
    }

    try {
        agr::orchestration::EngineOrchestrator::SelectionRequest sel_req;
        sel_req.model_path = model->path;
        sel_req.model_arch = model->arch;
        sel_req.quant_type = model->quant_type;
        sel_req.model_size_bytes = model->model_size_bytes;
        sel_req.n_ctx = (model->context_length > 0) ? (std::min)(model->context_length, 2048u) : 512;
        sel_req.prefer_latency = false;
        sel_req.allow_cpu_fallback = true;

        auto& orch = agr::orchestration::EngineOrchestrator::instance();
        auto sel_res = orch.select_backend(sel_req);

        std::string create_err;
        auto engine = orch.create_engine(sel_res, sel_req, create_err);

        // Explicit fallback to CPU if orchestrator returned null
        if (!engine) {
            auto cpu_engine = std::make_shared<agr::orchestration::HydroXLCpuEngineAdapter>();
            std::string cpu_err;
            if (cpu_engine->initialize(model->path, cpu_err)) {
                engine = cpu_engine;
            } else {
                if (err_out) *err_out = HA_ERR_MODEL_LOAD;
                return nullptr;
            }
        }

        if (!engine || !engine->is_ready()) {
            if (err_out) *err_out = HA_ERR_NO_BACKEND;
            return nullptr;
        }

        auto* session = new HaSession_();
        session->model = model;
        session->engine = engine;
        session->last_backend_used = engine->backend_id();

        return session;
    } catch (const std::bad_alloc&) {
        if (err_out) *err_out = HA_ERR_OOM;
        return nullptr;
    } catch (...) {
        if (err_out) *err_out = HA_ERR_INTERNAL;
        return nullptr;
    }
}

void ha_session_destroy(HaSession session) {
    if (!session) return;
    try {
        if (session->engine) {
            session->engine->shutdown();
        }
        delete session;
    } catch (...) {}
}

// ---------------------------------------------------------------------------
// Inference Execution
// ---------------------------------------------------------------------------

HaResult ha_infer(HaSession session, const HaInferParams* params) {
    HaResult res = {};
    res.error = HA_OK;
    res.error_message = "";
    res.output_text = const_cast<char*>("");
    res.output_token_ids = nullptr;
    res.n_output_tokens = 0;
    res.n_prompt_tokens = 0;
    res.tokens_per_second = 0.0f;
    res.gpu_used = 0;
    res.backend_used = "";

    if (!session || !params) {
        res.error = HA_ERR_INVALID_ARG;
        res.error_message = "Session or params pointer is NULL";
        return res;
    }

    std::lock_guard<std::mutex> lk(session->mutex);

    if (!session->engine || !session->engine->is_ready()) {
        session->last_error_message = "Inference engine is not initialized or ready";
        res.error = HA_ERR_INFERENCE;
        res.error_message = session->last_error_message.c_str();
        return res;
    }

    try {
        agr::orchestration::InferenceRequest req;
        req.max_new_tokens = (params->max_new_tokens > 0) ? params->max_new_tokens : 16;
        req.temperature = params->temperature;
        req.top_p = (params->top_p > 0.0f) ? params->top_p : 0.95f;

        if (params->token_ids && params->n_token_ids > 0) {
            req.prompt_token_ids.assign(params->token_ids, params->token_ids + params->n_token_ids);
        } else if (params->prompt && params->prompt[0] != '\0') {
            if (session->model && session->model->tokenizer && session->model->tokenizer->isLoaded()) {
                auto ids = session->model->tokenizer->encode(params->prompt, session->model->tokenizer->addsBos());
                req.prompt_token_ids.reserve(ids.size());
                for (auto id : ids) {
                    req.prompt_token_ids.push_back(static_cast<int>(id));
                }
            }
        }

        agr::orchestration::InferenceResult inf_res = session->engine->run(req);

        session->last_backend_used = inf_res.backend_used;
        if (!inf_res.success) {
            session->last_error_message = inf_res.error.empty() ? "Inference run failed" : inf_res.error;
            session->last_output_text.clear();
            session->last_output_token_ids.clear();

            res.error = HA_ERR_INFERENCE;
            res.error_message = session->last_error_message.c_str();
            res.backend_used = session->last_backend_used.c_str();
            res.gpu_used = inf_res.gpu_used ? 1 : 0;
            return res;
        }

        session->last_output_text = std::move(inf_res.output_text);
        session->last_output_token_ids = std::move(inf_res.output_token_ids);
        session->last_error_message.clear();

        res.output_text = const_cast<char*>(session->last_output_text.c_str());
        res.output_token_ids = session->last_output_token_ids.empty() ? nullptr : session->last_output_token_ids.data();
        res.n_output_tokens = static_cast<int>(session->last_output_token_ids.size());
        res.n_prompt_tokens = inf_res.prompt_tokens;
        res.tokens_per_second = inf_res.tokens_per_second;
        res.gpu_used = inf_res.gpu_used ? 1 : 0;
        res.backend_used = session->last_backend_used.c_str();
        res.error = HA_OK;
        res.error_message = "";
        return res;
    } catch (const std::bad_alloc&) {
        session->last_error_message = "Out of memory during inference";
        res.error = HA_ERR_OOM;
        res.error_message = session->last_error_message.c_str();
        return res;
    } catch (const std::exception& e) {
        session->last_error_message = e.what();
        res.error = HA_ERR_INTERNAL;
        res.error_message = session->last_error_message.c_str();
        return res;
    } catch (...) {
        session->last_error_message = "Unknown exception during inference";
        res.error = HA_ERR_INTERNAL;
        res.error_message = session->last_error_message.c_str();
        return res;
    }
}

HaError ha_infer_stream(HaSession session, const HaInferParams* params,
                        HaTokenCallback cb, void* user_data) {
    if (!session || !params || !cb) {
        return HA_ERR_INVALID_ARG;
    }

    // Post-hoc simulated streaming: executes inference pass, then delivers
    // token pieces in sequence.
    HaResult res = ha_infer(session, params);
    if (res.error != HA_OK) {
        return res.error;
    }

    if (session->model && session->model->tokenizer && session->model->tokenizer->isLoaded()
        && !session->last_output_token_ids.empty()) {
        for (int tid : session->last_output_token_ids) {
            std::string piece = session->model->tokenizer->decodeToken(static_cast<agr::hydroxl::TokenID>(tid));
            int cancel = cb(piece.c_str(), tid, user_data);
            if (cancel != 0) {
                return HA_ERR_CANCELLED;
            }
        }
    } else {
        std::istringstream iss(session->last_output_text);
        std::string word;
        int idx = 0;
        while (iss >> word) {
            int cancel = cb(word.c_str(), idx++, user_data);
            if (cancel != 0) {
                return HA_ERR_CANCELLED;
            }
        }
    }

    return HA_OK;
}

// ---------------------------------------------------------------------------
// Backend Introspection & Error Strings
// ---------------------------------------------------------------------------

int ha_backend_count(HaRuntime rt) {
    if (!rt) return 0;
    return static_cast<int>(rt->backends.size());
}

HaError ha_backend_get(HaRuntime rt, int index, HaBackendInfo* info_out) {
    if (!rt || !info_out || index < 0 || index >= static_cast<int>(rt->backends.size())) {
        return HA_ERR_INVALID_ARG;
    }

    const auto& b = rt->backends[index];
    info_out->backend_id = b.id.c_str();
    info_out->available = b.available;
    info_out->requires_vulkan = b.requires_vulkan;
    info_out->requires_cuda = b.requires_cuda;
    return HA_OK;
}

const char* ha_error_string(HaError err) {
    switch (err) {
        case HA_OK:              return "Success";
        case HA_ERR_INVALID_ARG: return "Invalid argument";
        case HA_ERR_MODEL_LOAD:  return "Failed to load model";
        case HA_ERR_NO_BACKEND:  return "No suitable backend found";
        case HA_ERR_INFERENCE:   return "Inference execution failed";
        case HA_ERR_CANCELLED:   return "Operation cancelled";
        case HA_ERR_OOM:         return "Out of memory";
        case HA_ERR_UNSUPPORTED: return "Operation unsupported";
        case HA_ERR_INTERNAL:    return "Internal runtime error";
        default:                 return "Unknown error";
    }
}
