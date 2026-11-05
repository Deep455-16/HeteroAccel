// src/sdk/HeteroAccelSDK.cpp
// HeteroAccel Phase 24: Native Streaming and Intelligent Runtime Resource Management

#define _CRT_SECURE_NO_WARNINGS
#include "heteroaccel/heteroaccel.h"

#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/HydroXLEngineAdapter.h"
#include "orchestration/MemoryEstimator.h"
#include "orchestration/PerfProfile.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "hydroxl/GGUFLoader.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
    agr::orchestration::PerfProfile perf_profile;
    std::string perf_profile_path;
    std::mutex perf_mutex;
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

    // Hyperparameters for memory estimation & profiling
    uint32_t n_layers = 0;
    uint32_t n_embd = 0;
    uint32_t n_heads = 0;
    uint32_t n_kv_heads = 0;
    uint32_t n_ff = 0;
    std::string model_hash;
    mutable std::string last_notes;
};

struct HaSession_ {
    std::mutex mutex;
    HaModel model = nullptr;
    std::shared_ptr<agr::orchestration::IInferenceEngine> engine;
    std::atomic<bool> cancelled{false};

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

        rt->perf_profile_path = agr::orchestration::PerfProfile::default_profile_path();
        rt->perf_profile = agr::orchestration::PerfProfile::load(rt->perf_profile_path);

        // Seed orchestrator benchmark history with known records from profile
        for (const auto& rec : rt->perf_profile.records()) {
            if (rec.tokens_per_second > 0.0f) {
                rt->orch->record_performance(rec.backend_id, rec.tokens_per_second, 0);
            }
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
    return "0.25.0";
}

HaError ha_runtime_perf_reset(HaRuntime rt) {
    if (!rt) return HA_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> plk(rt->perf_mutex);
    rt->perf_profile.reset();
    rt->perf_profile.save(rt->perf_profile_path);
    return HA_OK;
}

HaError ha_runtime_perf_path(HaRuntime rt, char* buf, size_t buf_size) {
    if (!rt || !buf || buf_size == 0) return HA_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> plk(rt->perf_mutex);
    const auto& p = rt->perf_profile_path;
    if (p.size() >= buf_size) {
        std::strncpy(buf, p.c_str(), buf_size - 1);
        buf[buf_size - 1] = '\0';
    } else {
        std::strcpy(buf, p.c_str());
    }
    return HA_OK;
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

        uint32_t n_layers   = loader.kvUInt32(arch + ".block_count", 0);
        if (n_layers == 0) n_layers = loader.kvUInt32("qwen2.block_count", 24);

        uint32_t n_embd     = loader.kvUInt32(arch + ".embedding_length", 0);
        if (n_embd == 0) n_embd = loader.kvUInt32("qwen2.embedding_length", 896);

        uint32_t n_heads    = loader.kvUInt32(arch + ".attention.head_count", 0);
        if (n_heads == 0) n_heads = loader.kvUInt32("qwen2.attention.head_count", 14);

        uint32_t n_kv_heads = loader.kvUInt32(arch + ".attention.head_count_kv", 0);
        if (n_kv_heads == 0) n_kv_heads = loader.kvUInt32("qwen2.attention.head_count_kv", 2);

        uint32_t n_ff       = loader.kvUInt32(arch + ".feed_forward_length", 0);
        if (n_ff == 0) n_ff = loader.kvUInt32("qwen2.feed_forward_length", 4864);

        auto* model = new HaModel_();
        model->rt = rt;
        model->path = path;
        model->arch = arch;
        model->context_length = ctx;
        model->vocab_size = vocab;
        model->model_size_bytes = static_cast<size_t>(loader.fileSize());
        model->quant_type = quant_type;
        model->tokenizer = std::move(tokenizer);
        model->n_layers = n_layers;
        model->n_embd = n_embd;
        model->n_heads = n_heads;
        model->n_kv_heads = n_kv_heads;
        model->n_ff = n_ff;
        model->model_hash = agr::orchestration::PerfProfile::compute_model_hash(path);

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

HaError ha_model_memory_estimate(HaModel model, uint32_t max_seq_len,
                                 HaMemoryEstimate* out) {
    if (!model || !out) return HA_ERR_INVALID_ARG;

    if (max_seq_len == 0) max_seq_len = 2048;

    auto est = agr::orchestration::MemoryEstimator::estimate(
        model->n_layers,
        model->n_embd,
        model->n_heads,
        model->n_kv_heads,
        model->n_ff,
        model->vocab_size,
        max_seq_len
    );

    model->last_notes = est.notes;

    out->weights_bytes = est.weights_bytes;
    out->kv_cache_bytes = est.kv_cache_bytes;
    out->activations_bytes = est.activations_bytes;
    out->overhead_bytes = est.overhead_bytes;
    out->total_estimated = est.total_estimated;
    out->unified_memory = est.unified_memory ? 1 : 0;
    out->notes = model->last_notes.c_str();

    return HA_OK;
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
        sel_req.n_layers = model->n_layers;
        sel_req.n_embd = model->n_embd;
        sel_req.n_heads = model->n_heads;
        sel_req.n_kv_heads = model->n_kv_heads;
        sel_req.n_ff = model->n_ff;
        sel_req.vocab_size = model->vocab_size;

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

HaError ha_session_cancel(HaSession session) {
    if (!session) return HA_ERR_INVALID_ARG;
    session->cancelled.store(true);
    return HA_OK;
}

HaError ha_infer_cancel(HaSession session) {
    return ha_session_cancel(session);
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
    if (session->cancelled.exchange(false)) {
        session->last_error_message = "Operation cancelled before execution";
        res.error = HA_ERR_CANCELLED;
        res.error_message = session->last_error_message.c_str();
        return res;
    }

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
        req.stream = false;

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

        // Record performance measurement and save asynchronously
        if (session->model && session->model->rt && res.tokens_per_second > 0.0f) {
            auto* rt = session->model->rt;
            std::string b_id = session->last_backend_used;
            std::string m_hash = session->model->model_hash;
            uint32_t q_type = static_cast<uint32_t>(session->model->quant_type);
            uint32_t n_lay = session->model->n_layers;
            uint32_t n_emb = session->model->n_embd;
            float tps = res.tokens_per_second;

            {
                std::lock_guard<std::mutex> plk(rt->perf_mutex);
                rt->perf_profile.record(b_id, m_hash, q_type, n_lay, n_emb, tps);
            }

            auto profile_copy = rt->perf_profile;
            std::string save_path = rt->perf_profile_path;
            try {
                std::thread([profile_copy, save_path]() {
                    profile_copy.save(save_path);
                }).detach();
            } catch (...) {
                rt->perf_profile.save(save_path);
            }
        }

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

    std::lock_guard<std::mutex> lk(session->mutex);
    if (session->cancelled.exchange(false)) {
        session->last_error_message = "Operation cancelled before execution";
        return HA_ERR_CANCELLED;
    }

    if (!session->engine || !session->engine->is_ready()) {
        session->last_error_message = "Inference engine is not initialized or ready";
        return HA_ERR_INFERENCE;
    }

    try {
        agr::orchestration::InferenceRequest req;
        req.max_new_tokens = (params->max_new_tokens > 0) ? params->max_new_tokens : 16;
        req.temperature = params->temperature;
        req.top_p = (params->top_p > 0.0f) ? params->top_p : 0.95f;
        req.stream = true;

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

        // Native incremental token streaming
        if (session->engine->supports_streaming()) {
            agr::orchestration::InferenceResult inf_res;
            bool ok = session->engine->run_stream(req, [&](const std::string& text, int id) -> bool {
                if (session->cancelled.load()) {
                    return false;
                }
                int cancel = cb(text.c_str(), id, user_data);
                if (cancel != 0) {
                    session->cancelled.store(true);
                    return false;
                }
                return true;
            }, inf_res);

            session->last_backend_used = inf_res.backend_used;
            session->last_output_text = std::move(inf_res.output_text);
            session->last_output_token_ids = std::move(inf_res.output_token_ids);

            // Record perf if available
            if (session->model && session->model->rt && inf_res.tokens_per_second > 0.0f) {
                auto* rt = session->model->rt;
                std::string b_id = session->last_backend_used;
                std::string m_hash = session->model->model_hash;
                uint32_t q_type = static_cast<uint32_t>(session->model->quant_type);
                uint32_t n_lay = session->model->n_layers;
                uint32_t n_emb = session->model->n_embd;
                float tps = inf_res.tokens_per_second;

                {
                    std::lock_guard<std::mutex> plk(rt->perf_mutex);
                    rt->perf_profile.record(b_id, m_hash, q_type, n_lay, n_emb, tps);
                }

                auto profile_copy = rt->perf_profile;
                std::string save_path = rt->perf_profile_path;
                try {
                    std::thread([profile_copy, save_path]() {
                        profile_copy.save(save_path);
                    }).detach();
                } catch (...) {
                    rt->perf_profile.save(save_path);
                }
            }

            if (session->cancelled.load()) {
                session->last_error_message = "Operation cancelled";
                return HA_ERR_CANCELLED;
            }

            if (!ok || !inf_res.success) {
                session->last_error_message = inf_res.error.empty() ? "Inference streaming failed" : inf_res.error;
                return HA_ERR_INFERENCE;
            }

            session->last_error_message.clear();
            return HA_OK;
        }

        // Fallback: post-hoc simulated streaming for engines that don't support native streaming
        agr::orchestration::InferenceResult inf_res = session->engine->run(req);
        session->last_backend_used = inf_res.backend_used;
        if (!inf_res.success) {
            session->last_error_message = inf_res.error.empty() ? "Inference run failed" : inf_res.error;
            return HA_ERR_INFERENCE;
        }

        session->last_output_text = std::move(inf_res.output_text);
        session->last_output_token_ids = std::move(inf_res.output_token_ids);

        if (session->model && session->model->tokenizer && session->model->tokenizer->isLoaded()
            && !session->last_output_token_ids.empty()) {
            for (int tid : session->last_output_token_ids) {
                if (session->cancelled.load()) return HA_ERR_CANCELLED;
                std::string piece = session->model->tokenizer->decodeToken(static_cast<agr::hydroxl::TokenID>(tid));
                int cancel = cb(piece.c_str(), tid, user_data);
                if (cancel != 0) {
                    session->cancelled.store(true);
                    return HA_ERR_CANCELLED;
                }
            }
        } else {
            std::istringstream iss(session->last_output_text);
            std::string word;
            int idx = 0;
            while (iss >> word) {
                if (session->cancelled.load()) return HA_ERR_CANCELLED;
                int cancel = cb(word.c_str(), idx++, user_data);
                if (cancel != 0) {
                    session->cancelled.store(true);
                    return HA_ERR_CANCELLED;
                }
            }
        }

        return HA_OK;
    } catch (const std::bad_alloc&) {
        session->last_error_message = "Out of memory during streaming inference";
        return HA_ERR_OOM;
    } catch (const std::exception& e) {
        session->last_error_message = e.what();
        return HA_ERR_INTERNAL;
    } catch (...) {
        session->last_error_message = "Unknown exception during streaming inference";
        return HA_ERR_INTERNAL;
    }
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
