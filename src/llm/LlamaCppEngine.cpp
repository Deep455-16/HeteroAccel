#include "llm/LlamaCppEngine.h"
#include "generation/GenerationBudget.h"

// llama.cpp C API — only included here, not in any header.
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>

namespace agr {

// ---------------------------------------------------------------------------
// Static state
// ---------------------------------------------------------------------------
LlamaCppEngine* LlamaCppEngine::s_currentEngine_ = nullptr;
bool LlamaCppEngine::s_vulkanConfirmed_ = false;
std::string LlamaCppEngine::s_vulkanDevice_;

static int s_backendRefCount = 0;
static std::mutex s_backendMutex;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return s;
}

// ggml backend registry name ("CUDA", "Vulkan", "CPU", ...) for a device.
std::string deviceKind(ggml_backend_dev_t dev) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* n = reg ? ggml_backend_reg_name(reg) : nullptr;
    return lowerCopy(n ? n : "");
}

// GPU-class devices of one backend kind, in ggml enumeration order.
std::vector<ggml_backend_dev_t> gpuDevicesOfKind(const std::string& kind) {
    std::vector<ggml_backend_dev_t> out;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        auto t = ggml_backend_dev_type(d);
        if (t != GGML_BACKEND_DEVICE_TYPE_GPU && t != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
        if (deviceKind(d) == kind) out.push_back(d);
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// Global log callback (llama.cpp allows exactly one)
// ---------------------------------------------------------------------------
void LlamaCppEngine::logCallback(int level, const char* text, void* /*user_data*/) {
    if (!text) return;
    std::string msg(text);
    
    // Detect Vulkan initialization
    if (msg.find("ggml_vulkan") != std::string::npos) {
        s_vulkanConfirmed_ = true;

        // Extract device name from a line like:
        // "ggml_vulkan: Using device 0: Intel(R) Iris(R) Xe Graphics ..."
        const char* usingDevice = "Using device";
        auto pos = msg.find(usingDevice);
        if (pos != std::string::npos) {
            // Find the colon after "Using device N:"
            auto colon = msg.find(':', pos + strlen(usingDevice));
            if (colon != std::string::npos) {
                std::string device = msg.substr(colon + 1);
                // Trim leading/trailing whitespace and newlines
                auto first = device.find_first_not_of(" \t\r\n");
                auto last  = device.find_last_not_of(" \t\r\n");
                if (first != std::string::npos)
                    s_vulkanDevice_ = device.substr(first, last - first + 1);
            }
        }
    }

    if (!s_currentEngine_) return;
    LlamaCppEngine* eng = s_currentEngine_;

    // Capture ERROR-level llama.cpp/ggml lines (e.g. CUDA out-of-memory) so a
    // failed load can report the real cause rather than a generic message.
    if (level == 4 /*GGML_LOG_LEVEL_ERROR*/ && eng->initErrorLog_.size() < 2048) {
        eng->initErrorLog_ += msg;
    }
    // Detect GPU layer count from lines like:
    // "llm_load_tensors: offloaded 24/24 layers to GPU"
    if (msg.find("offloaded") != std::string::npos && msg.find("layers to GPU") != std::string::npos) {
        // Try to parse "offloaded N/M layers"
        auto pos = msg.find("offloaded");
        if (pos != std::string::npos) {
            int n = 0;
            if (std::sscanf(msg.c_str() + pos, "offloaded %d", &n) == 1) {
                eng->gpuLayersActual_ = n;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// initialize()
// ---------------------------------------------------------------------------
bool LlamaCppEngine::initialize(const LlmConfig& config) {
    if (initialized_) {
        lastError_ = "Already initialized. Call shutdown() first.";
        return false;
    }

    config_ = config;

    // Validate configuration
    if (config_.n_ctx <= 0) config_.n_ctx = 2048;
    if (config_.n_batch <= 0) config_.n_batch = 512;
    if (config_.max_new_tokens <= 0) config_.max_new_tokens = 256;

    // Validate model path
    if (config_.model_path.empty()) {
        lastError_ = "Model path is empty";
        return false;
    }
    {
        // Quick file-exists check before handing off to llama.cpp
        FILE* f = std::fopen(config_.model_path.c_str(), "rb");
        if (!f) {
            lastError_ = "Model file not found or not readable: " + config_.model_path;
            return false;
        }
        std::fclose(f);
    }

    // Install our log callback FIRST so we capture Vulkan init messages
    s_currentEngine_ = this;
    llama_log_set(reinterpret_cast<ggml_log_callback>(&LlamaCppEngine::logCallback), nullptr);

    // Initialize llama.cpp backend (loads Vulkan/CUDA/CPU backends)
    {
        std::lock_guard<std::mutex> lk(s_backendMutex);
        if (s_backendRefCount == 0) {
            llama_backend_init();
        }
        s_backendRefCount++;
    }
    backendInited_ = true;

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config_.effective_gpu_layers();

    // Explicit backend selection: restrict llama.cpp to exactly one device kind.
    // Storage must outlive llama_model_load_from_file (NULL-terminated list).
    std::vector<ggml_backend_dev_t> selectedDevices;
    const std::string pref = lowerCopy(config_.backend_preference);
    const bool explicitPref = !pref.empty() && pref != "auto";
    initErrorLog_.clear();
    activeBackend_.clear();
    activeDevice_.clear();
    devTotal_ = devFree_ = 0;

    if (explicitPref) {
        if (pref == "cpu") {
            config_.cpu_only     = true;
            mparams.n_gpu_layers = 0;
            selectedDevices.push_back(nullptr);          // empty device list: CPU only
            activeBackend_ = "CPU";
        } else if (pref == "cuda" || pref == "vulkan") {
            auto devs = gpuDevicesOfKind(pref);
            const std::string label = (pref == "cuda") ? "CUDA" : "Vulkan";
            if (devs.empty()) {
                lastError_ = label + " backend unavailable: llama.cpp was " +
                             (pref == "cuda" ? "not built with GGML_CUDA (configure with -DHETEROACCEL_ENABLE_CUDA=ON) or no usable CUDA device exists"
                                             : "not built with GGML_VULKAN or no usable Vulkan device exists");
                s_currentEngine_ = nullptr;
                {
                    std::lock_guard<std::mutex> lk(s_backendMutex);
                    s_backendRefCount--;
                    if (s_backendRefCount == 0) llama_backend_free();
                }
                backendInited_ = false;
                return false;
            }
            int idx = config_.gpu_device_index;
            if (idx < 0 || idx >= static_cast<int>(devs.size())) {
                lastError_ = label + " device index " + std::to_string(idx) + " out of range (" +
                             std::to_string(devs.size()) + " device(s) available)";
                s_currentEngine_ = nullptr;
                {
                    std::lock_guard<std::mutex> lk(s_backendMutex);
                    s_backendRefCount--;
                    if (s_backendRefCount == 0) llama_backend_free();
                }
                backendInited_ = false;
                return false;
            }
            ggml_backend_dev_t dev = devs[static_cast<size_t>(idx)];
            selectedDevices.push_back(dev);
            selectedDevices.push_back(nullptr);
            mparams.devices    = selectedDevices.data();
            mparams.main_gpu   = 0;
            mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
            const char* desc = ggml_backend_dev_description(dev);
            activeDevice_  = desc ? desc : "";
            activeBackend_ = label;
            if (config_.effective_gpu_layers() <= 0) activeBackend_ = "CPU";
        } else {
            lastError_ = "Unknown backend preference '" + config_.backend_preference +
                         "' (expected auto, cpu, cuda or vulkan)";
            s_currentEngine_ = nullptr;
            {
                std::lock_guard<std::mutex> lk(s_backendMutex);
                s_backendRefCount--;
                if (s_backendRefCount == 0) llama_backend_free();
            }
            backendInited_ = false;
            return false;
        }
        if (pref == "cpu") mparams.devices = selectedDevices.data();
    }

    double t0 = nowMs();
    model_ = llama_model_load_from_file(config_.model_path.c_str(), mparams);
    modelLoadMs_ = nowMs() - t0;

    // Done capturing init-time logs
    s_currentEngine_ = nullptr;

    if (!model_) {
        lastError_ = "llama_model_load_from_file failed. "
                     "Check that the file is a valid GGUF model and that "
                     "there is enough RAM/VRAM available.";
        if (!initErrorLog_.empty()) lastError_ += " llama.cpp reported: " + initErrorLog_;
        {
            std::lock_guard<std::mutex> lk(s_backendMutex);
            s_backendRefCount--;
            if (s_backendRefCount == 0) llama_backend_free();
        }
        backendInited_ = false;
        return false;
    }

    if (explicitPref && selectedDevices.size() >= 2 && selectedDevices[0]) {
        ggml_backend_dev_memory(selectedDevices[0], &devFree_, &devTotal_);
        if (gpuLayersActual_ == 0) {
            // Log line not captured; the device list was restricted to this GPU so
            // every requested layer that exists in the model was offloaded to it.
            gpuLayersActual_ = std::min(config_.effective_gpu_layers(),
                                        static_cast<int>(llama_model_n_layer(model_)) + 1);
        }
    }

    // -----------------------------------------------------------------------
    // Create inference context
    // -----------------------------------------------------------------------
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(config_.n_ctx);
    cparams.n_batch   = static_cast<uint32_t>(config_.n_batch);
    if (config_.n_threads > 0) {
        cparams.n_threads = static_cast<int32_t>(config_.n_threads);
    } else {
        cparams.n_threads = static_cast<int32_t>(
            std::max(1u, std::thread::hardware_concurrency() / 2));
    }

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        lastError_ = "llama_init_from_model failed";
        llama_model_free(model_);
        model_ = nullptr;
        {
            std::lock_guard<std::mutex> lk(s_backendMutex);
            s_backendRefCount--;
            if (s_backendRefCount == 0) llama_backend_free();
        }
        backendInited_ = false;
        return false;
    }

    // Populate actual GPU layer count if not captured from logs
    if (gpuLayersActual_ == 0 && s_vulkanConfirmed_ && config_.effective_gpu_layers() > 0) {
        // llama.cpp may have loaded all requested layers; use requested count as best estimate
        gpuLayersActual_ = config_.effective_gpu_layers();
    }

    initialized_ = true;
    return true;
}

void LlamaCppEngine::fillMetadata(LlmResult& result) {
    // Fill result metadata from engine state
    // -----------------------------------------------------------------------
    result.gpu_layers_requested = config_.effective_gpu_layers();
    result.gpu_layers_actual    = gpuLayersActual_;
    bool vulkan_used_this_run   = s_vulkanConfirmed_ && (config_.effective_gpu_layers() > 0);
    result.vulkan_confirmed     = vulkan_used_this_run;
    result.device_name          = s_vulkanDevice_;
    result.backend              = vulkan_used_this_run ? "Vulkan" : "CPU";
    if (!activeBackend_.empty()) {
        // Explicit selection (Phase 26): report exactly what was selected, and only
        // claim a GPU backend when layers were really offloaded to it.
        const bool gpuBackend = (activeBackend_ == "CUDA" || activeBackend_ == "Vulkan");
        if (gpuBackend && gpuLayersActual_ > 0) {
            result.backend = activeBackend_;
            result.vulkan_confirmed = (activeBackend_ == "Vulkan");
            result.cuda_confirmed   = (activeBackend_ == "CUDA");
            result.device_name      = activeDevice_;
            result.device_total_bytes = devTotal_;
            result.device_free_bytes  = devFree_;
        } else {
            result.backend = "CPU";
            result.gpu_layers_actual = 0;
            result.vulkan_confirmed = false;
            result.cuda_confirmed   = false;
            result.device_name.clear();
        }
    }
    result.model_load_ms        = modelLoadMs_;
    modelLoadMs_                = 0.0; // 0 for subsequent inferences on this already-loaded engine

}

void LlamaCppEngine::runDecode(std::vector<int32_t>& tokens, LlmResult& result,
                               std::atomic<bool>* cancel_flag) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);

    // Every inference starts from an empty KV cache so repeated calls on one
    // engine are independent and reproducible (positions restart at 0).
    llama_memory_clear(llama_get_memory(ctx_), true);

    // -----------------------------------------------------------------------
    // Prompt evaluation
    // -----------------------------------------------------------------------
    double t_prompt_start = nowMs();
    {
        int n_batch = std::max(1, config_.n_batch);
        for (int i = 0; i < static_cast<int>(tokens.size()); i += n_batch) {
            int n_eval = std::min(n_batch, static_cast<int>(tokens.size()) - i);
            llama_batch batch = llama_batch_get_one(tokens.data() + i, n_eval);

            int ret = llama_decode(ctx_, batch);
            if (ret != 0) {
                result.error = "llama_decode failed during prompt evaluation (ret=" +
                               std::to_string(ret) + ", chunk starting at " + std::to_string(i) + ")";
                return;
            }
        }
    }
    result.prompt_eval_ms = nowMs() - t_prompt_start;

    if (config_.capture_first_logits) {
        const float* lg = llama_get_logits_ith(ctx_, -1);
        const int32_t nv = llama_vocab_n_tokens(vocab);
        if (lg && nv > 0) result.first_token_logits.assign(lg, lg + nv);
    }

    // -----------------------------------------------------------------------
    // Token generation
    // -----------------------------------------------------------------------

    // Build sampler chain: optional penalties -> temperature -> dist (multinomial)
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    if (config_.repeat_penalty > 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(64));
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, config_.repeat_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(config_.seed));

    std::string output;
    output.reserve(256);

    // Phase 33: wall-clock budget, stop sequences, real streaming, finish reason.
    generation::Deadline deadline(config_.time_budget_ms);
    generation::StopSequenceFilter stop_filter(config_.stop_sequences);
    std::string finish_reason = generation::finish::kMaxTokens;
    std::string utf8_pending;
    bool stop_hit = false;
    bool stream_cancelled = false;
    bool cancelled = false;
    TokenStreamCallback stream_cb = std::move(stream_cb_);
    stream_cb_ = nullptr;
    auto emit = [&](const std::string& text, int tok_id) {
        if (text.empty()) return;
        output += text;
        if (stream_cb && !stream_cb(text, tok_id)) stream_cancelled = true;
    };

    double t_gen_start = nowMs();
    int n_generated = 0;

    for (int i = 0; i < config_.max_new_tokens; ++i) {
        if (cancel_flag && cancel_flag->load(std::memory_order_acquire)) {
            result.error = "Generation cancelled";
            finish_reason = generation::finish::kCancelled;
            cancelled = true;
            break;
        }
        if (deadline.expired()) {
            finish_reason = generation::finish::kTimeBudget;
            break;
        }

        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1);

        // End-of-generation token?
        if (llama_vocab_is_eog(vocab, new_token)) {
            finish_reason = generation::finish::kEos;
            break;
        }

        // Token -> text piece (complete UTF-8 codepoints only)
        char piece[256];
        int piece_len = llama_token_to_piece(vocab, new_token,
                                              piece, static_cast<int32_t>(sizeof(piece)),
                                              /* lstrip = */ 0,
                                              /* special = */ false);
        if (piece_len > 0) {
            utf8_pending.append(piece, static_cast<size_t>(piece_len));
            std::string complete = utf8_pending;
            generation::trimIncompleteUtf8Tail(complete);
            utf8_pending.erase(0, complete.size());
            bool hit = false;
            emit(stop_filter.feed(complete, hit), static_cast<int>(new_token));
            if (hit) stop_hit = true;
        }
        result.output_token_ids.push_back(static_cast<int>(new_token));

        if (stop_hit) {
            finish_reason = generation::finish::kStopSequence;
            break;
        }
        if (stream_cancelled) {
            result.error = "Generation cancelled";
            finish_reason = generation::finish::kCancelled;
            cancelled = true;
            break;
        }

        // Advance context: single-token decode
        llama_pos current_pos = static_cast<llama_pos>(tokens.size() + i);
        if (current_pos >= config_.n_ctx) {
            // Context limit reached
            finish_reason = generation::finish::kContextLimit;
            break;
        }

        llama_batch next = llama_batch_get_one(&new_token, 1);
        int ret = llama_decode(ctx_, next);
        if (ret != 0) {
            // Treat as soft stop (context may be full)
            finish_reason = generation::finish::kContextLimit;
            break;
        }

        ++n_generated;
    }

    if (!stop_hit && !stream_cancelled && !cancelled) {
        emit(stop_filter.flush(), result.output_token_ids.empty() ? 0 : result.output_token_ids.back());
        if (stream_cancelled) {
            result.error = "Generation cancelled";
            finish_reason = generation::finish::kCancelled;
            cancelled = true;
        }
    }
    // A token limit may cut a multibyte character in half; utf8_pending is dropped.
    result.finish_reason = finish_reason;

    result.generation_ms = nowMs() - t_gen_start;

    llama_sampler_free(sampler);

    // -----------------------------------------------------------------------
    // Populate result
    // -----------------------------------------------------------------------
    result.output        = output;
    result.output_tokens = n_generated;
    result.total_ms      = result.model_load_ms + result.prompt_eval_ms + result.generation_ms;
    result.tokens_per_sec = (result.generation_ms > 0.0)
                                ? (n_generated / (result.generation_ms / 1000.0))
                                : 0.0;
    result.success = !cancelled;
}

// ---------------------------------------------------------------------------
// infer()
// ---------------------------------------------------------------------------
LlmResult LlamaCppEngine::infer(const std::string& prompt, std::atomic<bool>* cancel_flag) {
    LlmResult result;

    if (!initialized_) {
        result.error = "Engine not initialized. Call initialize() first.";
        return result;
    }

    fillMetadata(result);

    // -----------------------------------------------------------------------
    // Tokenize prompt
    // -----------------------------------------------------------------------
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(config_.n_ctx);
    int n_prompt = llama_tokenize(vocab,
                                  prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  /* add_special = */ true,
                                  /* parse_special = */ config_.parse_special);
    if (n_prompt < 0) {
        // Buffer was too small — resize to exact requirement and retry
        tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab,
                                  prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  true, config_.parse_special);
    }
    if (n_prompt <= 0) {
        result.error = "Tokenization failed (returned " + std::to_string(n_prompt) + ")";
        return result;
    }
    tokens.resize(static_cast<size_t>(n_prompt));
    result.prompt_tokens = n_prompt;

    if (cancel_flag && cancel_flag->load(std::memory_order_acquire)) {
        result.error = "Cancelled before evaluation";
        result.finish_reason = generation::finish::kCancelled;
        return result;
    }

    runDecode(tokens, result, cancel_flag);
    return result;
}

LlmResult LlamaCppEngine::inferTokens(const std::vector<int>& prompt_ids,
                                     std::atomic<bool>* cancel_flag) {
    LlmResult result;
    if (!initialized_) {
        result.error = "Engine not initialized. Call initialize() first.";
        return result;
    }
    if (prompt_ids.empty()) {
        result.error = "Empty prompt token list";
        return result;
    }
    if (static_cast<int>(prompt_ids.size()) >= config_.n_ctx) {
        result.error = "Prompt (" + std::to_string(prompt_ids.size()) + " tokens) does not fit the context window (" + std::to_string(config_.n_ctx) + ")";
        return result;
    }
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<int32_t> tokens;
    tokens.reserve(prompt_ids.size());
    for (int id : prompt_ids) {
        if (id < 0 || id >= n_vocab) {
            result.error = "Prompt token id " + std::to_string(id) + " is outside the model vocabulary";
            return result;
        }
        tokens.push_back(static_cast<int32_t>(id));
    }
    fillMetadata(result);
    result.prompt_tokens = static_cast<int>(tokens.size());
    runDecode(tokens, result, cancel_flag);
    return result;
}

// ---------------------------------------------------------------------------
// shutdown()
// ---------------------------------------------------------------------------
void LlamaCppEngine::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    if (backendInited_) {
        {
            std::lock_guard<std::mutex> lk(s_backendMutex);
            s_backendRefCount--;
            if (s_backendRefCount == 0) llama_backend_free();
        }
        backendInited_ = false;
    }
    initialized_     = false;
    gpuLayersActual_ = 0;
    lastError_.clear();
}

} // namespace agr
