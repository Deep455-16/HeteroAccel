#include "llm/LlamaCppEngine.h"

// llama.cpp C API — only included here, not in any header.
#include "llama.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace agr {

// ---------------------------------------------------------------------------
// Static state — used only during the initialize() window so that the global
// llama_log_set callback can route messages to the right engine instance.
// ---------------------------------------------------------------------------
LlamaCppEngine* LlamaCppEngine::s_currentEngine_ = nullptr;
bool LlamaCppEngine::s_vulkanConfirmed_ = false;
std::string LlamaCppEngine::s_vulkanDevice_;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

// ---------------------------------------------------------------------------
// Global log callback (llama.cpp allows exactly one)
// ---------------------------------------------------------------------------
void LlamaCppEngine::logCallback(int /*level*/, const char* text, void* /*user_data*/) {
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
    llama_backend_init();
    backendInited_ = true;

    // -----------------------------------------------------------------------
    // Load model
    // -----------------------------------------------------------------------
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = config_.effective_gpu_layers();

    double t0 = nowMs();
    model_ = llama_model_load_from_file(config_.model_path.c_str(), mparams);
    modelLoadMs_ = nowMs() - t0;

    // Done capturing init-time logs
    s_currentEngine_ = nullptr;

    if (!model_) {
        lastError_ = "llama_model_load_from_file failed. "
                     "Check that the file is a valid GGUF model and that "
                     "there is enough RAM/VRAM available.";
        llama_backend_free();
        backendInited_ = false;
        return false;
    }

    // -----------------------------------------------------------------------
    // Create inference context
    // -----------------------------------------------------------------------
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(config_.n_ctx);
    cparams.n_batch   = static_cast<uint32_t>(config_.n_batch);
    // Use up to half the available hardware threads for the CPU layers
    // (leaves threads free for GPU driver and OS).
    cparams.n_threads = static_cast<int32_t>(
        std::max(1u, std::thread::hardware_concurrency() / 2));

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        lastError_ = "llama_init_from_model failed";
        llama_model_free(model_);
        model_ = nullptr;
        llama_backend_free();
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

// ---------------------------------------------------------------------------
// infer()
// ---------------------------------------------------------------------------
LlmResult LlamaCppEngine::infer(const std::string& prompt) {
    LlmResult result;

    if (!initialized_) {
        result.error = "Engine not initialized. Call initialize() first.";
        return result;
    }

    // -----------------------------------------------------------------------
    // Fill result metadata from engine state
    // -----------------------------------------------------------------------
    result.gpu_layers_requested = config_.effective_gpu_layers();
    result.gpu_layers_actual    = gpuLayersActual_;
    result.vulkan_confirmed     = s_vulkanConfirmed_;
    result.device_name          = s_vulkanDevice_;
    result.backend              = (s_vulkanConfirmed_ && config_.effective_gpu_layers() > 0)
                                      ? "Vulkan" : "CPU";
    result.model_load_ms        = modelLoadMs_;

    // -----------------------------------------------------------------------
    // Tokenize prompt
    // -----------------------------------------------------------------------
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::vector<llama_token> tokens(config_.n_ctx);
    int n_prompt = llama_tokenize(vocab,
                                  prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  /* add_special = */ true,
                                  /* parse_special = */ true);
    if (n_prompt < 0) {
        // Buffer was too small — resize to exact requirement and retry
        tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(vocab,
                                  prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                  tokens.data(), static_cast<int32_t>(tokens.size()),
                                  true, true);
    }
    if (n_prompt <= 0) {
        result.error = "Tokenization failed (returned " + std::to_string(n_prompt) + ")";
        return result;
    }
    tokens.resize(static_cast<size_t>(n_prompt));
    result.prompt_tokens = n_prompt;

    // -----------------------------------------------------------------------
    // Prompt evaluation
    // -----------------------------------------------------------------------
    double t_prompt_start = nowMs();
    {
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        int ret = llama_decode(ctx_, batch);
        if (ret != 0) {
            result.error = "llama_decode failed during prompt evaluation (ret=" +
                           std::to_string(ret) + ")";
            return result;
        }
    }
    result.prompt_eval_ms = nowMs() - t_prompt_start;

    // -----------------------------------------------------------------------
    // Token generation
    // -----------------------------------------------------------------------

    // Build sampler chain: temperature → dist (multinomial)
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(config_.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(config_.seed));

    std::string output;
    output.reserve(256);

    double t_gen_start = nowMs();
    int n_generated = 0;

    for (int i = 0; i < config_.max_new_tokens; ++i) {
        llama_token new_token = llama_sampler_sample(sampler, ctx_, -1);

        // End-of-generation token?
        if (llama_token_is_eog(vocab, new_token)) break;

        // Token → text piece
        char piece[256];
        int piece_len = llama_token_to_piece(vocab, new_token,
                                              piece, static_cast<int32_t>(sizeof(piece)),
                                              /* lstrip = */ 0,
                                              /* special = */ false);
        if (piece_len > 0) {
            output.append(piece, static_cast<size_t>(piece_len));
        }

        // Advance context: single-token decode
        llama_batch next = llama_batch_get_one(&new_token, 1);
        int ret = llama_decode(ctx_, next);
        if (ret != 0) {
            // Treat as soft stop (context may be full)
            break;
        }

        ++n_generated;
    }

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
    result.success = true;
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
        llama_backend_free();
        backendInited_ = false;
    }
    initialized_     = false;
    gpuLayersActual_ = 0;
    lastError_.clear();
}

} // namespace agr
