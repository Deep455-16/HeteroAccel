#pragma once
#include "llm/LlmConfig.h"
#include "llm/LlmResult.h"

#include <string>
#include <vector>
#include <atomic>
#include <functional>
#include <utility>

// Forward-declare llama.cpp opaque types so callers of this header
// do NOT need to include llama.h (keeping llama.cpp out of the wider codebase).
struct llama_model;
struct llama_context;
struct llama_sampler;

namespace agr {

// Adapter over the llama.cpp C API.
//
// Usage:
//   LlamaCppEngine engine;
//   if (!engine.initialize(config)) { /* handle error */ }
//   LlmResult r = engine.infer("Hello, what is Vulkan?");
//   engine.shutdown();
//
// Thread safety: NOT thread-safe. Create one engine per thread if needed.
class LlamaCppEngine {
public:
    LlamaCppEngine()  = default;
    ~LlamaCppEngine() { shutdown(); }

    // Disallow copy
    LlamaCppEngine(const LlamaCppEngine&)            = delete;
    LlamaCppEngine& operator=(const LlamaCppEngine&) = delete;

    // Load model and create context.
    // Must be called before infer().
    // Returns false on failure; call lastError() for the reason.
    bool initialize(const LlmConfig& config);

    // Run inference on a prompt.
    // Returns a populated LlmResult (check result.success).
    LlmResult infer(const std::string& prompt, std::atomic<bool>* cancel_flag = nullptr);

    // Run inference from already-tokenized prompt ids (vocabulary ids of THIS
    // model). Used by the orchestrated CUDA/llama.cpp path, whose requests carry
    // token ids produced by the shared GGUF tokenizer. Ids are range-checked.
    LlmResult inferTokens(const std::vector<int>& prompt_ids,
                          std::atomic<bool>* cancel_flag = nullptr);

    // Adjust generation settings between runs (does not reload the model).
    void setGeneration(int max_new_tokens, float temperature, uint32_t seed) {
        if (max_new_tokens > 0) config_.max_new_tokens = max_new_tokens;
        config_.temperature = temperature;
        config_.seed = seed;
    }
    int contextSize() const { return config_.n_ctx; }

    // Phase 33: per-run wall-clock budget and stop sequences (does not reload).
    void setBudget(int time_budget_ms, std::vector<std::string> stop_sequences) {
        config_.time_budget_ms = time_budget_ms;
        config_.stop_sequences = std::move(stop_sequences);
    }
    void setParseSpecial(bool parse_special) { config_.parse_special = parse_special; }
    void setRepeatPenalty(float penalty) { config_.repeat_penalty = penalty; }

    // Phase 33: real per-token streaming. The callback receives complete UTF-8
    // text pieces as they are generated; returning false cancels generation.
    // Cleared automatically after each infer()/inferTokens() call.
    using TokenStreamCallback = std::function<bool(const std::string& piece, int token_id)>;
    void setStreamCallback(TokenStreamCallback cb) { stream_cb_ = std::move(cb); }

    // Release model, context, sampler and free llama.cpp backend.
    // Safe to call multiple times.
    void shutdown();

    bool        isInitialized()        const { return initialized_; }
    std::string lastError()            const { return lastError_; }
    std::string detectedVulkanDevice() const { return s_vulkanDevice_; }
    bool        vulkanConfirmed()      const { return s_vulkanConfirmed_; }
    int         gpuLayersActual()      const { return gpuLayersActual_; }
    double      modelLoadMs()          const { return modelLoadMs_; }
    /// Backend actually in use after initialize(): "CUDA" | "Vulkan" | "CPU".
    std::string activeBackend()        const { return activeBackend_; }
    std::string activeDeviceName()     const { return activeDevice_; }
    size_t      deviceTotalBytes()     const { return devTotal_; }
    size_t      deviceFreeBytes()      const { return devFree_; }

private:
    void fillMetadata(LlmResult& result);
    void runDecode(std::vector<int32_t>& tokens, LlmResult& result,
                   std::atomic<bool>* cancel_flag);

    // Static log callback installed via llama_log_set().
    // llama.cpp uses a single global callback, so we track the
    // "engine currently being initialized" via a static pointer.
    static void logCallback(int level, const char* text, void* user_data);

    llama_model*   model_   = nullptr;
    llama_context* ctx_     = nullptr;

    LlmConfig   config_;
    TokenStreamCallback stream_cb_;
    bool        initialized_     = false;
    bool        backendInited_   = false;

    int         gpuLayersActual_ = 0;

    // Per-engine selection state (Phase 26)
    std::string activeBackend_;
    std::string activeDevice_;
    size_t      devTotal_ = 0;
    size_t      devFree_  = 0;
    std::string initErrorLog_;   // llama.cpp ERROR-level log lines captured during init

    double      modelLoadMs_     = 0.0;
    std::string lastError_;

    // Global state for the log-capture window (set/cleared in initialize)
    static LlamaCppEngine* s_currentEngine_;
    
    // Global state for Vulkan discovery (llama.cpp only logs this once per process)
    static bool        s_vulkanConfirmed_;
    static std::string s_vulkanDevice_;
};

} // namespace agr
