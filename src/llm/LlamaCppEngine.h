#pragma once
#include "llm/LlmConfig.h"
#include "llm/LlmResult.h"

#include <string>
#include <vector>

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
    LlmResult infer(const std::string& prompt);

    // Release model, context, sampler and free llama.cpp backend.
    // Safe to call multiple times.
    void shutdown();

    bool        isInitialized()        const { return initialized_; }
    std::string lastError()            const { return lastError_; }
    std::string detectedVulkanDevice() const { return vulkanDevice_; }
    bool        vulkanConfirmed()      const { return vulkanConfirmed_; }
    int         gpuLayersActual()      const { return gpuLayersActual_; }
    double      modelLoadMs()          const { return modelLoadMs_; }

private:
    // Static log callback installed via llama_log_set().
    // llama.cpp uses a single global callback, so we track the
    // "engine currently being initialized" via a static pointer.
    static void logCallback(int level, const char* text, void* user_data);

    llama_model*   model_   = nullptr;
    llama_context* ctx_     = nullptr;

    LlmConfig   config_;
    bool        initialized_     = false;
    bool        backendInited_   = false;

    // Populated by logCallback during initialize()
    bool        vulkanConfirmed_ = false;
    std::string vulkanDevice_;
    int         gpuLayersActual_ = 0;

    double      modelLoadMs_     = 0.0;
    std::string lastError_;

    // Global state for the log-capture window (set/cleared in initialize)
    static LlamaCppEngine* s_currentEngine_;
};

} // namespace agr
