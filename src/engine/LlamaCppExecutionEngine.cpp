// src/engine/LlamaCppExecutionEngine.cpp
// Phase 11: IExecutionEngine adapter over LlamaCppBackend.
#include "engine/LlamaCppExecutionEngine.h"
#include "engine/ExecutionEngineRegistry.h"

#include <filesystem>

namespace agr {

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------
LlamaCppExecutionEngine::LlamaCppExecutionEngine()
    : backend_(std::make_shared<LlamaCppBackend>())
{}

LlamaCppExecutionEngine::~LlamaCppExecutionEngine() {
    shutdown();
}

// ---------------------------------------------------------------------------
// Identity & capabilities
// ---------------------------------------------------------------------------
EngineIdentity LlamaCppExecutionEngine::identity() const {
    return {
        /* name         */ "llama.cpp",
        /* version      */ "b9999",
        /* type         */ "LLM",
        /* author       */ "Georgi Gerganov / ggerganov",
        /* registry_key */ "llama.cpp"
    };
}

EngineCapabilitySet LlamaCppExecutionEngine::capabilities() const {
    return EngineCapabilitySet{
        EngineCapability::TEXT_GENERATION,
        EngineCapability::BACKEND_CPU,
        EngineCapability::BACKEND_VULKAN,
        EngineCapability::BACKEND_CUDA,
        EngineCapability::STREAMING,
        EngineCapability::CANCELLATION,
        EngineCapability::DYNAMIC_LOADING,
    };
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------
bool LlamaCppExecutionEngine::initialize() {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    if (initialized_) return true;
    // LlamaCppBackend is already constructed; the underlying llama_backend_init
    // is called lazily inside LlamaCppEngine::initialize() when a model loads.
    initialized_ = true;
    lastError_   = EngineError::success();
    return true;
}

void LlamaCppExecutionEngine::shutdown() {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    if (!initialized_) return;
    if (backend_) {
        backend_->unload();
    }
    modelLoaded_ = false;
    initialized_ = false;
}

bool LlamaCppExecutionEngine::isInitialized() const {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    return initialized_;
}

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------
bool LlamaCppExecutionEngine::loadModel(const ModelDescriptor& descriptor) {
    std::lock_guard<std::mutex> lk(engine_mutex_);

    if (!initialized_) {
        lastError_ = EngineError::make(EngineErrorKind::ENGINE_UNAVAILABLE,
                                       "Engine not initialised. Call initialize() first.");
        return false;
    }

    if (descriptor.source_path.empty()) {
        lastError_ = EngineError::make(EngineErrorKind::INVALID_REQUEST,
                                       "ModelDescriptor.source_path is empty.");
        return false;
    }

    // Check file exists
    std::error_code ec;
    if (!std::filesystem::exists(descriptor.source_path, ec) || ec) {
        lastError_ = EngineError::make(EngineErrorKind::MODEL_NOT_FOUND,
                                       "Model file not found: " + descriptor.source_path);
        return false;
    }

    // Unload any previously loaded model
    backend_->unload();
    modelLoaded_ = false;

    // Determine n_gpu_layers from descriptor metadata (if present)
    // Default: attempt to offload all layers (99 = "as many as fit")
    int n_gpu_layers = 99;
    bool cpu_only    = false;

    // Phase 13 will implement planning; Phase 11 always uses defaults.
    // A caller can pre-populate descriptor.metadata["n_gpu_layers"] for now.
    if (descriptor.metadata.count("n_gpu_layers")) {
        try { n_gpu_layers = std::stoi(descriptor.metadata.at("n_gpu_layers")); }
        catch (...) {}
    }
    if (descriptor.metadata.count("cpu_only")) {
        cpu_only = (descriptor.metadata.at("cpu_only") == "1" ||
                    descriptor.metadata.at("cpu_only") == "true");
    }

    if (!backend_->loadModel(descriptor.source_path, n_gpu_layers, cpu_only)) {
        lastError_ = EngineError::make(EngineErrorKind::MODEL_LOAD_FAILED,
                                       backend_->lastError());
        return false;
    }

    // Create a default inference context
    GenerationOptions opts;
    if (descriptor.metadata.count("n_ctx")) {
        try { opts.n_ctx = std::stoi(descriptor.metadata.at("n_ctx")); } catch (...) {}
    }
    if (!backend_->createContext(opts)) {
        lastError_ = EngineError::make(EngineErrorKind::MODEL_LOAD_FAILED,
                                       backend_->lastError());
        backend_->unload();
        return false;
    }

    currentDescriptor_ = descriptor;
    // Enrich descriptor from backend's reported ModelInfo
    ModelInfo info = backend_->modelInfo();
    currentDescriptor_.architecture = info.architecture;
    currentDescriptor_.quantization  = info.quantization;
    if (!currentDescriptor_.size_bytes.has_value() && info.size_bytes > 0)
        currentDescriptor_.size_bytes = info.size_bytes;

    modelLoaded_ = true;
    lastError_   = EngineError::success();
    return true;
}

void LlamaCppExecutionEngine::unloadModel() {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    if (backend_) backend_->unload();
    modelLoaded_ = false;
}

bool LlamaCppExecutionEngine::isModelLoaded() const {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    return modelLoaded_;
}

ModelDescriptor LlamaCppExecutionEngine::currentModel() const {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    return currentDescriptor_;
}

// ---------------------------------------------------------------------------
// Execution helpers
// ---------------------------------------------------------------------------
InferenceRequest LlamaCppExecutionEngine::buildRequest(
    const ExecutionContext& ctx) const
{
    InferenceRequest req;
    req.request_id         = ctx.request_id;
    req.prompt             = ctx.prompt;
    req.options.max_tokens = ctx.max_tokens;
    req.options.temperature= ctx.temperature;
    req.options.seed       = ctx.seed;
    req.options.n_ctx      = ctx.n_ctx;
    req.options.n_batch    = ctx.n_batch;
    req.options.n_threads  = ctx.n_threads;
    req.prefer_gpu         = !ctx.cpu_only;
    req.allow_cpu_fallback = true;
    req.priority           = ctx.priority;
    req.workload_class     = ctx.workload_class;
    req.cancel_flag        = ctx.cancel_flag;
    return req;
}

ExecutionResult LlamaCppExecutionEngine::mapResult(const InferenceResult& r) const {
    ExecutionResult er;
    er.success = r.success;
    er.output  = r.text;
    er.error   = r.error;

    er.telemetry.backend_name   = "LlamaCppBackend";
    er.telemetry.device_name    = r.telemetry.device_name;
    er.telemetry.gpu_used       = r.telemetry.vulkan_used || r.telemetry.gpu_layers > 0;
    er.telemetry.gpu_layers     = r.telemetry.gpu_layers;
    er.telemetry.prompt_tokens  = r.telemetry.prompt_tokens;
    er.telemetry.output_tokens  = r.telemetry.output_tokens;
    er.telemetry.model_load_ms  = r.telemetry.model_load_ms;
    er.telemetry.prompt_eval_ms = r.telemetry.prompt_eval_ms;
    er.telemetry.generation_ms  = r.telemetry.generation_ms;
    er.telemetry.total_ms       = r.telemetry.total_ms;
    er.telemetry.ttft_ms        = r.telemetry.ttft_ms;
    er.telemetry.tokens_per_sec = r.telemetry.tokens_per_sec;
    er.telemetry.fallbacks      = r.telemetry.fallbacks;

    // Map backend string to enum
    if (r.telemetry.backend == "Vulkan" || r.telemetry.vulkan_used) {
        er.telemetry.backend_type = ExecutionBackendType::VULKAN;
    } else if (r.telemetry.backend == "CUDA") {
        er.telemetry.backend_type = ExecutionBackendType::CUDA;
    } else {
        er.telemetry.backend_type = ExecutionBackendType::CPU;
    }

    return er;
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------
ExecutionResult LlamaCppExecutionEngine::execute(const ExecutionContext& ctx) {
    std::lock_guard<std::mutex> lk(engine_mutex_);

    if (!initialized_) {
        ExecutionResult er;
        er.error = "Engine not initialised";
        lastError_ = EngineError::make(EngineErrorKind::ENGINE_UNAVAILABLE, er.error);
        return er;
    }
    if (!modelLoaded_) {
        ExecutionResult er;
        er.error = "No model loaded";
        lastError_ = EngineError::make(EngineErrorKind::MODEL_NOT_LOADED, er.error);
        return er;
    }

    // Re-create context with the current execution context's parameters
    GenerationOptions opts;
    opts.max_tokens  = ctx.max_tokens;
    opts.temperature = ctx.temperature;
    opts.seed        = ctx.seed;
    opts.n_ctx       = ctx.n_ctx;
    opts.n_batch     = ctx.n_batch;
    opts.n_threads   = ctx.n_threads;
    backend_->createContext(opts);

    InferenceRequest req = buildRequest(ctx);
    InferenceResult  r   = backend_->generate(req);
    auto er = mapResult(r);
    if (!er.success) {
        lastError_ = EngineError::make(EngineErrorKind::EXECUTION_FAILED, er.error);
    } else {
        lastError_ = EngineError::success();
    }
    return er;
}

ExecutionResult LlamaCppExecutionEngine::executeStreaming(
    const ExecutionContext& ctx, TokenCallback cb)
{
    std::lock_guard<std::mutex> lk(engine_mutex_);

    if (!initialized_) {
        ExecutionResult er;
        er.error = "Engine not initialised";
        lastError_ = EngineError::make(EngineErrorKind::ENGINE_UNAVAILABLE, er.error);
        return er;
    }
    if (!modelLoaded_) {
        ExecutionResult er;
        er.error = "No model loaded";
        lastError_ = EngineError::make(EngineErrorKind::MODEL_NOT_LOADED, er.error);
        return er;
    }

    GenerationOptions opts;
    opts.max_tokens  = ctx.max_tokens;
    opts.temperature = ctx.temperature;
    opts.seed        = ctx.seed;
    opts.n_ctx       = ctx.n_ctx;
    opts.n_batch     = ctx.n_batch;
    opts.n_threads   = ctx.n_threads;
    backend_->createContext(opts);

    InferenceRequest req = buildRequest(ctx);
    InferenceResult  r   = cb ? backend_->generateStreaming(req, cb)
                               : backend_->generate(req);
    auto er = mapResult(r);
    if (!er.success) {
        lastError_ = EngineError::make(EngineErrorKind::EXECUTION_FAILED, er.error);
    } else {
        lastError_ = EngineError::success();
    }
    return er;
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------
EngineError LlamaCppExecutionEngine::lastEngineError() const {
    std::lock_guard<std::mutex> lk(engine_mutex_);
    return lastError_;
}

} // namespace agr
