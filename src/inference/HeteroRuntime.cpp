// src/inference/HeteroRuntime.cpp
// Phase 7: HeteroRuntime implementation.
#include "inference/HeteroRuntime.h"
#include "inference/LlamaCppBackend.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>

namespace agr {

namespace {
double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}
} // namespace

// ---------------------------------------------------------------------------
bool HeteroRuntime::initialize() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (initialized_) return true;

    // Phase 1: Hardware Detection
    hardware_ = HardwareDetector::detectAll();

    // Phase 4: Vulkan backend + backend manager
    vulkan_     = std::make_unique<VulkanBackend>();
    backendMgr_ = std::make_unique<BackendManager>(*vulkan_);
    backendMgr_->discover();

    // Phase 4: Memory Manager
    memMgr_ = std::make_unique<MemoryManager>(*vulkan_);

    // Phase 5: Adaptive Scheduler
    scheduler_ = std::make_unique<AdaptiveScheduler>(*backendMgr_, *memMgr_);

    // Phase 8: Profiling and Auto-Tuning
    profiler_ = std::make_unique<Profiler>(scheduler_->history(), hardware_);
    autoTuner_ = std::make_unique<AutoTuner>(scheduler_->history(), hardware_);

    // Try loading persistent profile
    profiler_->loadProfile("profiles/performance.json");

    initialized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
void HeteroRuntime::resolveBackendConfig(int& out_gpu_layers, bool& out_cpu_only) const {
    // Legacy Phase 7 logic — we keep this signature but AutoTuner handles tuning.
    out_gpu_layers = 0;
    out_cpu_only   = true;
    if (!backendMgr_) return;
    bool vulkan_avail = backendMgr_->isBackendAvailable(ComputeBackend::VULKAN);
    bool cuda_avail   = backendMgr_->isBackendAvailable(ComputeBackend::CUDA);
    MemoryStats memStats;
    if (memMgr_) memStats = memMgr_->statistics();
    bool gpu_memory_ok = (memStats.gpu_pressure != PressureLevel::CRITICAL);
    if ((cuda_avail || vulkan_avail) && gpu_memory_ok) {
        out_gpu_layers = 99; 
        out_cpu_only   = false;
    } else {
        out_gpu_layers = 0;
        out_cpu_only   = true;
    }
}

// ---------------------------------------------------------------------------
bool HeteroRuntime::loadModel(const std::string& model_path,
                               const GenerationOptions& opts) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!initialized_) return false;
    if (backends_.count(model_path)) return true;

    int max_threads = std::thread::hardware_concurrency();
    int gpu_layers = 0;
    bool cpu_only = true;
    resolveBackendConfig(gpu_layers, cpu_only);

    ProfileKey key;
    key.hardware_id = hardware_.cpu.model_name;
    key.backend = cpu_only ? ComputeBackend::CPU : ComputeBackend::VULKAN;
    key.workload_type = "llm-inference";
    key.model_name = model_path;

    TuningConfig tcfg = autoTuner_->suggestConfiguration(key, !cpu_only, max_threads);
    GenerationOptions tunedOpts = opts;
    tunedOpts.n_threads = tcfg.n_threads;

    auto backend = std::make_unique<LlamaCppBackend>();
    if (!backend->loadModel(model_path, tcfg.n_gpu_layers, tcfg.n_gpu_layers == 0)) return false;
    if (!backend->createContext(tunedOpts)) return false;

    backends_[model_path] = std::move(backend);
    return true;
}

// ---------------------------------------------------------------------------
InferenceResult HeteroRuntime::doGenerate(const std::string& model_path,
                                           const std::string& prompt,
                                           const GenerationOptions& opts,
                                           TokenCallback cb) {
    std::lock_guard<std::mutex> lk(mutex_);
    InferenceResult res;

    if (!initialized_) { res.error = "HeteroRuntime not initialized"; return res; }
    if (prompt.empty()) { res.error = "Prompt is empty"; return res; }

    int max_threads = std::thread::hardware_concurrency();
    int base_gpu_layers = 0;
    bool cpu_only = true;
    resolveBackendConfig(base_gpu_layers, cpu_only);

    ProfileKey key;
    key.hardware_id = hardware_.cpu.model_name;
    key.backend = cpu_only ? ComputeBackend::CPU : ComputeBackend::VULKAN;
    key.workload_type = "llm-inference";
    key.model_name = model_path;

    TuningConfig tcfg = autoTuner_->suggestConfiguration(key, !cpu_only, max_threads);
    GenerationOptions tunedOpts = opts;
    tunedOpts.n_threads = tcfg.n_threads;

    if (!backends_.count(model_path)) {
        auto backend = std::make_unique<LlamaCppBackend>();
        if (!backend->loadModel(model_path, tcfg.n_gpu_layers, tcfg.n_gpu_layers == 0)) {
            res.error = "Failed to load model: " + backend->lastError();
            return res;
        }
        if (!backend->createContext(tunedOpts)) {
            res.error = "Failed to create context: " + backend->lastError();
            return res;
        }
        backends_[model_path] = std::move(backend);
    }

    IInferenceBackend* backend = backends_.at(model_path).get();

    InferenceRequest req;
    req.prompt  = prompt;
    req.options = tunedOpts;
    req.prefer_gpu = !backend->modelInfo().architecture.empty();

    double t0 = nowMs();
    if (cb) res = backend->generateStreaming(req, cb);
    else    res = backend->generate(req);
    double wall_ms = nowMs() - t0;

    res.telemetry.scheduler_reason = "Tuned: layers=" + std::to_string(tcfg.n_gpu_layers) +
                                     " threads=" + std::to_string(tcfg.n_threads);

    // Phase 8: Record telemetry event
    ProfileEvent event;
    event.workload_type = key.workload_type;
    event.model_name = key.model_name;
    event.backend = key.backend;
    event.device_name = res.telemetry.device_name;
    event.start_timestamp_ms = t0;
    event.end_timestamp_ms = t0 + wall_ms;
    event.execution_duration_ms = wall_ms;
    event.prompt_processing_ms = res.telemetry.prompt_eval_ms;
    event.generation_ms = res.telemetry.generation_ms;
    event.ttft_ms = res.telemetry.ttft_ms;
    event.input_tokens = res.telemetry.prompt_tokens;
    event.output_tokens = res.telemetry.output_tokens;
    event.tokens_per_sec = res.telemetry.tokens_per_sec;
    event.scheduler_gpu_layers = tcfg.n_gpu_layers;
    event.scheduler_threads = tcfg.n_threads;
    event.predicted_cost_ms = scheduler_->history().predictDurationMs(key);
    event.actual_cost_ms = wall_ms;
    event.success = res.success;
    event.error_message = res.error;
    
    profiler_->recordEvent(event);
    autoTuner_->recordResult(key, tcfg, wall_ms, res.success);
    profiler_->saveProfile("profiles/performance.json");

    return res;
}

InferenceResult HeteroRuntime::generate(const std::string& model_path,
                                         const std::string& prompt,
                                         const GenerationOptions& opts) {
    return doGenerate(model_path, prompt, opts, nullptr);
}

InferenceResult HeteroRuntime::generateStreaming(const std::string& model_path,
                                                  const std::string& prompt,
                                                  TokenCallback cb,
                                                  const GenerationOptions& opts) {
    return doGenerate(model_path, prompt, opts, cb);
}

// ---------------------------------------------------------------------------
void HeteroRuntime::unloadModel(const std::string& model_path) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = backends_.find(model_path);
    if (it != backends_.end()) {
        it->second->unload();
        backends_.erase(it);
    }
}

void HeteroRuntime::shutdown() {
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto& [path, backend] : backends_) {
        backend->unload();
    }
    backends_.clear();
    scheduler_.reset();
    memMgr_.reset();
    backendMgr_.reset();
    vulkan_.reset();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
std::string HeteroRuntime::diagnosticsReport() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::ostringstream oss;
    oss << "================================================\n";
    oss << "HeteroAccel LLM Runtime — Phase 7\n";
    oss << "================================================\n\n";

    oss << "Hardware\n";
    oss << "  CPU:    " << hardware_.cpu.model_name << "\n";
    oss << "  Cores:  " << hardware_.cpu.physical_cores << " physical / "
        << hardware_.cpu.logical_processors << " logical\n";
    oss << "  RAM:    " << hardware_.memory.total_physical_mb << " MB total  |  "
        << hardware_.memory.available_physical_mb << " MB available\n";

    if (backendMgr_) {
        const auto* vk = backendMgr_->getDevice(ComputeBackend::VULKAN);
        if (vk && vk->is_available) {
            oss << "  Vulkan: " << vk->name << "  ("
                << (vk->memory_capacity / (1024*1024*1024.0)) << " GB)\n";
        } else {
            oss << "  Vulkan: unavailable\n";
        }
        const auto* cu = backendMgr_->getDevice(ComputeBackend::CUDA);
        if (cu && cu->is_available) {
            oss << "  CUDA:   " << cu->name << "\n";
        } else {
            oss << "  CUDA:   unavailable\n";
        }
    }

    oss << "\nLoaded Models: " << backends_.size() << "\n";
    for (const auto& [path, backend] : backends_) {
        auto info = backend->modelInfo();
        oss << "  " << path << "\n";
        oss << "    State:        " << toString(backend->state()) << "\n";
        oss << "    Quant:        " << info.quantization << "\n";
        oss << "    Size:         " << (info.size_bytes / (1024*1024)) << " MB\n";
    }

    oss << "\n================================================\n";
    return oss.str();
}

} // namespace agr
