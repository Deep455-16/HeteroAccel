#include "inference/HeteroRuntime.h"
#include "inference/LlamaCppBackend.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
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

    // Phase 9: Workload registry and execution policy engine
    workloadRegistry_ = std::make_unique<WorkloadRegistry>();
    policyEngine_ = std::make_unique<ExecutionPolicyEngine>(*backendMgr_, *memMgr_, scheduler_->history());

    initialized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
TuningConfig HeteroRuntime::resolveAndApplyConfig(const std::string& model_path, int max_threads, WorkloadClass wclass) {
    size_t actual_model_size = 0;
    if (std::filesystem::exists(model_path)) {
        actual_model_size = std::filesystem::file_size(model_path);
    }
    
    bool vulkan_avail = backendMgr_ && backendMgr_->isBackendAvailable(ComputeBackend::VULKAN);
    bool cuda_avail   = backendMgr_ && backendMgr_->isBackendAvailable(ComputeBackend::CUDA);
    bool gpu_avail    = vulkan_avail || cuda_avail;

    ProfileKey key;
    key.hardware_id = hardware_.cpu.model_name;
    key.backend = gpu_avail ? ComputeBackend::VULKAN : ComputeBackend::CPU;
    key.workload_type = "llm-inference";
    key.model_name = model_path;

    int max_gpu_layers = gpu_avail ? 99 : 0;
    
    if (policyEngine_) {
        PolicyInput pInput;
        pInput.model_size_bytes = actual_model_size;
        pInput.gpu_available = gpu_avail;
        pInput.profile_key   = key;
        pInput.wclass        = wclass;
        pInput.memory_pressure = memMgr_ ? memMgr_->statistics().gpu_pressure : PressureLevel::NORMAL;
        if (backendMgr_) {
            const auto* vkDev = backendMgr_->getDevice(ComputeBackend::VULKAN);
            if (vkDev && vkDev->is_available) {
                pInput.available_gpu_mem = vkDev->memory_available;
            }
        }
        pInput.active_workload_count = workloadRegistry_ ? static_cast<int>(workloadRegistry_->activeCount()) : 0;
        
        PolicyDecision dec = policyEngine_->selectStrategy(pInput);
        max_gpu_layers = dec.n_gpu_layers;
    }

    key.backend = (max_gpu_layers == 0) ? ComputeBackend::CPU : ComputeBackend::VULKAN;
    
    return autoTuner_->suggestConfiguration(key, max_gpu_layers, max_threads);
}

// ---------------------------------------------------------------------------
bool HeteroRuntime::loadModel(const std::string& model_path,
                               const GenerationOptions& opts) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!initialized_) return false;

    int max_threads = std::thread::hardware_concurrency();
    TuningConfig tcfg = resolveAndApplyConfig(model_path, max_threads, WorkloadClass::DEFAULT);
    
    // Check if loaded config differs significantly (e.g. CPU fallback)
    if (backends_.count(model_path)) {
        TuningConfig loaded = loaded_configs_[model_path];
        if ((loaded.n_gpu_layers > 0 && tcfg.n_gpu_layers == 0) || 
            (loaded.n_gpu_layers == 0 && tcfg.n_gpu_layers > 0)) {
            // Unload to apply new placement constraint
            backends_.erase(model_path);
            loaded_configs_.erase(model_path);
        } else {
            return true;
        }
    }

    GenerationOptions tunedOpts = opts;
    tunedOpts.n_threads = tcfg.n_threads;

    auto backend = std::make_shared<LlamaCppBackend>();
    if (!backend->loadModel(model_path, tcfg.n_gpu_layers, tcfg.n_gpu_layers == 0)) return false;
    if (!backend->createContext(tunedOpts)) return false;

    backends_[model_path] = backend;
    loaded_configs_[model_path] = tcfg;
    return true;
}

// ---------------------------------------------------------------------------
InferenceResult HeteroRuntime::doGenerate(const std::string& model_path,
                                           const std::string& prompt,
                                           const GenerationOptions& opts,
                                           TokenCallback cb,
                                           uint64_t workload_id) {
    std::shared_ptr<IInferenceBackend> backend;
    TuningConfig tcfg;
    ProfileKey key;
    double t0, wall_ms = 0;
    
    {
        std::lock_guard<std::mutex> lk(mutex_);
        InferenceResult res;
        if (!initialized_) { res.error = "HeteroRuntime not initialized"; return res; }
        if (prompt.empty()) { res.error = "Prompt is empty"; return res; }

        if (workload_id != 0 && workloadRegistry_) {
            if (workloadRegistry_->isCancelled(workload_id)) {
                res.error = "Workload cancelled before execution";
                return res;
            }
            workloadRegistry_->setRunning(workload_id);
        }

        WorkloadClass wclass = WorkloadClass::DEFAULT;
        if (workload_id != 0 && workloadRegistry_) {
            wclass = workloadRegistry_->getClass(workload_id);
        }
        
        int max_threads = std::thread::hardware_concurrency();
        tcfg = resolveAndApplyConfig(model_path, max_threads, wclass);
        
        key.hardware_id = hardware_.cpu.model_name;
        key.backend = (tcfg.n_gpu_layers == 0) ? ComputeBackend::CPU : ComputeBackend::VULKAN;
        key.workload_type = "llm-inference";
        key.model_name = model_path;

        if (backends_.count(model_path)) {
            TuningConfig loaded = loaded_configs_[model_path];
            if ((loaded.n_gpu_layers > 0 && tcfg.n_gpu_layers == 0) || 
                (loaded.n_gpu_layers == 0 && tcfg.n_gpu_layers > 0)) {
                backends_.erase(model_path);
                loaded_configs_.erase(model_path);
            }
        }
        
        if (!backends_.count(model_path)) {
            GenerationOptions tunedOpts = opts;
            tunedOpts.n_threads = tcfg.n_threads;
            auto new_backend = std::make_shared<LlamaCppBackend>();
            if (!new_backend->loadModel(model_path, tcfg.n_gpu_layers, tcfg.n_gpu_layers == 0)) {
                res.error = "Failed to load model: " + new_backend->lastError();
                return res;
            }
            if (!new_backend->createContext(tunedOpts)) {
                res.error = "Failed to create context: " + new_backend->lastError();
                return res;
            }
            backends_[model_path] = new_backend;
            loaded_configs_[model_path] = tcfg;
        }

        backend = backends_.at(model_path);
    } // Unlock global mutex before inference!

    InferenceRequest req;
    req.prompt  = prompt;
    req.options = opts;
    req.options.n_threads = tcfg.n_threads;
    req.prefer_gpu = !backend->modelInfo().architecture.empty();
    
    if (workload_id != 0 && workloadRegistry_) {
        req.cancel_flag = workloadRegistry_->cancelFlag(workload_id);
    }

    t0 = nowMs();
    InferenceResult res;
    if (cb) res = backend->generateStreaming(req, cb);
    else    res = backend->generate(req);
    wall_ms = nowMs() - t0;

    {
        std::lock_guard<std::mutex> lk(mutex_);
        res.telemetry.scheduler_reason = "Tuned: layers=" + std::to_string(tcfg.n_gpu_layers) +
                                         " threads=" + std::to_string(tcfg.n_threads);
        
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
        if (scheduler_) {
            event.predicted_cost_ms = scheduler_->history().predictDurationMs(key);
        }
        event.actual_cost_ms = wall_ms;
        event.success = res.success;
        event.error_message = res.error;
        
        profiler_->recordEvent(event);
        autoTuner_->recordResult(key, tcfg, wall_ms, res.success);
        profiler_->saveProfile("profiles/performance.json");

        if (workload_id != 0 && workloadRegistry_) {
            if (res.success) {
                workloadRegistry_->setCompleted(workload_id);
            } else if (res.error == "Generation cancelled" || res.error == "Cancelled before evaluation") {
                // Keep it cancelled
            } else {
                workloadRegistry_->setFailed(workload_id, res.error);
            }
        }
    }
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

// ---------------------------------------------------------------------------
// Phase 9: Workload management
// ---------------------------------------------------------------------------
uint64_t HeteroRuntime::registerWorkload(const std::string& name,
                                          const std::string& model_path,
                                          WorkloadPriority priority,
                                          WorkloadClass wclass) {
    if (!workloadRegistry_) return 0;
    return workloadRegistry_->registerWorkload(name, model_path, priority, wclass);
}

bool HeteroRuntime::cancelWorkload(uint64_t workload_id) {
    if (!workloadRegistry_) return false;
    return workloadRegistry_->cancel(workload_id);
}

std::vector<WorkloadEntry> HeteroRuntime::workloadSnapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!workloadRegistry_) return {};
    return workloadRegistry_->snapshot();
}

InferenceResult HeteroRuntime::generateWorkload(const std::string& model_path,
                                                 const std::string& prompt,
                                                 const GenerationOptions& opts,
                                                 WorkloadPriority priority,
                                                 WorkloadClass wclass) {
    if (!workloadRegistry_) {
        return generate(model_path, prompt, opts);
    }
    uint64_t wid = workloadRegistry_->registerWorkload("inference", model_path, priority, wclass);
    return doGenerate(model_path, prompt, opts, nullptr, wid);
}

} // namespace agr
