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

    initialized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
void HeteroRuntime::resolveBackendConfig(int& out_gpu_layers, bool& out_cpu_only) const {
    // Ask Phase 5 scheduler for the best backend.
    // We represent inference as a generic Workload with memory requirements.
    out_gpu_layers = 0;
    out_cpu_only   = true;

    if (!backendMgr_) return;

    bool vulkan_avail = backendMgr_->isBackendAvailable(ComputeBackend::VULKAN);
    bool cuda_avail   = backendMgr_->isBackendAvailable(ComputeBackend::CUDA);

    // Check memory availability to decide if GPU offload is safe.
    // We use the Phase 4 memory stats.
    MemoryStats memStats;
    if (memMgr_) memStats = memMgr_->statistics();

    bool gpu_memory_ok = (memStats.gpu_pressure != PressureLevel::CRITICAL);

    if ((cuda_avail || vulkan_avail) && gpu_memory_ok) {
        out_gpu_layers = 99; // offload as many layers as fit — llama.cpp decides
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

    // If already loaded, skip
    if (backends_.count(model_path)) return true;

    int  n_gpu_layers;
    bool cpu_only;
    resolveBackendConfig(n_gpu_layers, cpu_only);

    auto backend = std::make_unique<LlamaCppBackend>();
    if (!backend->loadModel(model_path, n_gpu_layers, cpu_only)) return false;
    if (!backend->createContext(opts)) return false;

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

    if (!initialized_) {
        res.error = "HeteroRuntime not initialized";
        return res;
    }
    if (prompt.empty()) {
        res.error = "Prompt is empty";
        return res;
    }

    // Auto-load if not yet loaded
    if (!backends_.count(model_path)) {
        int  n_gpu_layers;
        bool cpu_only;
        resolveBackendConfig(n_gpu_layers, cpu_only);

        auto backend = std::make_unique<LlamaCppBackend>();
        if (!backend->loadModel(model_path, n_gpu_layers, cpu_only)) {
            res.error = "Failed to load model: " + backend->lastError();
            return res;
        }
        if (!backend->createContext(opts)) {
            res.error = "Failed to create context: " + backend->lastError();
            return res;
        }
        backends_[model_path] = std::move(backend);
    }

    IInferenceBackend* backend = backends_.at(model_path).get();

    // Build inference request
    InferenceRequest req;
    req.prompt  = prompt;
    req.options = opts;
    req.prefer_gpu = !backend->modelInfo().architecture.empty();

    // Determine backend info for scheduler annotation
    bool vulkan_avail = backendMgr_->isBackendAvailable(ComputeBackend::VULKAN);
    bool cuda_avail   = backendMgr_->isBackendAvailable(ComputeBackend::CUDA);

    // Run inference
    double t0 = nowMs();
    if (cb) {
        res = backend->generateStreaming(req, cb);
    } else {
        res = backend->generate(req);
    }
    double wall_ms = nowMs() - t0;

    // Annotate scheduler decision in telemetry
    if (cuda_avail) {
        res.telemetry.scheduler_reason = "CUDA available → offloaded";
    } else if (vulkan_avail) {
        res.telemetry.scheduler_reason = "Vulkan available → offloaded";
    } else {
        res.telemetry.scheduler_reason = "CPU-only (no GPU available)";
    }

    // Feed result back into Phase 5 PerformanceHistory
    if (scheduler_ && res.success) {
        Workload w;
        w.name = "llm-inference";
        w.compute_ops_estimate = static_cast<size_t>(res.telemetry.output_tokens) * 1000;
        // We submit a dummy CPU task to record timing in PerformanceHistory
        w.cpu_execute = [&]() -> bool { return true; };
        // Don't block — just record directly via a completed TaskResult
        TaskResult tr;
        tr.success    = true;
        tr.status     = TaskStatus::COMPLETED;
        tr.compute_ms = res.telemetry.generation_ms;
        tr.total_ms   = wall_ms;
        // PerformanceHistory doesn't have a direct record() API yet;
        // this will be wired in Phase 8. For Phase 7 we document this
        // as a known limitation: scheduler feedback loop is partially wired.
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

} // namespace agr
