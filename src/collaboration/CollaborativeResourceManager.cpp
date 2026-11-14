// src/collaboration/CollaborativeResourceManager.cpp
// Shared Resource & Memory-Aware Residency Coordinator implementation.

#include "collaboration/CollaborativeResourceManager.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/HardwareDetector.h"
#include "orchestration/MemoryEstimator.h"
#include "hydroxl/GGUFLoader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define PSAPI_VERSION 2
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace agr {
namespace collaboration {

CollaborativeResourceManager::CollaborativeResourceManager() = default;

CollaborativeResourceManager::~CollaborativeResourceManager() {
    evictAll();
}

ModelMemoryProfile CollaborativeResourceManager::estimateModelMemory(
    const std::string& model_path,
    uint32_t n_ctx) {
    ModelMemoryProfile prof;
    prof.model_path = model_path;
    prof.n_ctx = n_ctx > 0 ? n_ctx : 512;
    prof.colibri_estimated_bytes = 8ULL * 1024ULL * 1024ULL; // 8 MB analyzer workspace

    std::error_code ec;
    if (!model_path.empty() && std::filesystem::exists(model_path, ec) && !ec) {
        prof.file_size_bytes = static_cast<size_t>(std::filesystem::file_size(model_path, ec));
        hydroxl::GGUFLoader loader;
        if (loader.open(model_path)) {
            prof.arch = loader.kvString("general.architecture", "qwen2");
            if (prof.arch.empty()) prof.arch = "qwen2";
            prof.n_layers   = loader.kvUInt32(prof.arch + ".block_count", 24);
            prof.n_embd     = loader.kvUInt32(prof.arch + ".embedding_length", 896);
            prof.n_heads    = loader.kvUInt32(prof.arch + ".attention.head_count", 14);
            prof.n_kv_heads = loader.kvUInt32(prof.arch + ".attention.head_count_kv", 2);
            prof.n_ff       = loader.kvUInt32(prof.arch + ".feed_forward_length", 4864);
            prof.vocab_size = loader.kvUInt32("llm.vocab_size", 0);
            if (prof.vocab_size == 0) {
                prof.vocab_size = loader.kvUInt32(prof.arch + ".vocab_size", 151936);
            }
            // Determine model quantization type from GGUF general.file_type when present,
            // or from transformer block weight tensors for synthetic GGUFs.
            uint32_t ftype = loader.kvUInt32("general.file_type", 0xFFFFFFFFu);
            int mapped_ftype = -1;
            switch (ftype) {
                case 0:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::F32);  break;
                case 1:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::F16);  break;
                case 2:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q4_0); break;
                case 3:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q4_1); break;
                case 7:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q8_0); break;
                case 8:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q5_0); break;
                case 9:  mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q5_1); break;
                case 10: mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q2_K); break;
                case 11:
                case 12:
                case 13: mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q3_K); break;
                case 14:
                case 15: mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q4_K); break;
                case 16:
                case 17: mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q5_K); break;
                case 18: mapped_ftype = static_cast<int>(hydroxl::GGMLType::Q6_K); break;
                default: break;
            }
            if (mapped_ftype >= 0) {
                prof.quant_type = mapped_ftype;
            } else {
                std::map<int, size_t> blk_quant_counts;
                std::map<int, size_t> all_quant_counts;
                for (const auto& t : loader.tensors()) {
                    if (t.type == hydroxl::GGMLType::UNKNOWN || t.type == hydroxl::GGMLType::F32) {
                        continue;
                    }
                    all_quant_counts[static_cast<int>(t.type)] += 1;
                    if (t.name.find("blk.") != std::string::npos &&
                        t.name.find("norm") == std::string::npos &&
                        t.name.find(".bias") == std::string::npos) {
                        blk_quant_counts[static_cast<int>(t.type)] += 1;
                    }
                }
                const auto& target_counts = !blk_quant_counts.empty() ? blk_quant_counts : all_quant_counts;
                if (!target_counts.empty()) {
                    int best_q = prof.quant_type;
                    size_t best_score = 0;
                    for (const auto& kv : target_counts) {
                        if (kv.second > best_score) {
                            best_score = kv.second;
                            best_q = kv.first;
                        }
                    }
                    prof.quant_type = best_q;
                } else {
                    for (const auto& t : loader.tensors()) {
                        if (t.type != hydroxl::GGMLType::UNKNOWN) {
                            prof.quant_type = static_cast<int>(t.type);
                            break;
                        }
                    }
                }
            }
        }
    }

    auto est = orchestration::MemoryEstimator::estimate(
        prof.n_layers, prof.n_embd, prof.n_heads, prof.n_kv_heads,
        prof.n_ff, prof.vocab_size, prof.n_ctx
    );
    prof.hydroxl_estimated_bytes = est.total_estimated;

    // llama.cpp keeps weights quantized in memory + KV cache + compute buffer
    size_t base_weights = prof.file_size_bytes > 0
                              ? prof.file_size_bytes
                              : (est.weights_bytes / 4); // ~Q4 compression ratio
    prof.llamacpp_estimated_bytes = static_cast<size_t>(base_weights * 1.20) +
                                    est.kv_cache_bytes +
                                    (64ULL * 1024ULL * 1024ULL);
    return prof;
}

void CollaborativeResourceManager::configureBudget(size_t budget_bytes) {
    std::lock_guard<std::mutex> lk(mu_);
    configured_budget_bytes_ = budget_bytes;
}

size_t CollaborativeResourceManager::availableRamBytes() const {
    size_t avail = orchestration::MemoryEstimator::available_ram_bytes();
    if (avail == 0) {
        avail = orchestration::HardwareDetector::instance().available_system_ram();
    }
    if (avail == 0) {
        avail = 8ULL * 1024ULL * 1024ULL * 1024ULL; // 8 GB conservative fallback
    }
    return avail;
}

size_t CollaborativeResourceManager::effectiveBudgetBytes() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (configured_budget_bytes_ > 0) {
        return configured_budget_bytes_;
    }
    size_t avail = availableRamBytes();
    return static_cast<size_t>(avail * 0.80);
}

size_t CollaborativeResourceManager::currentProcessResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<size_t>(pmc.WorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            size_t kb = 0;
            iss >> kb;
            return kb * 1024ULL;
        }
    }
    return 0;
#else
    return 0;
#endif
}

size_t CollaborativeResourceManager::peakProcessResidentBytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<size_t>(pmc.PeakWorkingSetSize);
    }
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size_max);
    }
    return 0;
#elif defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            size_t kb = 0;
            iss >> kb;
            return kb * 1024ULL;
        }
    }
    return currentProcessResidentBytes();
#else
    return 0;
#endif
}

bool CollaborativeResourceManager::canKeepConcurrentNeuralEngines(
    const ModelMemoryProfile& profile) const {
    size_t budget = effectiveBudgetBytes();
    size_t concurrent_needed = profile.hydroxl_estimated_bytes +
                               profile.llamacpp_estimated_bytes +
                               profile.colibri_estimated_bytes;
    return concurrent_needed <= budget;
}

std::shared_ptr<orchestration::IInferenceEngine> CollaborativeResourceManager::acquireEngine(
    const std::string& orchestrator_backend_id,
    const std::string& model_path,
    size_t estimated_engine_bytes,
    bool memory_constrained_mode,
    std::string& error_out) {
    std::lock_guard<std::mutex> lk(mu_);

    bool is_heavy = (orchestrator_backend_id.find("colibri") == std::string::npos);

    // 1. Check if already resident with the same model
    auto it = resident_.find(orchestrator_backend_id);
    if (it != resident_.end() && it->second.engine && it->second.engine->is_ready() &&
        it->second.model_path == model_path) {
        return it->second.engine;
    }

    // 2. If memory-constrained mode is active (or adding this engine would exceed
    //    the configured budget), evict other heavy neural engines before loading!
    size_t budget = configured_budget_bytes_ > 0
                        ? configured_budget_bytes_
                        : static_cast<size_t>(availableRamBytes() * 0.80);

    if (is_heavy && (memory_constrained_mode ||
                     (current_resident_bytes_ + estimated_engine_bytes > budget))) {
        for (auto map_it = resident_.begin(); map_it != resident_.end(); ) {
            if (map_it->first != orchestrator_backend_id && map_it->second.is_heavy_neural) {
                if (map_it->second.engine) {
                    map_it->second.engine->shutdown();
                }
                if (current_resident_bytes_ >= map_it->second.estimated_bytes) {
                    current_resident_bytes_ -= map_it->second.estimated_bytes;
                } else {
                    current_resident_bytes_ = 0;
                }
                ++models_evicted_count_;
                map_it = resident_.erase(map_it);
            } else {
                ++map_it;
            }
        }
    }

    // If replacing an existing instance of the same backend_id, clean it up first
    if (it != resident_.end()) {
        if (it->second.engine) {
            it->second.engine->shutdown();
        }
        if (current_resident_bytes_ >= it->second.estimated_bytes) {
            current_resident_bytes_ -= it->second.estimated_bytes;
        } else {
            current_resident_bytes_ = 0;
        }
        resident_.erase(it);
    }

    // 3. Instantiate via EngineOrchestrator
    auto& orch = orchestration::EngineOrchestrator::instance();
    orchestration::EngineOrchestrator::SelectionResult sel;
    sel.selected_backend = orchestrator_backend_id;
    sel.reason = "CollaborativeExecutionOrchestrator task assignment";

    orchestration::EngineOrchestrator::SelectionRequest sreq;
    sreq.model_path = model_path;
    sreq.allow_cpu_fallback = false; // Collaborative orchestrator handles engine/backend fallback explicitly

    auto engine = orch.create_engine(sel, sreq, error_out);
    if (!engine) {
        return nullptr;
    }

    ResidentEntry entry;
    entry.engine = engine;
    entry.model_path = model_path;
    entry.estimated_bytes = estimated_engine_bytes;
    entry.is_heavy_neural = is_heavy;
    resident_[orchestrator_backend_id] = std::move(entry);

    current_resident_bytes_ += estimated_engine_bytes;
    if (current_resident_bytes_ > peak_resident_bytes_) {
        peak_resident_bytes_ = current_resident_bytes_;
    }
    ++models_loaded_count_;

    return engine;
}

void CollaborativeResourceManager::releaseEngine(const std::string& orchestrator_backend_id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = resident_.find(orchestrator_backend_id);
    if (it != resident_.end()) {
        if (it->second.engine) {
            it->second.engine->shutdown();
        }
        if (current_resident_bytes_ >= it->second.estimated_bytes) {
            current_resident_bytes_ -= it->second.estimated_bytes;
        } else {
            current_resident_bytes_ = 0;
        }
        ++models_evicted_count_;
        resident_.erase(it);
    }
}

void CollaborativeResourceManager::evictAll() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : resident_) {
        if (kv.second.engine) {
            kv.second.engine->shutdown();
        }
    }
    resident_.clear();
    current_resident_bytes_ = 0;
}

std::unique_lock<std::mutex> CollaborativeResourceManager::acquireGpuExecutionLock() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        ++gpu_lock_acquisitions_;
    }
    return std::unique_lock<std::mutex>(gpu_exec_mu_);
}

ResourceCoordinationTelemetry CollaborativeResourceManager::telemetrySnapshot(
    const ModelMemoryProfile& profile,
    bool memory_constrained_mode,
    bool concurrent_wave_executed) const {
    std::lock_guard<std::mutex> lk(mu_);
    ResourceCoordinationTelemetry t;
    t.available_ram_bytes = availableRamBytes();
    t.effective_ram_budget_bytes = configured_budget_bytes_ > 0
                                       ? configured_budget_bytes_
                                       : static_cast<size_t>(t.available_ram_bytes * 0.80);
    t.estimated_concurrent_neural_bytes = profile.hydroxl_estimated_bytes +
                                          profile.llamacpp_estimated_bytes +
                                          profile.colibri_estimated_bytes;
    t.peak_resident_bytes = peak_resident_bytes_;
    t.actual_process_rss_bytes = currentProcessResidentBytes();
    t.actual_process_peak_rss_bytes = peakProcessResidentBytes();
    t.memory_constrained_mode = memory_constrained_mode;
    t.concurrent_wave_executed = concurrent_wave_executed;
    t.models_loaded_count = models_loaded_count_;
    t.models_evicted_count = models_evicted_count_;
    t.gpu_lock_acquisitions = gpu_lock_acquisitions_;
    return t;
}

} // namespace collaboration
} // namespace agr
