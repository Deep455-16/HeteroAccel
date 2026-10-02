// src/scheduler/ExecutionPolicyEngine.cpp
#include "scheduler/ExecutionPolicyEngine.h"
#include <algorithm>
#include <cmath>
#include <sstream>

namespace agr {

ExecutionPolicyEngine::ExecutionPolicyEngine(const BackendManager& backends,
                                             MemoryManager& memMgr,
                                             const PerformanceHistory& history)
    : backends_(backends), memMgr_(memMgr), history_(history) {}

int ExecutionPolicyEngine::computeGpuLayers(size_t model_size_bytes,
                                             size_t gpu_mem_bytes,
                                             int total_layers) const {
    if (gpu_mem_bytes == 0 || total_layers == 0) return 0;
    if (model_size_bytes == 0) return total_layers; // unknown -> optimistic
    double fraction = static_cast<double>(gpu_mem_bytes) / static_cast<double>(model_size_bytes);
    fraction = std::min(fraction, 1.0);
    return static_cast<int>(std::floor(fraction * total_layers));
}

int ExecutionPolicyEngine::adaptivePrefetchDistance(PressureLevel pressure, int base_distance) const {
    switch (pressure) {
        case PressureLevel::NORMAL:   return std::max(1, base_distance);
        case PressureLevel::WARNING:  return std::max(1, base_distance / 2);
        case PressureLevel::HIGH:     return 1;
        case PressureLevel::CRITICAL: return 0;
        default:                      return base_distance;
    }
}

PolicyDecision ExecutionPolicyEngine::selectStrategy(const PolicyInput& input) const {
    PolicyDecision decision;
    std::ostringstream reason;

    auto stats = memMgr_.statistics();
    PressureLevel pressure = (stats.gpu_pressure > stats.cpu_pressure)
                           ? stats.gpu_pressure : stats.cpu_pressure;
    // Also honour caller-supplied pressure override (e.g., from tests)
    if (input.memory_pressure > pressure) pressure = input.memory_pressure;

    // 1. CRITICAL -> CPU fallback immediately
    if (pressure == PressureLevel::CRITICAL) {
        decision.strategy          = ExecutionStrategy::CPU_FALLBACK;
        decision.n_gpu_layers      = 0;
        decision.n_threads         = 4;
        decision.streaming_enabled = true;
        decision.prefetch_distance = 0;
        reason << "CRITICAL memory pressure -> CPU_FALLBACK";
        decision.reason = reason.str();
        return decision;
    }

    // 2. No GPU -> CPU fallback
    if (!input.gpu_available) {
        decision.strategy          = ExecutionStrategy::CPU_FALLBACK;
        decision.n_gpu_layers      = 0;
        decision.n_threads         = 8;
        decision.streaming_enabled = true;
        decision.prefetch_distance = adaptivePrefetchDistance(pressure);
        reason << "No GPU available -> CPU_FALLBACK";
        decision.reason = reason.str();
        return decision;
    }

    // 3. Phase 8 history: unreliable GPU -> fallback
    HistoricalStats hist = history_.getStats(input.profile_key);
    bool history_suggests_cpu = (hist.sample_count > 3 && hist.regression_detected &&
                                  hist.reliability() < 0.5);
    if (history_suggests_cpu) {
        decision.strategy          = ExecutionStrategy::CPU_FALLBACK;
        decision.n_gpu_layers      = 0;
        decision.n_threads         = 8;
        decision.streaming_enabled = true;
        decision.prefetch_distance = adaptivePrefetchDistance(pressure);
        reason << "Phase8 history: GPU unreliable (reliability=" << hist.reliability() << ") -> CPU_FALLBACK";
        decision.reason = reason.str();
        return decision;
    }

    size_t gpu_mem  = input.available_gpu_mem;
    size_t model_sz = input.model_size_bytes;

    if (model_sz == 0 || (gpu_mem > 0 && model_sz <= static_cast<size_t>(gpu_mem * kFullResidentThreshold))) {
        decision.strategy          = ExecutionStrategy::FULL_RESIDENT;
        decision.n_gpu_layers      = 99;
        decision.n_threads         = 4;
        decision.streaming_enabled = false;
        decision.prefetch_distance = 0;
        reason << "Model fits fully in GPU mem -> FULL_RESIDENT";
    } else if (gpu_mem > 0 && model_sz > 0 &&
               (static_cast<double>(gpu_mem) / model_sz) >= kPartialResidentMin) {
        int gpu_layers = computeGpuLayers(model_sz, gpu_mem, 32);
        decision.strategy          = (pressure >= PressureLevel::HIGH)
                                   ? ExecutionStrategy::MEMORY_PRESSURE
                                   : ExecutionStrategy::PARTIAL_RESIDENT;
        decision.n_gpu_layers      = gpu_layers;
        decision.n_threads         = 4;
        decision.streaming_enabled = true;
        decision.prefetch_distance = adaptivePrefetchDistance(pressure, 2);
        reason << "Model partially fits (" << gpu_mem << "/" << model_sz << ") -> "
               << toString(decision.strategy) << " layers=" << gpu_layers;
    } else {
        decision.strategy          = ExecutionStrategy::STREAMING;
        decision.n_gpu_layers      = 0;
        decision.n_threads         = 6;
        decision.streaming_enabled = true;
        decision.prefetch_distance = adaptivePrefetchDistance(pressure, 1);
        reason << "Model too large for GPU -> STREAMING";
    }

    // Adjust for workload class
    if (input.wclass == WorkloadClass::BACKGROUND || input.wclass == WorkloadClass::BATCH) {
        decision.n_threads = std::max(2, decision.n_threads - 2);
        reason << " [bg/batch: reduced threads]";
    } else if (input.wclass == WorkloadClass::INTERACTIVE || input.wclass == WorkloadClass::LOW_LATENCY) {
        if (decision.strategy == ExecutionStrategy::PARTIAL_RESIDENT) {
            decision.n_gpu_layers = 99;
        }
        reason << " [interactive: maximise responsiveness]";
    }

    // Multi-workload contention
    if (input.active_workload_count > 1 && pressure >= PressureLevel::WARNING) {
        decision.prefetch_distance = std::max(0, decision.prefetch_distance - 1);
        reason << " [contention: reduced prefetch]";
    }

    decision.reason = reason.str();
    return decision;
}

} // namespace agr
