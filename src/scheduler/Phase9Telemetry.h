// src/scheduler/Phase9Telemetry.h
// Phase 9: Extended telemetry structures for large-model and multi-workload behavior.
#pragma once
#include "model/ModelTypes.h"
#include "scheduler/WorkloadRegistry.h"
#include <atomic>
#include <cstdint>
#include <string>

namespace agr {

/// Prefetch statistics - tracked atomically
struct PrefetchStats {
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> wasted{0};
    std::atomic<uint64_t> cancelled{0};

    double hitRate() const {
        uint64_t total = requests.load();
        if (total == 0) return 0.0;
        return static_cast<double>(hits.load()) / total;
    }

    void reset() {
        requests.store(0); hits.store(0); misses.store(0);
        wasted.store(0); cancelled.store(0);
    }
};

/// Streaming / loading statistics
struct StreamingStats {
    std::atomic<uint64_t> resources_loaded{0};
    std::atomic<uint64_t> bytes_loaded{0};
    std::atomic<uint64_t> stalls{0};
    std::atomic<uint64_t> total_load_time_ms{0};

    void reset() {
        resources_loaded.store(0); bytes_loaded.store(0);
        stalls.store(0); total_load_time_ms.store(0);
    }
};

/// Per-inference Phase 9 extended telemetry
struct Phase9Telemetry {
    ExecutionStrategy  strategy           = ExecutionStrategy::AUTO;
    WorkloadClass      workload_class     = WorkloadClass::DEFAULT;
    WorkloadPriority   workload_priority  = WorkloadPriority::NORMAL;
    uint64_t           workload_id        = 0;
    bool               cancelled          = false;
    bool               fallback_triggered = false;
    std::string        fallback_reason;
    int                prefetch_distance  = 1;
    uint64_t           prefetch_hits      = 0;
    uint64_t           prefetch_misses    = 0;
    double             streaming_stalls_ms = 0.0;
    int                active_workloads   = 0;
    std::string        strategy_reason;
};

} // namespace agr
