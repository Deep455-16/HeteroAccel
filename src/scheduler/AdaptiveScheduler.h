// src/scheduler/AdaptiveScheduler.h
// The orchestrator of Phase 5.
#pragma once

#include "scheduler/SchedulerTypes.h"
#include "scheduler/CostModel.h"
#include "scheduler/PerformanceHistory.h"
#include "scheduler/CPUWorker.h"
#include "scheduler/VulkanWorker.h"
#include "scheduler/CUDAWorker.h"
#include "mem/MemoryManager.h"
#include "backend/BackendManager.h"

#include <memory>
#include <unordered_map>
#include <mutex>

namespace agr {

class AdaptiveScheduler {
public:
    AdaptiveScheduler(BackendManager& backendManager, MemoryManager& memoryManager);

    /// Schedules a workload automatically based on capabilities, cost, and memory state.
    /// Does not block for execution.
    TaskHandle schedule(Workload workload);

    /// Waits for a scheduled task to complete and returns the execution result.
    TaskResult wait(TaskHandle handle);

    /// Check the status of a scheduled task.
    TaskStatus status(TaskHandle handle);

    /// Allows bypassing the cost model for testing or manual overrides.
    TaskHandle scheduleToBackend(Workload workload, ComputeBackend backend);
    
    const PerformanceHistory& history() const { return history_; }

private:
    TaskHandle submitToWorker(const Workload& workload, const ExecutionPlan& plan);
    IWorker*   getWorker(ComputeBackend backend);

    BackendManager&    backendManager_;
    MemoryManager&     memoryManager_;
    PerformanceHistory history_;
    CostModel          costModel_;

    CPUWorker    cpuWorker_;
    VulkanWorker vulkanWorker_;
    CUDAWorker   cudaWorker_;

    struct TaskRecord {
        TaskHandle    worker_handle;
        ComputeBackend backend;
        Workload      workload;
        ExecutionPlan plan;
    };

    std::mutex mutex_;
    uint64_t   next_handle_{1};
    std::unordered_map<TaskHandle, TaskRecord> records_;
};

} // namespace agr
