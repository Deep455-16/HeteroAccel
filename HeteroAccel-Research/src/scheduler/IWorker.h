// src/scheduler/IWorker.h
// Abstract interface for backend execution workers.
#pragma once

#include "scheduler/SchedulerTypes.h"

namespace agr {

class IWorker {
public:
    virtual ~IWorker() = default;

    /// Submit a workload for execution. Returns an internal task ID for this worker.
    virtual TaskHandle submit(const Workload& workload, const ExecutionPlan& plan) = 0;

    /// Block until the task completes and return the result.
    virtual TaskResult wait(TaskHandle handle) = 0;

    /// Check current status without blocking.
    virtual TaskStatus status(TaskHandle handle) = 0;
};

} // namespace agr
