// src/scheduler/SchedulerTypes.h
// Core data structures for Phase 5 Adaptive Heterogeneous Scheduler.
#pragma once

#include "backend/ComputeDevice.h"
#include "mem/MemoryTypes.h"
#include <string>
#include <functional>
#include <cstdint>

namespace agr {

/// Generic workload abstraction.
struct Workload {
    size_t      id                   = 0;
    std::string name                 = "Unknown";
    size_t      input_bytes          = 0;
    size_t      output_bytes         = 0;
    size_t      compute_ops_estimate = 0; // Relative complexity metric
    
    // Existing data residency. MemoryLocation tracks current placement.
    MemoryBlock input_block;
    MemoryBlock output_block;

    bool        compute_intensive    = true;
    
    // Generic functional callbacks for each backend type.
    // The scheduler decides *where* to execute, the callbacks define *what* executes.
    // In Phase 5, returning true = success, false = failure.
    std::function<bool()> cpu_execute;
    std::function<bool()> vulkan_execute;
    std::function<bool()> cuda_execute;
};

/// The output of the cost model: the decided path.
struct ExecutionPlan {
    ComputeBackend selected_backend   = ComputeBackend::CPU;
    std::string    device_name;
    
    // Theoretical estimates for cost tracking
    double estimated_transfer_ms      = 0.0;
    double estimated_compute_ms       = 0.0;
    double estimated_queue_penalty_ms = 0.0;
    double total_cost_ms              = 0.0;
    
    // What transfers are required before execution?
    bool   requires_input_transfer    = false;
    bool   requires_output_transfer   = false;
};

enum class TaskStatus {
    QUEUED,
    TRANSFERRING,
    RUNNING,
    COMPLETED,
    FAILED
};

/// Actual runtime measurements (Telemetry).
struct TaskResult {
    bool       success     = false;
    TaskStatus status      = TaskStatus::FAILED;
    
    // Timing in milliseconds
    double     wait_ms     = 0.0; // Time spent in queue
    double     transfer_ms = 0.0; // Time spent moving memory
    double     compute_ms  = 0.0; // Time spent executing
    double     total_ms    = 0.0; // Total wall clock
    
    std::string error_message;
};

// Simple opaque handle for async tasks.
using TaskHandle = uint64_t;

} // namespace agr
