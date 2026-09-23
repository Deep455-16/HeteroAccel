// src/scheduler/CostModel.cpp
#include "scheduler/CostModel.h"
#include <limits>
#include <algorithm>
#include <iostream>

namespace agr {

CostModel::CostModel(const BackendManager& backendManager, const PerformanceHistory& history)
    : backendManager_(backendManager), history_(history) {}

ExecutionPlan CostModel::evaluate(const Workload& workload) const {
    const auto& devices = backendManager_.allDevices();
    
    ExecutionPlan bestPlan;
    bestPlan.total_cost_ms = std::numeric_limits<double>::max();

    for (const auto& dev : devices) {
        if (!dev.is_available) continue;

        ExecutionPlan candidate = evaluateCandidate(workload, dev);
        if (candidate.total_cost_ms < bestPlan.total_cost_ms) {
            bestPlan = candidate;
        }
    }

    // Safety fallback: if all devices fail (e.g. OOM), force CPU
    if (bestPlan.total_cost_ms == std::numeric_limits<double>::max()) {
        const ComputeDevice* cpu = backendManager_.getDevice(ComputeBackend::CPU);
        if (cpu) {
            bestPlan = evaluateCandidate(workload, *cpu);
            // Overwrite costs to ensure it's picked as fallback
            bestPlan.total_cost_ms = 999999.0;
        }
    }

    return bestPlan;
}

ExecutionPlan CostModel::evaluateCandidate(const Workload& workload, const ComputeDevice& device) const {
    ExecutionPlan plan;
    plan.selected_backend = device.backend;
    plan.device_name      = device.name;
    plan.total_cost_ms    = std::numeric_limits<double>::max(); // Start infeasible

    // 1. Feasibility Check: Memory Capacity
    // In Phase 5, if it doesn't fit, it doesn't fit.
    size_t required_memory = workload.input_bytes + workload.output_bytes;
    if (required_memory > 0 && device.memory_available > 0 && device.memory_available < required_memory) {
        return plan; // Infeasible
    }

    // 2. Feasibility Check: Compute intensive prefers GPU, but CPU can do it.
    // However, if the workload lacks a callback for this backend, it's infeasible.
    if (device.backend == ComputeBackend::CPU && !workload.cpu_execute) return plan;
    if (device.backend == ComputeBackend::VULKAN && !workload.vulkan_execute) return plan;
    if (device.backend == ComputeBackend::CUDA && !workload.cuda_execute) return plan;

    // 3. Estimate Compute Cost
    double ops_per_ms = history_.getEstimatedOpsPerMs(device.backend);
    plan.estimated_compute_ms = static_cast<double>(workload.compute_ops_estimate) / ops_per_ms;

    // 4. Estimate Transfer Cost (Residency Bias)
    double bw_per_ms = history_.getEstimatedBandwidthBytesPerMs(device.backend);
    plan.estimated_transfer_ms = 0.0;
    
    // Determine target location for this backend
    MemoryLocation targetLoc = MemoryLocation::CPU;
    if (device.backend == ComputeBackend::VULKAN || device.backend == ComputeBackend::CUDA) {
        targetLoc = MemoryLocation::ACCELERATOR; 
        // Note: For Phase 5, we assume ACCELERATOR means the primary GPU.
        // A multi-GPU setup would refine this.
    }

    if (workload.input_block.isValid() && workload.input_block.location != targetLoc) {
        plan.estimated_transfer_ms += static_cast<double>(workload.input_bytes) / bw_per_ms;
        plan.requires_input_transfer = true;
    }
    
    if (workload.output_block.isValid() && workload.output_block.location != targetLoc) {
        // Output transfer cost is debatable if we leave it resident, but for scheduling
        // a pure generic workload, if the caller requires it back at CPU, we add penalty.
        // For now, if the output location is pre-allocated elsewhere, we add transfer cost.
        plan.estimated_transfer_ms += static_cast<double>(workload.output_bytes) / bw_per_ms;
        plan.requires_output_transfer = true;
    }

    // 5. Total Cost Formula
    // We add a switching/queue penalty. A crude proxy for load.
    plan.estimated_queue_penalty_ms = (device.backend == ComputeBackend::CPU) ? 2.0 : 10.0; 
    
    plan.total_cost_ms = plan.estimated_compute_ms + plan.estimated_transfer_ms + plan.estimated_queue_penalty_ms;

    // Artificial CPU penalty for highly compute intensive tasks to encourage GPU usage
    // This implements the heuristic: "Don't blindly use CPU just because transfer is 1ms faster"
    if (device.backend == ComputeBackend::CPU && workload.compute_intensive) {
        plan.total_cost_ms *= 1.5;
    }

    return plan;
}

} // namespace agr
