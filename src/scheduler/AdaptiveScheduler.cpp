// src/scheduler/AdaptiveScheduler.cpp
#include "scheduler/AdaptiveScheduler.h"
#include <iostream>

namespace agr {

AdaptiveScheduler::AdaptiveScheduler(BackendManager& backendManager, MemoryManager& memoryManager)
    : backendManager_(backendManager), memoryManager_(memoryManager),
      history_(), costModel_(backendManager_, history_) {}

TaskHandle AdaptiveScheduler::schedule(Workload workload) {
    ExecutionPlan plan = costModel_.evaluate(workload);
    return submitToWorker(workload, plan);
}

TaskHandle AdaptiveScheduler::scheduleToBackend(Workload workload, ComputeBackend backend) {
    const ComputeDevice* device = backendManager_.getDevice(backend);
    ExecutionPlan plan;
    if (device) {
        plan = costModel_.evaluateCandidate(workload, *device);
    } else {
        plan.selected_backend = backend;
        plan.total_cost_ms = 999999.0;
    }
    return submitToWorker(workload, plan);
}

TaskHandle AdaptiveScheduler::submitToWorker(const Workload& workload, const ExecutionPlan& plan) {
    IWorker* worker = getWorker(plan.selected_backend);
    if (!worker) {
        // Fallback to CPU if something goes wrong
        worker = &cpuWorker_;
    }

    TaskHandle worker_handle = worker->submit(workload, plan);

    std::lock_guard<std::mutex> lock(mutex_);
    TaskHandle handle = next_handle_++;
    TaskRecord record;
    record.worker_handle = worker_handle;
    record.backend = plan.selected_backend;
    record.workload = workload;
    record.plan = plan;
    records_[handle] = std::move(record);

    return handle;
}

IWorker* AdaptiveScheduler::getWorker(ComputeBackend backend) {
    switch (backend) {
        case ComputeBackend::CPU:    return &cpuWorker_;
        case ComputeBackend::VULKAN: return &vulkanWorker_;
        case ComputeBackend::CUDA:   return &cudaWorker_;
        default:                     return nullptr;
    }
}

TaskResult AdaptiveScheduler::wait(TaskHandle handle) {
    TaskRecord record;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = records_.find(handle);
        if (it == records_.end()) {
            TaskResult r;
            r.success = false;
            r.error_message = "Invalid TaskHandle";
            return r;
        }
        record = it->second;
    }

    IWorker* worker = getWorker(record.backend);
    TaskResult result = worker->wait(record.worker_handle);

    // After completion, feed back the real telemetry into our cost model!
    if (result.success) {
        // We pretend transfer happened in the compute_ms for this stub, but ideally 
        // the worker would time transfers explicitly.
        history_.recordCompletion(record.backend, record.workload, result);
    }

    // Safety fallback: if Vulkan/CUDA fails, we could try CPU here in a real production system.
    if (!result.success && record.backend != ComputeBackend::CPU) {
        // Log fallback scenario
        // In this basic version, we just return the failure so the caller knows.
    }

    return result;
}

TaskStatus AdaptiveScheduler::status(TaskHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(handle);
    if (it != records_.end()) {
        IWorker* worker = getWorker(it->second.backend);
        return worker->status(it->second.worker_handle);
    }
    return TaskStatus::FAILED;
}

} // namespace agr
