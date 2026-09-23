// src/scheduler/VulkanWorker.h
#pragma once

#include "scheduler/IWorker.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <atomic>

namespace agr {

class VulkanWorker : public IWorker {
public:
    VulkanWorker();
    ~VulkanWorker() override;

    TaskHandle submit(const Workload& workload, const ExecutionPlan& plan) override;
    TaskResult wait(TaskHandle handle) override;
    TaskStatus status(TaskHandle handle) override;

private:
    void workerLoop();

    struct Job {
        TaskHandle    handle;
        Workload      workload;
        ExecutionPlan plan;
        TaskStatus    status = TaskStatus::QUEUED;
        TaskResult    result;
    };

    std::atomic<TaskHandle> next_handle_{1};
    std::mutex              mutex_;
    std::condition_variable cv_queue_;
    std::condition_variable cv_result_;
    
    std::queue<TaskHandle>       queue_;
    std::unordered_map<TaskHandle, Job> jobs_;
    
    bool        stop_ = false;
    std::thread thread_;
};

} // namespace agr
