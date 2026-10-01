// src/scheduler/VulkanWorker.cpp
#include "scheduler/VulkanWorker.h"
#include <chrono>

namespace agr {

VulkanWorker::VulkanWorker() {
    thread_ = std::thread(&VulkanWorker::workerLoop, this);
}

VulkanWorker::~VulkanWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_queue_.notify_one();
    if (thread_.joinable()) thread_.join();
}

TaskHandle VulkanWorker::submit(const Workload& workload, const ExecutionPlan& plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    TaskHandle handle = next_handle_++;
    
    Job job;
    job.handle   = handle;
    job.workload = workload;
    job.plan     = plan;
    job.status   = TaskStatus::QUEUED;
    
    jobs_[handle] = std::move(job);
    queue_.push(handle);
    
    cv_queue_.notify_one();
    return handle;
}

TaskResult VulkanWorker::wait(TaskHandle handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_result_.wait(lock, [this, handle]() {
        auto it = jobs_.find(handle);
        return it != jobs_.end() && 
               (it->second.status == TaskStatus::COMPLETED || it->second.status == TaskStatus::FAILED);
    });
    
    return jobs_[handle].result;
}

TaskStatus VulkanWorker::status(TaskHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(handle);
    if (it != jobs_.end()) return it->second.status;
    return TaskStatus::FAILED;
}

void VulkanWorker::workerLoop() {
    while (true) {
        TaskHandle handle = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_queue_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
            
            if (stop_ && queue_.empty()) return;
            
            handle = queue_.front();
            queue_.pop();
            jobs_[handle].status = TaskStatus::RUNNING;
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        bool success = false;
        std::string error_msg;
        
        std::function<bool()> execute_cb;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            execute_cb = jobs_[handle].workload.vulkan_execute;
        }

        if (execute_cb) {
            try {
                success = execute_cb();
            } catch (const std::exception& e) {
                success = false;
                error_msg = e.what();
            }
        } else {
            success = false;
            error_msg = "Vulkan execute callback not provided";
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double compute_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& job = jobs_[handle];
            job.status = success ? TaskStatus::COMPLETED : TaskStatus::FAILED;
            job.result.success = success;
            job.result.status = job.status;
            job.result.compute_ms = compute_ms;
            job.result.total_ms = compute_ms; 
            job.result.error_message = error_msg;
        }
        cv_result_.notify_all();
    }
}

} // namespace agr
