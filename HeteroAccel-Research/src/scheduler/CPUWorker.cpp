// src/scheduler/CPUWorker.cpp
#include "scheduler/CPUWorker.h"
#include <chrono>

namespace agr {

CPUWorker::CPUWorker() {
    thread_ = std::thread(&CPUWorker::workerLoop, this);
}

CPUWorker::~CPUWorker() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_queue_.notify_one();
    if (thread_.joinable()) thread_.join();
}

TaskHandle CPUWorker::submit(const Workload& workload, const ExecutionPlan& plan) {
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

TaskResult CPUWorker::wait(TaskHandle handle) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_result_.wait(lock, [this, handle]() {
        auto it = jobs_.find(handle);
        return it != jobs_.end() && 
               (it->second.status == TaskStatus::COMPLETED || it->second.status == TaskStatus::FAILED);
    });
    
    return jobs_[handle].result;
}

TaskStatus CPUWorker::status(TaskHandle handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(handle);
    if (it != jobs_.end()) {
        return it->second.status;
    }
    return TaskStatus::FAILED;
}

void CPUWorker::workerLoop() {
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
        
        // Execute outside lock
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = false;
        std::string error_msg;
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& job = jobs_[handle];
            if (job.workload.cpu_execute) {
                // We unlock during execution to allow parallel submissions
            } else {
                success = false;
                error_msg = "CPU execute callback not provided";
            }
        }
        
        // Retrieve callback safely
        std::function<bool()> execute_cb;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            execute_cb = jobs_[handle].workload.cpu_execute;
        }

        if (execute_cb) {
            try {
                success = execute_cb();
            } catch (const std::exception& e) {
                success = false;
                error_msg = e.what();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double compute_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        // Update job result
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& job = jobs_[handle];
            job.status = success ? TaskStatus::COMPLETED : TaskStatus::FAILED;
            job.result.success = success;
            job.result.status = job.status;
            job.result.compute_ms = compute_ms;
            job.result.total_ms = compute_ms; // Not tracking wait time currently
            job.result.error_message = error_msg;
        }
        cv_result_.notify_all();
    }
}

} // namespace agr
