// src/scheduler/WorkloadRegistry.cpp
#include "scheduler/WorkloadRegistry.h"
#include <algorithm>

namespace agr {

uint64_t WorkloadRegistry::registerWorkload(const std::string& name,
                                             const std::string& model_path,
                                             WorkloadPriority priority,
                                             WorkloadClass wclass) {
    std::lock_guard<std::mutex> lk(mutex_);
    uint64_t id = next_id_++;
    auto& e = entries_[id];
    e.id         = id;
    e.name       = name;
    e.model_path = model_path;
    e.priority   = priority;
    e.wclass     = wclass;
    e.state      = WorkloadState::QUEUED;
    e.queued_at  = std::chrono::steady_clock::now();
    e.cancel_requested.store(false, std::memory_order_relaxed);
    return id;
}

std::atomic<bool>* WorkloadRegistry::cancelFlag(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return nullptr;
    return &it->second.cancel_requested;
}

bool WorkloadRegistry::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    it->second.cancel_requested.store(true, std::memory_order_release);
    if (it->second.state == WorkloadState::QUEUED) {
        it->second.state = WorkloadState::CANCELLED;
        it->second.finished_at = std::chrono::steady_clock::now();
    }
    return true;
}

void WorkloadRegistry::setRunning(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return;
    it->second.state      = WorkloadState::RUNNING;
    it->second.started_at = std::chrono::steady_clock::now();
}

void WorkloadRegistry::setCompleted(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return;
    it->second.state       = WorkloadState::COMPLETED;
    it->second.finished_at = std::chrono::steady_clock::now();
}

void WorkloadRegistry::setFailed(uint64_t id, const std::string& error) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return;
    it->second.state       = WorkloadState::FAILED;
    it->second.error       = error;
    it->second.finished_at = std::chrono::steady_clock::now();
}

void WorkloadRegistry::setCancelled(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return;
    it->second.state       = WorkloadState::CANCELLED;
    it->second.finished_at = std::chrono::steady_clock::now();
}

WorkloadState WorkloadRegistry::getState(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return WorkloadState::FAILED;
    return it->second.state;
}

bool WorkloadRegistry::isCancelled(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return false;
    return it->second.cancel_requested.load(std::memory_order_acquire);
}

WorkloadClass WorkloadRegistry::getClass(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return WorkloadClass::DEFAULT;
    return it->second.wclass;
}

std::vector<WorkloadEntry*> WorkloadRegistry::activeWorkloadsSorted() {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<WorkloadEntry*> result;
    for (auto& [id, entry] : entries_) {
        if (entry.state == WorkloadState::QUEUED || entry.state == WorkloadState::RUNNING) {
            result.push_back(&entry);
        }
    }
    std::sort(result.begin(), result.end(), [](const WorkloadEntry* a, const WorkloadEntry* b) {
        return static_cast<int>(a->priority) > static_cast<int>(b->priority);
    });
    return result;
}

std::vector<WorkloadEntry> WorkloadRegistry::snapshot() const {
    std::lock_guard<std::mutex> lk(mutex_);
    std::vector<WorkloadEntry> result;
    result.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        result.push_back(entry);
    }
    return result;
}

void WorkloadRegistry::pruneOld(double max_age_sec) {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lk(mutex_);
    for (auto it = entries_.begin(); it != entries_.end(); ) {
        auto& e = it->second;
        bool terminal = (e.state == WorkloadState::COMPLETED ||
                         e.state == WorkloadState::FAILED ||
                         e.state == WorkloadState::CANCELLED);
        if (terminal) {
            double age = std::chrono::duration<double>(now - e.finished_at).count();
            if (age > max_age_sec) {
                it = entries_.erase(it);
                continue;
            }
        }
        ++it;
    }
}

size_t WorkloadRegistry::totalCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return entries_.size();
}

size_t WorkloadRegistry::activeCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    size_t n = 0;
    for (const auto& [id, e] : entries_) {
        if (e.state == WorkloadState::QUEUED || e.state == WorkloadState::RUNNING) ++n;
    }
    return n;
}

} // namespace agr
