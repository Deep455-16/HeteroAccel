// src/mem/ResidencyManager.cpp
#include "mem/ResidencyManager.h"

namespace agr {

void ResidencyManager::registerBlock(uint64_t id, Residency initialResidency) {
    std::lock_guard<std::mutex> lk(mutex_);
    records_[id] = {initialResidency, std::chrono::steady_clock::now(), 0};
}

void ResidencyManager::unregisterBlock(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    records_.erase(id);
}

void ResidencyManager::promote(uint64_t id, MemoryLocation target) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(id);
    if (it == records_.end()) return;
    
    if (target == MemoryLocation::GPU) {
        if (it->second.residency == Residency::CPU) {
            it->second.residency = Residency::CPU_AND_GPU;
        } else if (it->second.residency == Residency::DISK) {
            it->second.residency = Residency::GPU; // Direct to GPU conceptually
        }
    } else if (target == MemoryLocation::CPU) {
        if (it->second.residency == Residency::DISK) {
            it->second.residency = Residency::CPU;
        } else if (it->second.residency == Residency::GPU) {
            it->second.residency = Residency::CPU_AND_GPU;
        }
    }
}

void ResidencyManager::demote(uint64_t id, MemoryLocation from) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(id);
    if (it == records_.end()) return;

    if (from == MemoryLocation::GPU) {
        if (it->second.residency == Residency::CPU_AND_GPU) {
            it->second.residency = Residency::CPU;
        } else if (it->second.residency == Residency::GPU) {
            it->second.residency = Residency::DISK; // Evicted from system entirely
        }
    } else if (from == MemoryLocation::CPU) {
        if (it->second.residency == Residency::CPU_AND_GPU) {
            it->second.residency = Residency::GPU;
        } else if (it->second.residency == Residency::CPU) {
            it->second.residency = Residency::DISK;
        }
    }
}

Residency ResidencyManager::getResidency(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(id);
    return (it != records_.end()) ? it->second.residency : Residency::DISK;
}

void ResidencyManager::touch(uint64_t id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(id);
    if (it != records_.end()) {
        it->second.last_accessed = std::chrono::steady_clock::now();
        it->second.access_count++;
    }
}

bool ResidencyManager::getBlockInfo(uint64_t id, std::chrono::steady_clock::time_point& last_accessed, uint64_t& access_count) const {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = records_.find(id);
    if (it == records_.end()) return false;
    last_accessed = it->second.last_accessed;
    access_count = it->second.access_count;
    return true;
}

} // namespace agr
