// src/model/ModelResidencyManager.cpp
// Phase 14: Real model-data residency — file → RAM → accelerator.
//
// Design decisions:
//  - All state mutations are protected by mutex_.
//  - RESIDENT_* is only set AFTER the actual copy/upload has completed.
//  - Pinned or actively-transferring regions are never evicted.
//  - LRU is used to pick eviction candidates.
//  - Telemetry is accumulated atomically under the mutex; read under mutex for snapshot.

#include "model/ModelResidencyManager.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <new>

namespace agr {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ModelResidencyManager::ModelResidencyManager(IModelDataSource& source,
                                             ModelResidencyBudget budget,
                                             IAcceleratorMemory* accelerator)
    : source_(source), budget_(budget), accelerator_(accelerator) {}

ModelResidencyManager::~ModelResidencyManager() {
    releaseAll();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool ModelResidencyManager::ensureResident(const ModelRegion& region,
                                            ModelResidencyTarget target,
                                            std::atomic<bool>* cancel) {
    if (!region.valid()) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = "Invalid region (zero size)";
        return false;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    Record& rec = getOrCreateLocked(region);

    // Already at or above the requested target?
    if (target == ModelResidencyTarget::RAM) {
        if (rec.state == ModelResidencyState::RESIDENT_IN_RAM ||
            rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
            rec.last_used = std::chrono::steady_clock::now();
            return true;
        }
    } else {
        if (rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
            rec.last_used = std::chrono::steady_clock::now();
            return true;
        }
    }

    // If already loading, bail — caller should retry.
    if (rec.transferring) {
        last_error_ = "Region is already being transferred";
        return false;
    }

    // Step 1: ensure in RAM first.
    if (rec.state != ModelResidencyState::RESIDENT_IN_RAM &&
        rec.state != ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
        if (!loadToRamLocked(rec, cancel)) return false;
    }

    // Step 2: if accelerator requested, upload.
    if (target == ModelResidencyTarget::ACCELERATOR) {
        if (rec.state != ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
            if (!uploadLocked(rec, cancel)) return false;
        }
    }

    rec.last_used = std::chrono::steady_clock::now();
    return true;
}

bool ModelResidencyManager::prefetch(const ModelRegion& region,
                                      ModelResidencyTarget target,
                                      std::atomic<bool>* cancel) {
    if (!prefetch_enabled_) return false;
    if (!region.valid()) return false;

    std::unique_lock<std::mutex> lock(mutex_);
    Record& rec = getOrCreateLocked(region);

    // Already resident — count as hit.
    if ((target == ModelResidencyTarget::RAM &&
         (rec.state == ModelResidencyState::RESIDENT_IN_RAM ||
          rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR)) ||
        (target == ModelResidencyTarget::ACCELERATOR &&
         rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR)) {
        telemetry_.prefetch_hits++;
        return true;
    }

    telemetry_.prefetch_misses++;
    telemetry_.prefetch_count++;

    if (rec.transferring) return false;

    if (rec.state != ModelResidencyState::RESIDENT_IN_RAM &&
        rec.state != ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
        if (!loadToRamLocked(rec, cancel)) return false;
    }
    if (target == ModelResidencyTarget::ACCELERATOR) {
        if (rec.state != ModelResidencyState::RESIDENT_IN_ACCELERATOR) {
            if (!uploadLocked(rec, cancel)) return false;
        }
    }

    rec.prefetched = true;
    rec.last_used = std::chrono::steady_clock::now();
    return true;
}

void ModelResidencyManager::pin(uint64_t region_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(region_id);
    if (it != records_.end()) it->second.pins++;
}

void ModelResidencyManager::unpin(uint64_t region_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(region_id);
    if (it != records_.end() && it->second.pins > 0) it->second.pins--;
}

bool ModelResidencyManager::isResident(uint64_t region_id,
                                         ModelResidencyTarget target) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(region_id);
    if (it == records_.end()) return false;
    const auto& s = it->second.state;
    if (target == ModelResidencyTarget::RAM)
        return s == ModelResidencyState::RESIDENT_IN_RAM ||
               s == ModelResidencyState::RESIDENT_IN_ACCELERATOR;
    return s == ModelResidencyState::RESIDENT_IN_ACCELERATOR;
}

ModelResidencyState ModelResidencyManager::state(uint64_t region_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(region_id);
    if (it == records_.end()) return ModelResidencyState::NOT_RESIDENT;
    return it->second.state;
}

const uint8_t* ModelResidencyManager::ramPointer(uint64_t region_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = records_.find(region_id);
    if (it == records_.end()) return nullptr;
    if (it->second.state != ModelResidencyState::RESIDENT_IN_RAM &&
        it->second.state != ModelResidencyState::RESIDENT_IN_ACCELERATOR)
        return nullptr;
    return it->second.ram.get();
}

bool ModelResidencyManager::evict(uint64_t region_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    return evictLocked(region_id);
}

void ModelResidencyManager::releaseAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& kv : records_) {
        Record& rec = kv.second;
        if (rec.accel && accelerator_) {
            accelerator_->release(rec.accel);
            rec.accel = nullptr;
        }
        rec.ram.reset();
        rec.state = ModelResidencyState::NOT_RESIDENT;
    }
    records_.clear();
    telemetry_.resident_ram_bytes = 0;
    telemetry_.resident_accelerator_bytes = 0;
}

uint64_t ModelResidencyManager::residentRamBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_.resident_ram_bytes;
}

uint64_t ModelResidencyManager::residentAcceleratorBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_.resident_accelerator_bytes;
}

StreamingTelemetry ModelResidencyManager::telemetry() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return telemetry_;
}

std::string ModelResidencyManager::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

// ---------------------------------------------------------------------------
// Private helpers — all called with mutex_ already held
// ---------------------------------------------------------------------------

ModelResidencyManager::Record& ModelResidencyManager::getOrCreateLocked(
        const ModelRegion& region) {
    auto it = records_.find(region.id);
    if (it == records_.end()) {
        records_[region.id] = Record{};
        records_[region.id].region = region;
        it = records_.find(region.id);
    }
    return it->second;
}

bool ModelResidencyManager::loadToRamLocked(Record& rec,
                                              std::atomic<bool>* cancel) {
    if (cancel && cancel->load()) {
        last_error_ = "Cancelled before RAM load";
        return false;
    }

    // Make room if needed.
    if (!makeRamRoomLocked(rec.region.size, rec.region.id)) return false;

    rec.state = ModelResidencyState::LOADING_TO_RAM;
    rec.transferring = true;

    // Allocate host buffer.
    auto buf = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[rec.region.size]);
    if (!buf) {
        rec.state = ModelResidencyState::ERROR;
        rec.transferring = false;
        last_error_ = "RAM allocation failed for region " + rec.region.identifier;
        telemetry_.transfer_failures++;
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    bool ok = source_.read(rec.region.offset, buf.get(), rec.region.size, cancel);
    auto t1 = std::chrono::steady_clock::now();

    if (!ok) {
        rec.state = ModelResidencyState::ERROR;
        rec.transferring = false;
        last_error_ = source_.lastError();
        telemetry_.transfer_failures++;
        return false;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rec.ram = std::move(buf);
    rec.state = ModelResidencyState::RESIDENT_IN_RAM;
    rec.transferring = false;

    telemetry_.bytes_read_from_disk += rec.region.size;
    telemetry_.bytes_to_ram += rec.region.size;
    telemetry_.read_ms += ms;
    telemetry_.resident_ram_bytes += rec.region.size;
    if (telemetry_.resident_ram_bytes > telemetry_.peak_ram_bytes)
        telemetry_.peak_ram_bytes = telemetry_.resident_ram_bytes;

    return true;
}

bool ModelResidencyManager::uploadLocked(Record& rec,
                                           std::atomic<bool>* cancel) {
    if (!accelerator_ || !accelerator_->isAvailable()) {
        last_error_ = "Accelerator unavailable for upload";
        telemetry_.transfer_failures++;
        return false;
    }
    if (!rec.ram) {
        last_error_ = "Cannot upload region without RAM buffer";
        telemetry_.transfer_failures++;
        return false;
    }
    if (cancel && cancel->load()) {
        last_error_ = "Cancelled before accelerator upload";
        return false;
    }

    if (!makeAccelRoomLocked(rec.region.size, rec.region.id)) return false;

    rec.state = ModelResidencyState::LOADING_TO_ACCELERATOR;
    rec.transferring = true;

    void* dst = accelerator_->allocate(rec.region.size);
    if (!dst) {
        rec.state = ModelResidencyState::RESIDENT_IN_RAM;  // keep RAM state honest
        rec.transferring = false;
        last_error_ = accelerator_->lastError();
        telemetry_.transfer_failures++;
        return false;
    }

    auto t0 = std::chrono::steady_clock::now();
    bool ok = accelerator_->upload(rec.ram.get(), dst, rec.region.size);
    auto t1 = std::chrono::steady_clock::now();

    if (!ok) {
        accelerator_->release(dst);
        rec.state = ModelResidencyState::RESIDENT_IN_RAM;
        rec.transferring = false;
        last_error_ = accelerator_->lastError();
        telemetry_.transfer_failures++;
        return false;
    }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    rec.accel = dst;
    rec.state = ModelResidencyState::RESIDENT_IN_ACCELERATOR;
    rec.transferring = false;

    telemetry_.bytes_to_accelerator += rec.region.size;
    telemetry_.upload_ms += ms;
    telemetry_.resident_accelerator_bytes += rec.region.size;
    if (telemetry_.resident_accelerator_bytes > telemetry_.peak_accelerator_bytes)
        telemetry_.peak_accelerator_bytes = telemetry_.resident_accelerator_bytes;

    return true;
}

bool ModelResidencyManager::evictLocked(uint64_t id) {
    auto it = records_.find(id);
    if (it == records_.end()) return true;  // already gone

    Record& rec = it->second;
    if (rec.pins > 0 || rec.refs > 0 || rec.transferring) return false;

    // Release accelerator memory.
    if (rec.accel && accelerator_) {
        if (telemetry_.resident_accelerator_bytes >= rec.region.size)
            telemetry_.resident_accelerator_bytes -= rec.region.size;
        else
            telemetry_.resident_accelerator_bytes = 0;
        accelerator_->release(rec.accel);
        rec.accel = nullptr;
    }
    // Release RAM.
    if (rec.ram) {
        if (telemetry_.resident_ram_bytes >= rec.region.size)
            telemetry_.resident_ram_bytes -= rec.region.size;
        else
            telemetry_.resident_ram_bytes = 0;
        rec.ram.reset();
    }
    rec.state = ModelResidencyState::NOT_RESIDENT;
    telemetry_.eviction_count++;
    records_.erase(it);
    return true;
}

bool ModelResidencyManager::makeRamRoomLocked(uint64_t needed, uint64_t skip_id) {
    while (telemetry_.resident_ram_bytes + needed > budget_.ram_bytes) {
        uint64_t victim = pickLruLocked(ModelResidencyTarget::RAM, skip_id);
        if (victim == 0) {
            last_error_ = "RAM residency budget exhausted — cannot evict further";
            return false;
        }
        if (!evictLocked(victim)) {
            last_error_ = "Cannot evict LRU region (pinned or transferring)";
            return false;
        }
    }
    return true;
}

bool ModelResidencyManager::makeAccelRoomLocked(uint64_t needed, uint64_t skip_id) {
    if (!accelerator_) return true;
    while (accelerator_->available() < needed) {
        uint64_t victim = pickLruLocked(ModelResidencyTarget::ACCELERATOR, skip_id);
        if (victim == 0) {
            last_error_ = "Accelerator residency budget exhausted";
            return false;
        }
        auto it = records_.find(victim);
        if (it == records_.end()) break;
        Record& rec = it->second;
        if (rec.accel) {
            if (telemetry_.resident_accelerator_bytes >= rec.region.size)
                telemetry_.resident_accelerator_bytes -= rec.region.size;
            else
                telemetry_.resident_accelerator_bytes = 0;
            accelerator_->release(rec.accel);
            rec.accel = nullptr;
            rec.state = ModelResidencyState::RESIDENT_IN_RAM;
            telemetry_.eviction_count++;
        }
    }
    return true;
}

uint64_t ModelResidencyManager::pickLruLocked(ModelResidencyTarget target,
                                               uint64_t skip_id) const {
    uint64_t victim_id = 0;
    std::chrono::steady_clock::time_point oldest = std::chrono::steady_clock::now();
    bool found = false;

    for (const auto& kv : records_) {
        if (kv.first == skip_id) continue;
        const Record& rec = kv.second;
        if (rec.pins > 0 || rec.refs > 0 || rec.transferring) continue;

        bool qualifies = false;
        if (target == ModelResidencyTarget::RAM) {
            qualifies = (rec.state == ModelResidencyState::RESIDENT_IN_RAM ||
                         rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR);
        } else {
            qualifies = (rec.state == ModelResidencyState::RESIDENT_IN_ACCELERATOR);
        }
        if (!qualifies) continue;

        if (!found || rec.last_used < oldest) {
            oldest = rec.last_used;
            victim_id = kv.first;
            found = true;
        }
    }
    return victim_id;
}

} // namespace agr
