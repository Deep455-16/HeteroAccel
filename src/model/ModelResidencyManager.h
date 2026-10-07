#pragma once

#include "model/IAcceleratorMemory.h"
#include "model/IModelDataSource.h"
#include "model/ModelRegion.h"
#include "model/ModelResidencyTypes.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agr {

class ModelResidencyManager {
public:
    ModelResidencyManager(IModelDataSource& source,
                          ModelResidencyBudget budget,
                          IAcceleratorMemory* accelerator = nullptr);

    ~ModelResidencyManager();

    ModelResidencyManager(const ModelResidencyManager&) = delete;
    ModelResidencyManager& operator=(const ModelResidencyManager&) = delete;

    void setPrefetchEnabled(bool enabled) { prefetch_enabled_ = enabled; }
    bool prefetchEnabled() const { return prefetch_enabled_; }

    bool ensureResident(const ModelRegion& region, ModelResidencyTarget target,
                        std::atomic<bool>* cancel = nullptr);
    bool prefetch(const ModelRegion& region, ModelResidencyTarget target,
                  std::atomic<bool>* cancel = nullptr);

    void pin(uint64_t region_id);
    void unpin(uint64_t region_id);

    bool isResident(uint64_t region_id, ModelResidencyTarget target) const;
    ModelResidencyState state(uint64_t region_id) const;
    const uint8_t* ramPointer(uint64_t region_id) const;

    bool evict(uint64_t region_id);
    void releaseAll();

    uint64_t residentRamBytes() const;
    uint64_t residentAcceleratorBytes() const;
    StreamingTelemetry telemetry() const;
    std::string lastError() const;
    const ModelResidencyBudget& budget() const { return budget_; }
    IAcceleratorMemory* accelerator() const { return accelerator_; }

private:
    struct Record {
        ModelRegion region;
        ModelResidencyState state = ModelResidencyState::NOT_RESIDENT;
        std::unique_ptr<uint8_t[]> ram;
        void* accel = nullptr;
        int refs = 0;
        int pins = 0;
        std::chrono::steady_clock::time_point last_used{};
        bool transferring = false;
        bool prefetched = false;
    };

    bool loadToRamLocked(Record& rec, std::atomic<bool>* cancel);
    bool uploadLocked(Record& rec, std::atomic<bool>* cancel);
    bool evictLocked(uint64_t id);
    bool makeRamRoomLocked(uint64_t needed, uint64_t skip_id);
    bool makeAccelRoomLocked(uint64_t needed, uint64_t skip_id);
    uint64_t pickLruLocked(ModelResidencyTarget target, uint64_t skip_id) const;
    Record& getOrCreateLocked(const ModelRegion& region);

    IModelDataSource& source_;
    ModelResidencyBudget budget_;
    IAcceleratorMemory* accelerator_ = nullptr;
    bool prefetch_enabled_ = true;

    mutable std::mutex mutex_;
    std::unordered_map<uint64_t, Record> records_;
    StreamingTelemetry telemetry_;
    std::string last_error_;
};

} // namespace agr
