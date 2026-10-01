// src/profiler/Profiler.h
// Phase 8: Profiler implementation
#pragma once

#include "profiler/IProfiler.h"
#include "scheduler/PerformanceHistory.h"
#include "hardware/HardwareDetector.h"
#include <mutex>

namespace agr {

class Profiler : public IProfiler {
public:
    Profiler(PerformanceHistory& history, const HardwareInfo& hw);
    ~Profiler() override = default;

    void recordEvent(const ProfileEvent& event) override;
    std::vector<ProfileEvent> getEvents() const override;

    bool saveProfile(const std::string& path) override;
    bool loadProfile(const std::string& path) override;
    void reset() override;

private:
    std::string hardwareId() const;
    ProfileKey makeKey(const ProfileEvent& event) const;

    PerformanceHistory& history_;
    HardwareInfo hw_;

    mutable std::mutex mutex_;
    std::vector<ProfileEvent> events_; // bounded history of raw events
    size_t max_events_ = 1000;
};

} // namespace agr
