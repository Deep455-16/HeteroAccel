// src/mem/PressureMonitor.cpp
#include "mem/PressureMonitor.h"

namespace agr {

PressureMonitor::PressureMonitor(double warningThreshold, double highThreshold, double criticalThreshold)
    : warn_(warningThreshold), high_(highThreshold), crit_(criticalThreshold) {}

PressureLevel PressureMonitor::assess(size_t usedBytes, size_t capacityBytes) const {
    if (capacityBytes == 0) return PressureLevel::NORMAL; // cannot compute

    double ratio = static_cast<double>(usedBytes) / static_cast<double>(capacityBytes);
    
    if (ratio >= crit_) return PressureLevel::CRITICAL;
    if (ratio >= high_) return PressureLevel::HIGH;
    if (ratio >= warn_) return PressureLevel::WARNING;
    return PressureLevel::NORMAL;
}

void PressureMonitor::setThresholds(double warning, double high, double critical) {
    warn_ = warning;
    high_ = high;
    crit_ = critical;
}

} // namespace agr
