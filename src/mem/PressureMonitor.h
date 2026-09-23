// src/mem/PressureMonitor.h
#pragma once
#include "mem/MemoryTypes.h"
#include <cstddef>

namespace agr {

/// Assesses memory pressure based on utilisation ratio.
class PressureMonitor {
public:
    PressureMonitor(double warningThreshold = 0.70,
                    double highThreshold = 0.85,
                    double criticalThreshold = 0.95);

    /// Calculate pressure level from current usage and capacity.
    PressureLevel assess(size_t usedBytes, size_t capacityBytes) const;

    void setThresholds(double warning, double high, double critical);

private:
    double warn_ = 0.70;
    double high_ = 0.85;
    double crit_ = 0.95;
};

} // namespace agr
