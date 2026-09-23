// src/backend/DeviceSelector.h
// Automatic backend selection -- HeteroAccel decides, not the user.
// Implements a scoring/cost model over available ComputeDevices.
#pragma once
#include "backend/ComputeDevice.h"
#include "backend/BackendManager.h"

namespace agr {

/// Selects the best available compute device for a given workload.
/// This is where "HeteroAccel decides" is actually implemented.
class DeviceSelector {
public:
    explicit DeviceSelector(const BackendManager& manager);

    /// Select the best device. CPU is always the guaranteed fallback.
    ComputeDevice selectDevice(const WorkloadHint& hint) const;

    /// Score a single candidate. Higher = better. Negative = disqualified.
    double scoreDevice(const ComputeDevice& device, const WorkloadHint& hint) const;

private:
    const BackendManager& manager_;
};

} // namespace agr
