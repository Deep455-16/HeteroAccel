#pragma once
#include "core/HardwareInfo.h"

namespace agr {

// Runs all individual detectors and assembles a HardwareInfo. Each
// sub-detector is isolated with its own try/catch so that a failure in
// one optional backend (e.g. CUDA) can never crash detection of the
// others (e.g. CPU/RAM/Vulkan).
class HardwareDetector {
public:
    static HardwareInfo detectAll();
};

} // namespace agr
