#pragma once
#include "core/HardwareInfo.h"

namespace agr {

// Detects real CPU vendor/model/core counts from the running system.
// Never hardcodes a specific CPU model. Windows path uses CPUID +
// GetLogicalProcessorInformationEx; non-Windows path (used for local
// dev/testing in this repo) reads /proc/cpuinfo and sysconf().
class CPUDetector {
public:
    static CPUInfo detect();
};

} // namespace agr
