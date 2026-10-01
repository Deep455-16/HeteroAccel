#pragma once
#include "core/HardwareInfo.h"

namespace agr {

// Detects real total/available physical RAM.
// Windows: GlobalMemoryStatusEx. Linux (dev fallback): sysinfo().
class MemoryDetector {
public:
    static MemoryInfo detect();
};

} // namespace agr
