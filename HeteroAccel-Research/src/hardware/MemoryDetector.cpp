#include "hardware/MemoryDetector.h"

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <sys/sysinfo.h>
#endif

namespace agr {

MemoryInfo MemoryDetector::detect() {
    MemoryInfo info;

#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        info.total_physical_mb = static_cast<uint64_t>(status.ullTotalPhys / (1024ULL * 1024ULL));
        info.available_physical_mb = static_cast<uint64_t>(status.ullAvailPhys / (1024ULL * 1024ULL));
    }
    // If GlobalMemoryStatusEx fails, info stays zero-initialized rather
    // than fabricating a value.
#else
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        uint64_t unit = static_cast<uint64_t>(si.mem_unit > 0 ? si.mem_unit : 1);
        info.total_physical_mb = (static_cast<uint64_t>(si.totalram) * unit) / (1024ULL * 1024ULL);
        info.available_physical_mb = (static_cast<uint64_t>(si.freeram) * unit) / (1024ULL * 1024ULL);
    }
#endif

    return info;
}

} // namespace agr
