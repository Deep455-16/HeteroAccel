#include "hardware/MemoryDetector.h"

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__APPLE__)
    #include <mach/mach.h>
    #include <sys/sysctl.h>
    #include <sys/types.h>
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
#elif defined(__APPLE__)
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
        info.total_physical_mb = memsize / (1024ULL * 1024ULL);
    }
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS) {
        vm_size_t page = 0;
        host_page_size(mach_host_self(), &page);
        uint64_t avail_pages = static_cast<uint64_t>(vm.free_count) + vm.inactive_count;
        info.available_physical_mb = (avail_pages * page) / (1024ULL * 1024ULL);
    }
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
