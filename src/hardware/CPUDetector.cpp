#include "hardware/CPUDetector.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>

#if defined(_WIN32)
    #define AGR_PLATFORM_WINDOWS 1
    #include <windows.h>
    #include <intrin.h>
#else
    #define AGR_PLATFORM_WINDOWS 0
    #include <fstream>
    #include <string>
    #include <thread>
    #include <unistd.h>
#endif

namespace agr {

namespace {

std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    size_t end = s.find_last_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    return s.substr(start, end - start + 1);
}

std::string detectArchitecture() {
#if defined(_M_ARM64) || defined(__aarch64__)
    return "arm64";
#elif defined(_M_X64) || defined(__x86_64__) || defined(__amd64__)
    return "x86_64";
#elif defined(_M_IX86) || defined(__i386__)
    return "x86";
#else
    return "unknown";
#endif
}

#if AGR_PLATFORM_WINDOWS

std::string cpuidVendorString() {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0);
    char vendor[13];
    std::memcpy(vendor + 0, &regs[1], 4); // EBX
    std::memcpy(vendor + 4, &regs[3], 4); // EDX
    std::memcpy(vendor + 8, &regs[2], 4); // ECX
    vendor[12] = '\0';
    return std::string(vendor);
}

std::string cpuidBrandString() {
    int regs[4] = {0, 0, 0, 0};
    __cpuid(regs, 0x80000000);
    unsigned int maxExtended = static_cast<unsigned int>(regs[0]);
    if (maxExtended < 0x80000004) {
        return "";
    }
    char brand[49] = {0};
    for (unsigned int i = 0; i < 3; ++i) {
        __cpuid(regs, static_cast<int>(0x80000002 + i));
        std::memcpy(brand + i * 16, regs, sizeof(regs));
    }
    return trim(std::string(brand));
}

uint32_t countLogicalProcessorsWindows() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return static_cast<uint32_t>(sysInfo.dwNumberOfProcessors);
}

uint32_t countPhysicalCoresWindows() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) {
        return 0; // could not determine; caller falls back gracefully
    }

    std::vector<uint8_t> buffer(length);
    auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
        return 0;
    }

    uint32_t physicalCores = 0;
    size_t offset = 0;
    while (offset < length) {
        auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (current->Relationship == RelationProcessorCore) {
            ++physicalCores;
        }
        offset += current->Size;
    }
    return physicalCores;
}

#else // Linux / other POSIX (dev-time fallback, not the Phase-1 target platform)

struct ProcCpuInfo {
    std::string vendor;
    std::string modelName;
    std::set<std::pair<int,int>> physicalCoreIds; // (physical_id, core_id)
};

ProcCpuInfo readProcCpuInfo() {
    ProcCpuInfo result;
    std::ifstream file("/proc/cpuinfo");
    if (!file.is_open()) {
        return result;
    }

    std::string line;
    int currentPhysicalId = 0;
    int currentCoreId = -1;
    bool haveCoreId = false;

    auto flush = [&]() {
        if (haveCoreId) {
            result.physicalCoreIds.insert({currentPhysicalId, currentCoreId});
        }
    };

    while (std::getline(file, line)) {
        auto pos = line.find(':');
        if (pos == std::string::npos) {
            continue;
        }
        std::string key = trim(line.substr(0, pos));
        std::string value = trim(line.substr(pos + 1));

        if (key == "vendor_id" && result.vendor.empty()) {
            result.vendor = value;
        } else if (key == "model name" && result.modelName.empty()) {
            result.modelName = value;
        } else if (key == "physical id") {
            currentPhysicalId = std::atoi(value.c_str());
        } else if (key == "core id") {
            currentCoreId = std::atoi(value.c_str());
            haveCoreId = true;
        } else if (key == "processor") {
            // New logical processor block starting; flush the previous one.
            flush();
            haveCoreId = false;
        }
    }
    flush();

    return result;
}

#endif

} // namespace

CPUInfo CPUDetector::detect() {
    CPUInfo info;
    info.architecture = detectArchitecture();

#if AGR_PLATFORM_WINDOWS
    info.vendor = cpuidVendorString();
    std::string brand = cpuidBrandString();
    info.model_name = brand.empty() ? "Unknown CPU" : brand;
    info.logical_processors = countLogicalProcessorsWindows();
    info.physical_cores = countPhysicalCoresWindows();
    if (info.physical_cores == 0) {
        // Graceful degradation: could not resolve physical topology.
        info.physical_cores = info.logical_processors;
    }
#else
    ProcCpuInfo proc = readProcCpuInfo();
    info.vendor = proc.vendor.empty() ? "Unknown" : proc.vendor;
    info.model_name = proc.modelName.empty() ? "Unknown CPU" : proc.modelName;

    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    info.logical_processors = nprocs > 0 ? static_cast<uint32_t>(nprocs) : 0;

    uint32_t uniqueCores = static_cast<uint32_t>(proc.physicalCoreIds.size());
    info.physical_cores = uniqueCores > 0 ? uniqueCores : info.logical_processors;
#endif

    return info;
}

} // namespace agr
