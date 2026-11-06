#include "hardware/CUDADetector.h"
#include "cuda/CudaRuntime.h"

namespace agr {

// Detection reuses the real CUDA driver layer (agr::cuda::Runtime), which loads
// the driver dynamically; no CUDA Toolkit is needed to build this file.
CUDAInfo CUDADetector::detect(bool query_free_memory) {
    CUDAInfo info;

    cuda::Runtime& rt = cuda::Runtime::shared();
    if (!rt.initialize()) {
        info.available = false;
        info.unavailable_reason = rt.unavailableReason();
        return info;
    }

    info.driver_version = rt.driverVersion();

    if (rt.devices().empty()) {
        info.available = false;
        info.unavailable_reason = rt.unavailableReason().empty()
            ? "CUDA driver initialized successfully, but zero CUDA-capable devices were found"
            : rt.unavailableReason();
        return info;
    }

    for (const auto& d : rt.devices()) {
        CUDADeviceInfo di;
        di.index = d.index;
        di.name = d.name;
        di.compute_major = d.compute_major;
        di.compute_minor = d.compute_minor;
        di.total_memory_bytes = d.total_memory_bytes;
        di.multiprocessor_count = d.multiprocessor_count;
        di.integrated = d.integrated;
        di.meets_min_capability = cuda::meetsMinimumCapability(d);
        if (query_free_memory) {
            cuda::MemoryInfo mi;
            if (rt.memoryInfo(d.index, mi)) di.free_memory_bytes = mi.free_bytes;
        }
        info.devices.push_back(di);
    }

    info.available = !info.devices.empty();
    return info;
}

} // namespace agr
