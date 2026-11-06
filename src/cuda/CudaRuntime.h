// src/cuda/CudaRuntime.h
// Real CUDA device enumeration, memory allocation and host<->device transfers.
//
// Implemented on the CUDA *driver* API, resolved at runtime via
// agr::platform::DynamicLibrary (nvcuda.dll / libcuda.so.1). Consequences:
//   * No CUDA Toolkit or nvcc is needed to build this file.
//   * Machines without an NVIDIA driver build and run normally; every call
//     reports a descriptive error instead of crashing.
//
// This layer is distinct from CUDA *inference*, which is performed by
// llama.cpp's GGML_CUDA backend (see HETEROACCEL_ENABLE_CUDA).
#pragma once

#include "platform/DynamicLibrary.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace agr {
namespace cuda {

struct DeviceProperties {
    int         index = 0;
    std::string name;
    int         compute_major = 0;
    int         compute_minor = 0;
    size_t      total_memory_bytes = 0;
    int         multiprocessor_count = 0;
    int         max_threads_per_block = 0;
    bool        integrated = false;
};

struct MemoryInfo {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
};

/// Minimum compute capability supported by upstream ggml-cuda builds.
constexpr int kMinComputeMajor = 5;   // Maxwell (sm_50)

/// Returns true when the device meets the minimum capability for GGML_CUDA.
inline bool meetsMinimumCapability(const DeviceProperties& p) {
    return p.compute_major >= kMinComputeMajor;
}

class Runtime {
public:
    /// Default candidates: nvcuda.dll (Windows), libcuda.so.1 / libcuda.so (Linux).
    static std::vector<std::string> defaultLibraryCandidates();

    explicit Runtime(std::vector<std::string> candidates = defaultLibraryCandidates());
    ~Runtime();
    Runtime(const Runtime&)            = delete;
    Runtime& operator=(const Runtime&) = delete;

    /// Process-wide shared instance using the default library candidates.
    static Runtime& shared();

    /// Load the driver and call cuInit. Idempotent. False => see unavailableReason().
    bool initialize();
    bool available() const { return initialized_ && !devices_.empty(); }
    const std::string& unavailableReason() const { return reason_; }
    int driverVersion() const { return driver_version_; }

    const std::vector<DeviceProperties>& devices() const { return devices_; }
    int deviceCount() const { return static_cast<int>(devices_.size()); }

    /// Free/total memory of a device (queries the live driver).
    bool memoryInfo(int device, MemoryInfo& out, std::string* error = nullptr);

    // --- Raw device memory (prefer DeviceBuffer) -----------------------------
    bool allocate(int device, size_t bytes, uint64_t& dptr_out, std::string* error = nullptr);
    bool release(int device, uint64_t dptr, std::string* error = nullptr);
    bool copyHostToDevice(int device, uint64_t dst, const void* src, size_t bytes,
                          std::string* error = nullptr);
    bool copyDeviceToHost(int device, void* dst, uint64_t src, size_t bytes,
                          std::string* error = nullptr);
    bool memset(int device, uint64_t dst, uint8_t value, size_t bytes,
                std::string* error = nullptr);

    /// Number of device allocations currently live (for leak checks).
    size_t liveAllocations() const { return live_allocs_.load(); }
    size_t liveBytes() const { return live_bytes_.load(); }

private:
    struct Api;                       // resolved driver entry points
    bool setCurrent(int device, std::string* error);
    std::string errorString(int code, const std::string& what) const;

    std::vector<std::string> candidates_;
    platform::DynamicLibrary lib_;
    Api*                     api_ = nullptr;
    std::mutex               mu_;
    bool                     initialized_ = false;
    bool                     init_attempted_ = false;
    std::string              reason_;
    int                      driver_version_ = 0;
    std::vector<DeviceProperties> devices_;
    std::vector<void*>       primary_ctx_;   // CUcontext per device (retained lazily)
    std::atomic<size_t>      live_allocs_{0};
    std::atomic<size_t>      live_bytes_{0};
    // size per live pointer so release() can keep byte accounting exact
    std::vector<std::pair<uint64_t, size_t>> allocs_;
};

/// RAII owner of one device allocation. Move-only. Bounds-checked transfers.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(Runtime& rt, int device) : rt_(&rt), device_(device) {}
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept { moveFrom(o); }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { reset(); moveFrom(o); }
        return *this;
    }

    bool allocate(size_t bytes);
    bool upload(const void* host, size_t bytes, size_t offset = 0);
    bool download(void* host, size_t bytes, size_t offset = 0) const;
    bool fill(uint8_t value);
    void reset();

    size_t size() const { return size_; }
    uint64_t devicePtr() const { return dptr_; }
    bool valid() const { return dptr_ != 0; }
    const std::string& lastError() const { return error_; }

private:
    void moveFrom(DeviceBuffer& o) {
        rt_ = o.rt_; device_ = o.device_; dptr_ = o.dptr_; size_ = o.size_; error_ = std::move(o.error_);
        o.dptr_ = 0; o.size_ = 0;
    }
    Runtime*            rt_ = nullptr;
    int                 device_ = 0;
    uint64_t            dptr_ = 0;
    size_t              size_ = 0;
    mutable std::string error_;
};

} // namespace cuda
} // namespace agr
