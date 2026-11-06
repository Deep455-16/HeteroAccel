// src/cuda/CudaRuntime.cpp
#include "cuda/CudaRuntime.h"

#include <algorithm>
#include <limits>

namespace agr {
namespace cuda {

namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUdeviceptr = unsigned long long;

constexpr CUresult kSuccess = 0;

// CUdevice_attribute values from cuda.h (stable ABI constants).
constexpr int kAttrMaxThreadsPerBlock = 1;
constexpr int kAttrMultiprocessorCount = 16;
constexpr int kAttrIntegrated = 18;
constexpr int kAttrComputeMajor = 75;
constexpr int kAttrComputeMinor = 76;

} // namespace

struct Runtime::Api {
    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDriverGetVersion)(int*) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceTotalMem)(size_t*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice) = nullptr;
    CUresult (*cuDevicePrimaryCtxRelease)(CUdevice) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuMemGetInfo)(size_t*, size_t*) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t) = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
};

std::vector<std::string> Runtime::defaultLibraryCandidates() {
#if defined(_WIN32)
    return {"nvcuda.dll"};
#elif defined(__APPLE__)
    return {"/usr/local/cuda/lib/libcuda.dylib"};   // CUDA is not supported on modern macOS
#else
    return {"libcuda.so.1", "libcuda.so"};
#endif
}

Runtime::Runtime(std::vector<std::string> candidates) : candidates_(std::move(candidates)) {}

Runtime::~Runtime() {
    std::lock_guard<std::mutex> lk(mu_);
    if (api_) {
        // Leak guard: free anything the owner forgot to release.
        for (auto& a : allocs_) {
            if (api_->cuMemFree) api_->cuMemFree(static_cast<CUdeviceptr>(a.first));
        }
        allocs_.clear();
        for (size_t i = 0; i < primary_ctx_.size(); ++i) {
            if (primary_ctx_[i] && api_->cuDevicePrimaryCtxRelease) {
                api_->cuDevicePrimaryCtxRelease(static_cast<CUdevice>(i));
            }
        }
        delete api_;
        api_ = nullptr;
    }
}

Runtime& Runtime::shared() {
    // Intentionally leaked: avoids touching the driver during static teardown.
    static Runtime* inst = new Runtime();
    return *inst;
}

std::string Runtime::errorString(int code, const std::string& what) const {
    std::string msg = what + " failed (CUresult=" + std::to_string(code);
    if (api_ && api_->cuGetErrorString) {
        const char* s = nullptr;
        if (api_->cuGetErrorString(code, &s) == kSuccess && s) {
            msg += ": ";
            msg += s;
        }
    }
    msg += ")";
    return msg;
}

bool Runtime::initialize() {
    std::lock_guard<std::mutex> lk(mu_);
    if (init_attempted_) return initialized_;
    init_attempted_ = true;

    std::string err;
    if (!lib_.open(candidates_, &err)) {
        reason_ = "CUDA driver library not found (NVIDIA driver not installed?): " + err;
        return false;
    }

    auto api = new Api();
#define AGR_CU_REQUIRED(member, sym)                                         \
    api->member = lib_.symbolAs<decltype(api->member)>(sym);                \
    if (!api->member) {                                                      \
        reason_ = std::string("CUDA driver is missing required symbol ") + sym; \
        delete api; lib_.close(); return false;                              \
    }
    AGR_CU_REQUIRED(cuInit, "cuInit")
    AGR_CU_REQUIRED(cuDriverGetVersion, "cuDriverGetVersion")
    AGR_CU_REQUIRED(cuDeviceGetCount, "cuDeviceGetCount")
    AGR_CU_REQUIRED(cuDeviceGet, "cuDeviceGet")
    AGR_CU_REQUIRED(cuDeviceGetName, "cuDeviceGetName")
    AGR_CU_REQUIRED(cuDeviceGetAttribute, "cuDeviceGetAttribute")
    AGR_CU_REQUIRED(cuDeviceTotalMem, "cuDeviceTotalMem_v2")
    AGR_CU_REQUIRED(cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain")
    AGR_CU_REQUIRED(cuCtxSetCurrent, "cuCtxSetCurrent")
    AGR_CU_REQUIRED(cuMemGetInfo, "cuMemGetInfo_v2")
    AGR_CU_REQUIRED(cuMemAlloc, "cuMemAlloc_v2")
    AGR_CU_REQUIRED(cuMemFree, "cuMemFree_v2")
    AGR_CU_REQUIRED(cuMemcpyHtoD, "cuMemcpyHtoD_v2")
    AGR_CU_REQUIRED(cuMemcpyDtoH, "cuMemcpyDtoH_v2")
    AGR_CU_REQUIRED(cuMemsetD8, "cuMemsetD8_v2")
#undef AGR_CU_REQUIRED
    api->cuDevicePrimaryCtxRelease = lib_.symbolAs<decltype(api->cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease_v2");
    if (!api->cuDevicePrimaryCtxRelease)
        api->cuDevicePrimaryCtxRelease = lib_.symbolAs<decltype(api->cuDevicePrimaryCtxRelease)>("cuDevicePrimaryCtxRelease");
    api->cuGetErrorString = lib_.symbolAs<decltype(api->cuGetErrorString)>("cuGetErrorString");
    api_ = api;

    CUresult rc = api_->cuInit(0);
    if (rc != kSuccess) {
        reason_ = errorString(rc, "cuInit") + " -- driver present but no usable NVIDIA device";
        return false;
    }

    api_->cuDriverGetVersion(&driver_version_);

    int count = 0;
    rc = api_->cuDeviceGetCount(&count);
    if (rc != kSuccess) {
        reason_ = errorString(rc, "cuDeviceGetCount");
        return false;
    }
    if (count <= 0) {
        reason_ = "CUDA driver initialized, but zero CUDA-capable devices were found";
        initialized_ = true;       // driver is usable; there is simply no device
        return true;
    }

    for (int i = 0; i < count; ++i) {
        CUdevice dev = 0;
        if (api_->cuDeviceGet(&dev, i) != kSuccess) continue;
        DeviceProperties p;
        p.index = i;
        char name[256] = {0};
        api_->cuDeviceGetName(name, static_cast<int>(sizeof(name)), dev);
        p.name = name;
        int v = 0;
        if (api_->cuDeviceGetAttribute(&v, kAttrComputeMajor, dev) == kSuccess) p.compute_major = v;
        if (api_->cuDeviceGetAttribute(&v, kAttrComputeMinor, dev) == kSuccess) p.compute_minor = v;
        if (api_->cuDeviceGetAttribute(&v, kAttrMultiprocessorCount, dev) == kSuccess) p.multiprocessor_count = v;
        if (api_->cuDeviceGetAttribute(&v, kAttrMaxThreadsPerBlock, dev) == kSuccess) p.max_threads_per_block = v;
        if (api_->cuDeviceGetAttribute(&v, kAttrIntegrated, dev) == kSuccess) p.integrated = (v != 0);
        size_t total = 0;
        if (api_->cuDeviceTotalMem(&total, dev) == kSuccess) p.total_memory_bytes = total;
        devices_.push_back(p);
    }
    primary_ctx_.assign(devices_.size(), nullptr);
    initialized_ = true;
    if (devices_.empty()) reason_ = "Devices were reported but none could be queried successfully";
    return true;
}

bool Runtime::setCurrent(int device, std::string* error) {
    // Caller holds mu_.
    if (!api_ || !initialized_ || devices_.empty()) {
        if (error) *error = "CUDA unavailable: " + reason_;
        return false;
    }
    if (device < 0 || device >= static_cast<int>(devices_.size())) {
        if (error) *error = "Invalid CUDA device index " + std::to_string(device);
        return false;
    }
    size_t slot = static_cast<size_t>(device);
    if (!primary_ctx_[slot]) {
        CUcontext ctx = nullptr;
        CUresult rc = api_->cuDevicePrimaryCtxRetain(&ctx, device);
        if (rc != kSuccess) {
            if (error) *error = errorString(rc, "cuDevicePrimaryCtxRetain");
            return false;
        }
        primary_ctx_[slot] = ctx;
    }
    CUresult rc = api_->cuCtxSetCurrent(static_cast<CUcontext>(primary_ctx_[slot]));
    if (rc != kSuccess) {
        if (error) *error = errorString(rc, "cuCtxSetCurrent");
        return false;
    }
    return true;
}

bool Runtime::memoryInfo(int device, MemoryInfo& out, std::string* error) {
    if (!initialize()) { if (error) *error = "CUDA unavailable: " + reason_; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!setCurrent(device, error)) return false;
    size_t f = 0, t = 0;
    CUresult rc = api_->cuMemGetInfo(&f, &t);
    if (rc != kSuccess) { if (error) *error = errorString(rc, "cuMemGetInfo"); return false; }
    out.free_bytes = f;
    out.total_bytes = t;
    return true;
}

bool Runtime::allocate(int device, size_t bytes, uint64_t& dptr_out, std::string* error) {
    dptr_out = 0;
    if (bytes == 0) { if (error) *error = "Refusing zero-byte CUDA allocation"; return false; }
    if (!initialize()) { if (error) *error = "CUDA unavailable: " + reason_; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!setCurrent(device, error)) return false;
    CUdeviceptr p = 0;
    CUresult rc = api_->cuMemAlloc(&p, bytes);
    if (rc != kSuccess) {
        if (error) *error = errorString(rc, "cuMemAlloc(" + std::to_string(bytes) + " bytes)") ;
        return false;
    }
    dptr_out = static_cast<uint64_t>(p);
    allocs_.emplace_back(dptr_out, bytes);
    live_allocs_.fetch_add(1);
    live_bytes_.fetch_add(bytes);
    return true;
}

bool Runtime::release(int device, uint64_t dptr, std::string* error) {
    if (dptr == 0) return true;                       // freeing null is a no-op
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::find_if(allocs_.begin(), allocs_.end(),
                           [&](const std::pair<uint64_t, size_t>& a) { return a.first == dptr; });
    if (it == allocs_.end()) {
        if (error) *error = "Pointer was not allocated by this runtime (double free?)";
        return false;
    }
    if (!setCurrent(device, error)) return false;
    CUresult rc = api_->cuMemFree(static_cast<CUdeviceptr>(dptr));
    size_t sz = it->second;
    allocs_.erase(it);
    live_allocs_.fetch_sub(1);
    live_bytes_.fetch_sub(sz);
    if (rc != kSuccess) { if (error) *error = errorString(rc, "cuMemFree"); return false; }
    return true;
}

bool Runtime::copyHostToDevice(int device, uint64_t dst, const void* src, size_t bytes, std::string* error) {
    if (!src || bytes == 0 || dst == 0) { if (error) *error = "Invalid H2D copy arguments"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!setCurrent(device, error)) return false;
    CUresult rc = api_->cuMemcpyHtoD(static_cast<CUdeviceptr>(dst), src, bytes);
    if (rc != kSuccess) { if (error) *error = errorString(rc, "cuMemcpyHtoD"); return false; }
    return true;
}

bool Runtime::copyDeviceToHost(int device, void* dst, uint64_t src, size_t bytes, std::string* error) {
    if (!dst || bytes == 0 || src == 0) { if (error) *error = "Invalid D2H copy arguments"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!setCurrent(device, error)) return false;
    CUresult rc = api_->cuMemcpyDtoH(dst, static_cast<CUdeviceptr>(src), bytes);
    if (rc != kSuccess) { if (error) *error = errorString(rc, "cuMemcpyDtoH"); return false; }
    return true;
}

bool Runtime::memset(int device, uint64_t dst, uint8_t value, size_t bytes, std::string* error) {
    if (bytes == 0 || dst == 0) { if (error) *error = "Invalid memset arguments"; return false; }
    std::lock_guard<std::mutex> lk(mu_);
    if (!setCurrent(device, error)) return false;
    CUresult rc = api_->cuMemsetD8(static_cast<CUdeviceptr>(dst), value, bytes);
    if (rc != kSuccess) { if (error) *error = errorString(rc, "cuMemsetD8"); return false; }
    return true;
}

// ---------------------------------------------------------------------------
// DeviceBuffer
// ---------------------------------------------------------------------------
bool DeviceBuffer::allocate(size_t bytes) {
    if (!rt_) { error_ = "DeviceBuffer has no runtime"; return false; }
    reset();
    uint64_t p = 0;
    if (!rt_->allocate(device_, bytes, p, &error_)) return false;
    dptr_ = p;
    size_ = bytes;
    return true;
}

bool DeviceBuffer::upload(const void* host, size_t bytes, size_t offset) {
    if (!valid()) { error_ = "Buffer not allocated"; return false; }
    if (offset > size_ || bytes > size_ - offset) { error_ = "Upload out of bounds"; return false; }
    return rt_->copyHostToDevice(device_, dptr_ + offset, host, bytes, &error_);
}

bool DeviceBuffer::download(void* host, size_t bytes, size_t offset) const {
    if (!valid()) { error_ = "Buffer not allocated"; return false; }
    if (offset > size_ || bytes > size_ - offset) { error_ = "Download out of bounds"; return false; }
    return rt_->copyDeviceToHost(device_, host, dptr_ + offset, bytes, &error_);
}

bool DeviceBuffer::fill(uint8_t value) {
    if (!valid()) { error_ = "Buffer not allocated"; return false; }
    return rt_->memset(device_, dptr_, value, size_, &error_);
}

void DeviceBuffer::reset() {
    if (rt_ && dptr_ != 0) {
        std::string ignored;
        rt_->release(device_, dptr_, &ignored);
    }
    dptr_ = 0;
    size_ = 0;
}

} // namespace cuda
} // namespace agr
