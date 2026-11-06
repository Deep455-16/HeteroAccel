// src/backend/CUDAAllocator.cpp
#include "backend/CUDAAllocator.h"

namespace agr {

CUDAAllocator::CUDAAllocator(int device_index, cuda::Runtime& runtime)
    : device_(device_index), rt_(runtime) {}

CUDAAllocator::~CUDAAllocator() {
    std::lock_guard<std::mutex> lk(mu_);
    std::string ignored;
    for (auto& kv : live_) {
        rt_.release(device_, kv.first, &ignored);
    }
    live_.clear();
    used_ = 0;
}

bool CUDAAllocator::cudaAvailable() const {
    return rt_.initialize() && device_ >= 0 && device_ < rt_.deviceCount();
}

std::string CUDAAllocator::lastError() const {
    std::lock_guard<std::mutex> lk(mu_);
    return last_error_;
}

MemoryBlock CUDAAllocator::allocate(size_t size) {
    MemoryBlock invalid;
    std::string err;
    uint64_t dptr = 0;
    if (!rt_.allocate(device_, size, dptr, &err)) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = err;
        return invalid;
    }
    MemoryBlock b;
    b.id             = next_id_.fetch_add(1);
    b.size           = size;
    b.location       = MemoryLocation::GPU;
    b.state          = MemoryState::RESIDENT;
    b.residency      = Residency::GPU;
    b.backend_handle = dptr;
    b.last_accessed  = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lk(mu_);
        live_[dptr] = size;
        last_error_.clear();
    }
    used_.fetch_add(size);
    return b;
}

void CUDAAllocator::free(MemoryBlock& block) {
    if (!block.isValid()) return;
    uint64_t dptr = block.backend_handle;
    size_t sz = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = live_.find(dptr);
        if (it == live_.end()) {         // not ours / already freed: ignore safely
            last_error_ = "free() of unknown or already-freed CUDA block";
            block = MemoryBlock{};
            return;
        }
        sz = it->second;
        live_.erase(it);
    }
    std::string err;
    if (!rt_.release(device_, dptr, &err)) {
        std::lock_guard<std::mutex> lk(mu_);
        last_error_ = err;
    }
    used_.fetch_sub(sz);
    block = MemoryBlock{};
}

size_t CUDAAllocator::capacity() const {
    if (!cudaAvailable()) return 0;
    return rt_.devices()[static_cast<size_t>(device_)].total_memory_bytes;
}

size_t CUDAAllocator::available() const {
    if (!cudaAvailable()) return 0;
    cuda::MemoryInfo mi;
    if (!const_cast<cuda::Runtime&>(rt_).memoryInfo(device_, mi)) return 0;
    return mi.free_bytes;
}

std::string CUDAAllocator::name() const {
    if (!cudaAvailable()) return "CUDAAllocator(unavailable)";
    return "CUDAAllocator(" + rt_.devices()[static_cast<size_t>(device_)].name + ")";
}

bool CUDAAllocator::upload(const MemoryBlock& block, const void* host, size_t bytes, size_t offset) {
    if (!block.isValid() || offset > block.size || bytes > block.size - offset) return false;
    return rt_.copyHostToDevice(device_, block.backend_handle + offset, host, bytes);
}

bool CUDAAllocator::download(const MemoryBlock& block, void* host, size_t bytes, size_t offset) {
    if (!block.isValid() || offset > block.size || bytes > block.size - offset) return false;
    return rt_.copyDeviceToHost(device_, host, block.backend_handle + offset, bytes);
}

} // namespace agr
