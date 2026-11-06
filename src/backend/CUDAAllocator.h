// src/backend/CUDAAllocator.h
// CUDA device memory allocator (IMemoryAllocator) backed by agr::cuda::Runtime.
//
// Allocation is real cuMemAlloc/cuMemFree through the CUDA driver API, loaded
// dynamically. When no NVIDIA driver/device exists, allocate() returns an
// invalid block (id==0) and lastError() explains why -- absence is never fatal.
#pragma once
#include "mem/IMemoryAllocator.h"
#include "cuda/CudaRuntime.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>

namespace agr {

class CUDAAllocator : public IMemoryAllocator {
public:
    explicit CUDAAllocator(int device_index = 0,
                           cuda::Runtime& runtime = cuda::Runtime::shared());
    ~CUDAAllocator() override;

    MemoryBlock allocate(size_t size) override;
    void        free(MemoryBlock& block) override;

    size_t      used()      const override { return used_.load(); }
    size_t      available() const override;
    size_t      capacity()  const override;
    std::string name()      const override;

    bool cudaAvailable() const;
    int  deviceIndex() const { return device_; }
    std::string lastError() const;

    /// Copy helpers keyed by MemoryBlock (bounds-checked against block.size).
    bool upload(const MemoryBlock& block, const void* host, size_t bytes, size_t offset = 0);
    bool download(const MemoryBlock& block, void* host, size_t bytes, size_t offset = 0);

private:
    int                 device_;
    cuda::Runtime&      rt_;
    std::atomic<size_t> used_{0};
    std::atomic<uint64_t> next_id_{1};
    mutable std::mutex  mu_;
    std::unordered_map<uint64_t, size_t> live_;   // dptr -> size
    std::string         last_error_;
};

} // namespace agr
