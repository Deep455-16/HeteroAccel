// src/backend/CUDAAllocator.h
// CUDA memory allocator interface.
// Phase 4: graceful stub -- compiles and runs correctly on non-NVIDIA machines.
// Phase 5: real cudaMalloc/cudaFree when a CUDA machine is available.
#pragma once
#include "mem/IMemoryAllocator.h"

namespace agr {

/// CUDA memory allocator.
/// Currently a safe stub. CUDA absence is NEVER an error.
class CUDAAllocator : public IMemoryAllocator {
public:
    CUDAAllocator();
    ~CUDAAllocator() override = default;

    MemoryBlock allocate(size_t size) override;
    void        free(MemoryBlock& block) override;

    size_t      used()      const override { return 0; }
    size_t      available() const override { return 0; }
    size_t      capacity()  const override { return 0; }
    std::string name()      const override { return "CUDAAllocator(unavailable)"; }

    bool cudaAvailable() const { return false; }
};

} // namespace agr
