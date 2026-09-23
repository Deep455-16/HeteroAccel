// src/backend/CUDAAllocator.cpp
// Phase 4 graceful CUDA stub.
// No CUDA SDK or NVIDIA driver required -- discovery is done via CUDADetector
// which dynamically loads nvcuda.dll at runtime.
#include "backend/CUDAAllocator.h"

namespace agr {

CUDAAllocator::CUDAAllocator() {
    // CUDADetector handles absence at runtime -- no headers or linkage needed.
}

MemoryBlock CUDAAllocator::allocate(size_t /*size*/) {
    return MemoryBlock{}; // invalid: CUDA not yet wired in Phase 4
}

void CUDAAllocator::free(MemoryBlock& block) {
    block = MemoryBlock{}; // no-op
}

} // namespace agr
