#pragma once
#include "core/HardwareInfo.h"

namespace agr {

// CUDA is optional and MUST NOT be a hard build/runtime dependency.
// This detector dynamically loads the CUDA *driver* library at runtime
// (nvcuda.dll on Windows / libcuda.so on Linux) -- it does not require
// the CUDA Toolkit to be installed on the build machine. If the driver
// library or an NVIDIA GPU is absent, this reports available=false with
// a reason, and the rest of the application continues normally.
class CUDADetector {
public:
    static CUDAInfo detect();
};

} // namespace agr
