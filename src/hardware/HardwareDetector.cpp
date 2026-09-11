#include "hardware/HardwareDetector.h"
#include "hardware/CPUDetector.h"
#include "hardware/MemoryDetector.h"
#include "hardware/GPUDetector.h"
#include "hardware/VulkanDetector.h"
#include "hardware/CUDADetector.h"

namespace agr {

HardwareInfo HardwareDetector::detectAll() {
    HardwareInfo info;

    try {
        info.cpu = CPUDetector::detect();
    } catch (...) {
        // Leave default-constructed CPUInfo; never crash the whole run
        // because one detector misbehaved.
    }

    try {
        info.memory = MemoryDetector::detect();
    } catch (...) {
    }

    try {
        info.gpus = GPUDetector::detect();
    } catch (...) {
    }

    try {
        info.vulkan = VulkanDetector::detect();
    } catch (...) {
        info.vulkan.available = false;
        info.vulkan.unavailable_reason = "Vulkan detection threw an exception";
    }

    try {
        info.cuda = CUDADetector::detect();
    } catch (...) {
        info.cuda.available = false;
        info.cuda.unavailable_reason = "CUDA detection threw an exception";
    }

    return info;
}

} // namespace agr
