// src/backend/DeviceSelector.cpp
#include "backend/DeviceSelector.h"
#include <algorithm>
#include <limits>

namespace agr {

DeviceSelector::DeviceSelector(const BackendManager& manager)
    : manager_(manager) {}

double DeviceSelector::scoreDevice(const ComputeDevice& device, const WorkloadHint& hint) const {
    if (!device.is_available) return -1.0;

    double score = device.compute_score;

    // 1. Memory adequacy check
    if (hint.required_memory > 0 && device.memory_available > 0) {
        if (device.memory_available < hint.required_memory) {
            score -= 10000.0; // disqualify: not enough memory
        } else {
            double headroom = static_cast<double>(device.memory_available - hint.required_memory)
                            / static_cast<double>(device.memory_available);
            score += headroom * 50.0;
        }
    }

    // 2. GPU preference bonus
    if (hint.prefer_gpu) {
        if (device.backend == ComputeBackend::CUDA)   score += 200.0;
        if (device.backend == ComputeBackend::VULKAN) score += 150.0;
    }

    // 3. Compute-intensive tasks strongly prefer GPU
    if (hint.compute_intensive && device.backend == ComputeBackend::CPU) {
        score -= 50.0;
    }

    return score;
}

ComputeDevice DeviceSelector::selectDevice(const WorkloadHint& hint) const {
    const auto& all = manager_.allDevices();

    const ComputeDevice* best      = nullptr;
    double               bestScore = std::numeric_limits<double>::lowest();

    for (const auto& dev : all) {
        double s = scoreDevice(dev, hint);
        if (s > bestScore) {
            bestScore = s;
            best      = &dev;
        }
    }

    // CPU is always the guaranteed fallback
    if (!best || !best->is_available) {
        const ComputeDevice* cpu = manager_.getDevice(ComputeBackend::CPU);
        if (cpu) return *cpu;
        // Absolute last resort
        ComputeDevice fallback;
        fallback.backend          = ComputeBackend::CPU;
        fallback.is_available     = true;
        fallback.name             = "CPU (fallback)";
        fallback.supports_compute = true;
        return fallback;
    }

    return *best;
}

} // namespace agr
