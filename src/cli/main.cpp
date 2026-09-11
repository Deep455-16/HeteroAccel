#include "hardware/HardwareDetector.h"
#include "json/HardwareJson.h"
#include "gpu/VulkanBackend.h"
#include "benchmark/VectorAddBenchmark.h"

#include <iostream>
#include <string>
#include <vector>

using namespace agr;

namespace {

void printHumanReadable(const HardwareInfo& info) {
    std::cout << "=== CPU ===\n";
    std::cout << "  Vendor:            " << info.cpu.vendor << "\n";
    std::cout << "  Model:             " << info.cpu.model_name << "\n";
    std::cout << "  Architecture:      " << info.cpu.architecture << "\n";
    std::cout << "  Logical processors:" << info.cpu.logical_processors << "\n";
    std::cout << "  Physical cores:    " << info.cpu.physical_cores << "\n\n";

    std::cout << "=== Memory ===\n";
    std::cout << "  Total RAM:     " << info.memory.total_physical_mb << " MB\n";
    std::cout << "  Available RAM: " << info.memory.available_physical_mb << " MB\n\n";

    std::cout << "=== GPU(s) ===\n";
    if (info.gpus.empty()) {
        std::cout << "  No GPUs detected on this system.\n\n";
    } else {
        for (const auto& g : info.gpus) {
            std::cout << "  Name:   " << g.name << "\n";
            std::cout << "  Vendor: " << toString(g.vendor) << "\n";
            std::cout << "  Kind:   " << toString(g.kind) << "\n";
            std::cout << "  Dedicated VRAM:   " << g.dedicated_vram_mb << " MB\n";
            std::cout << "  Shared System Mem:" << g.shared_system_memory_mb << " MB\n";
            std::cout << "  ---\n";
        }
        std::cout << "\n";
    }

    std::cout << "=== Vulkan ===\n";
    std::cout << "  Available: " << (info.vulkan.available ? "YES" : "NO") << "\n";
    if (!info.vulkan.available) {
        std::cout << "  Reason:    " << info.vulkan.unavailable_reason << "\n";
    }
    std::cout << "  Loader instance API: " << info.vulkan.instance_api_version_major << "."
               << info.vulkan.instance_api_version_minor << "."
               << info.vulkan.instance_api_version_patch << "\n";
    for (const auto& d : info.vulkan.devices) {
        std::cout << "  Device: " << d.name << "\n";
        std::cout << "    Type:            " << d.device_type << "\n";
        std::cout << "    API version:     " << d.api_version_major << "." << d.api_version_minor << "." << d.api_version_patch << "\n";
        std::cout << "    Device-local mem:" << d.device_local_memory_mb << " MB\n";
        std::cout << "    Compute queue:   " << (d.has_compute_capable_queue ? "Available" : "Not available") << "\n";
    }
    std::cout << "\n";

    std::cout << "=== CUDA ===\n";
    std::cout << "  Available: " << (info.cuda.available ? "YES" : "NO") << "\n";
    if (!info.cuda.available) {
        std::cout << "  Reason:    " << info.cuda.unavailable_reason << "\n";
    }
    for (const auto& d : info.cuda.devices) {
        std::cout << "  Device[" << d.index << "]: " << d.name << "\n";
    }
}

void printUsage() {
    std::cout << "adaptive-gpu - Universal Adaptive GPU Acceleration Runtime (Phase 1-2)\n\n";
    std::cout << "Usage:\n";
    std::cout << "  adaptive-gpu hardware              Show detected hardware (human-readable)\n";
    std::cout << "  adaptive-gpu hardware --json       Show detected hardware (JSON)\n";
    std::cout << "  adaptive-gpu benchmark vulkan       Run the CPU vs Vulkan vector-add benchmark\n";
    std::cout << "  adaptive-gpu --help                 Show this message\n";
}

int runBenchmarkVulkan() {
    VulkanBackend backend;
    bool initialized = backend.initialize();

    if (initialized) {
        std::cout << "Vulkan backend initialized on: " << backend.deviceName() << "\n\n";
    } else {
        std::cout << "Vulkan backend NOT available: " << backend.lastError() << "\n";
        std::cout << "Continuing with CPU-only baseline (GPU columns will read N/A).\n\n";
    }

    std::vector<size_t> sizes = {
        1024,            // Small
        65536,           // Medium
        1048576,         // Large (1M)
        16777216,        // Very Large (16M) -- 64MB per buffer
    };

    BenchmarkReport report = VectorAddBenchmark::run(backend, sizes);
    std::cout << VectorAddBenchmark::formatText(report);

    backend.shutdown();
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.empty() || args[0] == "--help" || args[0] == "-h") {
        printUsage();
        return 0;
    }

    if (args[0] == "hardware") {
        bool json = false;
        for (size_t i = 1; i < args.size(); ++i) {
            if (args[i] == "--json") json = true;
        }

        HardwareInfo info = HardwareDetector::detectAll();

        if (json) {
            std::cout << toJson(info) << "\n";
        } else {
            printHumanReadable(info);
        }
        return 0;
    }

    if (args[0] == "benchmark") {
        if (args.size() >= 2 && args[1] == "vulkan") {
            return runBenchmarkVulkan();
        }
        std::cerr << "Usage: adaptive-gpu benchmark vulkan\n";
        return 1;
    }

    std::cerr << "Unknown command: " << args[0] << "\n";
    printUsage();
    return 1;
}
