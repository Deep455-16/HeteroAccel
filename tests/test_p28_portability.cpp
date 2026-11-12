// tests/test_p28_portability.cpp
// Phase 28: cross-platform & hardware-validation checks. Runs on Windows,
// Linux and macOS in CPU-only, Vulkan and CUDA builds.
#include "mini_test.h"
#include "core/HardwareInfo.h"
#include "hardware/CPUDetector.h"
#include "hardware/MemoryDetector.h"
#include "hardware/VulkanDetector.h"
#include "hardware/CUDADetector.h"
#include "platform/DynamicLibrary.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace agr;
using namespace agr::orchestration;

static DeviceInfo cpuDev() {
    DeviceInfo d; d.name = "Mock CPU";
    d.api_support = (uint32_t)DeviceInfo::APISupport::CPU;
    return d;
}

int main() {
    std::cout << "== test_p28_portability ==\n";

    // ---- 1. Software Vulkan classification --------------------------------
    AGR_CHECK(isSoftwareVulkanDevice("CPU", "anything"));
    AGR_CHECK(isSoftwareVulkanDevice("Other", "llvmpipe (LLVM 15.0.7, 256 bits)"));
    AGR_CHECK(isSoftwareVulkanDevice("Other", "Lavapipe"));
    AGR_CHECK(isSoftwareVulkanDevice("Other", "SwiftShader Device (LLVM 10.0.0)"));
    AGR_CHECK(isSoftwareVulkanDevice("Other", "Microsoft Basic Render Driver"));
    AGR_CHECK(!isSoftwareVulkanDevice("Integrated GPU", "Intel(R) Iris(R) Xe Graphics"));
    AGR_CHECK(!isSoftwareVulkanDevice("Discrete GPU", "NVIDIA GeForce RTX 4090"));
    AGR_CHECK(!isSoftwareVulkanDevice("Discrete GPU", "AMD Radeon RX 7900 XTX"));

    // ---- 2. Detector invariants on the real host ---------------------------
    CPUInfo cpu = CPUDetector::detect();
    AGR_CHECK(cpu.logical_processors > 0);
    AGR_CHECK(cpu.physical_cores > 0);
    AGR_CHECK(cpu.physical_cores <= cpu.logical_processors);
    AGR_CHECK(!cpu.architecture.empty());
    MemoryInfo mem = MemoryDetector::detect();
    AGR_CHECK(mem.total_physical_mb > 0);
    AGR_CHECK(mem.available_physical_mb <= mem.total_physical_mb);

    VulkanInfo vk = VulkanDetector::detect();
    std::cout << "  Vulkan available(physical)=" << vk.available << " devices=" << vk.devices.size() << "\n";
    for (const auto& d : vk.devices) {
        std::cout << "    - " << d.name << " [" << d.device_type << "]"
                  << (d.is_software_emulated ? " SOFTWARE" : " physical") << "\n";
        AGR_CHECK(d.is_software_emulated == isSoftwareVulkanDevice(d.device_type, d.name));
    }
    if (vk.available) {
        bool has_physical = false;
        for (const auto& d : vk.devices)
            if (!d.is_software_emulated && d.has_compute_capable_queue) has_physical = true;
        AGR_CHECK(has_physical);   // available must imply a physical compute GPU
    } else {
        AGR_CHECK(!vk.unavailable_reason.empty());
    }

    CUDAInfo cu = CUDADetector::detect(false);
    if (!cu.available) AGR_CHECK(!cu.unavailable_reason.empty());

    // The detector must agree with the raw Vulkan probe about physical availability.
    auto& hw = HardwareDetector::instance();
    hw.reset();
    hw.detect();
    AGR_CHECK(hw.has_vulkan() == vk.available);
    for (const auto& d : hw.devices()) {
        if (d.is_software_emulated) {
            AGR_CHECK(!d.vulkan_ok);
            AGR_CHECK(!(d.api_support & (uint32_t)DeviceInfo::APISupport::Vulkan));
        }
    }
    if (const auto* best = hw.best_gpu()) AGR_CHECK(!best->is_software_emulated);

    // ---- 3. Dynamic library loading ---------------------------------------
    {
        platform::DynamicLibrary missing;
        std::string err;
        AGR_CHECK(!missing.open({"definitely_not_a_real_library_p28.xyz"}, &err));
        AGR_CHECK(!err.empty());
        AGR_CHECK(!missing.isOpen());
        AGR_CHECK(missing.symbol("anything") == nullptr);

#if defined(_WIN32)
        const std::vector<std::string> sys = {"kernel32.dll"};
        const char* sym = "GetCurrentProcessId";
#elif defined(__APPLE__)
        const std::vector<std::string> sys = {"libSystem.B.dylib", "/usr/lib/libSystem.B.dylib"};
        const char* sym = "getpid";
#else
        const std::vector<std::string> sys = {"libc.so.6", "libm.so.6"};
        const char* sym = "getpid";
#endif
        platform::DynamicLibrary lib;
        AGR_CHECK(lib.open({"definitely_not_a_real_library_p28.xyz", sys[0], sys.back()}, &err));
        AGR_CHECK(lib.isOpen());
        AGR_CHECK(lib.symbol(sym) != nullptr);
        AGR_CHECK(lib.symbol("no_such_symbol_p28") == nullptr);
        platform::DynamicLibrary moved(std::move(lib));
        AGR_CHECK(moved.isOpen());
        AGR_CHECK(!lib.isOpen());
        moved.close();
        AGR_CHECK(!moved.isOpen());
    }

    // ---- 4. Filesystem portability ----------------------------------------
    {
        namespace fs = std::filesystem;
        fs::path dir = fs::temp_directory_path() / "heteroaccel_p28_fs";
        std::error_code ec;
        fs::remove_all(dir, ec);
        AGR_CHECK(fs::create_directories(dir / "sub dir" / "nested", ec));
        fs::path f = dir / "sub dir" / "nested" / "file.bin";
        {
            std::ofstream o(f, std::ios::binary);
            const char data[4] = {0, 1, 2, 3};
            o.write(data, 4);
        }
        AGR_CHECK(fs::file_size(f) == 4);
        std::ifstream in(f, std::ios::binary);
        char buf[4] = {};
        in.read(buf, 4);
        AGR_CHECK(buf[3] == 3);
        in.close();
        fs::remove_all(dir, ec);
        AGR_CHECK(!fs::exists(dir));
    }

    // ---- 5. Threading portability -----------------------------------------
    {
        std::atomic<long> sum{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < 8; ++t)
            ts.emplace_back([&] { for (int i = 0; i < 10000; ++i) sum.fetch_add(1); });
        for (auto& t : ts) t.join();
        AGR_CHECK(sum.load() == 80000);
    }

    // ---- 6. CPU-only fallback: no GPU of any kind -> hydroxl_cpu -----------
    {
        auto& orch = EngineOrchestrator::instance();
        orch.reset();
        hw.set_mock_environment({cpuDev()}, 16ull << 30, false, false);
        register_default_backends(orch, hw);
        EngineOrchestrator::SelectionRequest r;
        r.model_path = "mock.gguf"; r.model_arch = "llama"; r.quant_type = 12;
        r.model_size_bytes = 600ull << 20;
        auto sel = orch.select_backend(r);
        AGR_CHECK(sel.selected_backend == "hydroxl_cpu");

        // Forced GPU backends fall back to CPU when no hardware is present.
        for (const char* forced : {"hydroxl_vulkan", "cuda"}) {
            r.forced_backend = forced;
            r.allow_cpu_fallback = true;
            auto fs = orch.select_backend(r);
            AGR_CHECK(fs.selected_backend == "hydroxl_cpu");
            AGR_CHECK(fs.is_fallback);
        }

        // A software-only Vulkan machine must not select the Vulkan backend.
        DeviceInfo sw; sw.name = "llvmpipe"; sw.is_software_emulated = true;
        sw.vram_bytes = 4ull << 30;
        hw.set_mock_environment({cpuDev(), sw}, 16ull << 30, false, false);
        r.forced_backend.clear();
        auto sel2 = orch.select_backend(r);
        AGR_CHECK(sel2.selected_backend == "hydroxl_cpu");
        AGR_CHECK(hw.best_gpu() == nullptr);
    }

    hw.reset();
    AGR_TEST_MAIN_END();
}
