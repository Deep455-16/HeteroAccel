// tests/test_p26_backend_selection.cpp
// Phase 26: backend selection / fallback logic with CUDA, driven by mocked
// hardware descriptions (so it runs everywhere) and test-double engines.
// Real CUDA inference is covered by test_p26_llama_selection / *_real_model.
#include "mini_test.h"
#include "orchestration/EngineOrchestrator.h"
#include "orchestration/OrchestratorInit.h"
#include "orchestration/HardwareDetector.h"

#include <memory>
#include <string>
#include <vector>

using namespace agr::orchestration;

namespace {

class FakeEngine : public IInferenceEngine {
public:
    FakeEngine(std::string id, bool init_ok, std::string err = "")
        : id_(std::move(id)), init_ok_(init_ok), err_(std::move(err)) {}
    bool initialize(const std::string&, std::string& error) override {
        if (!init_ok_) { error = err_; return false; }
        ready_ = true; return true;
    }
    InferenceResult run(const InferenceRequest&) override { InferenceResult r; r.success = true; r.backend_used = id_; return r; }
    void shutdown() override { ready_ = false; }
    std::string backend_id() const override { return id_; }
    bool is_ready() const override { return ready_; }
private:
    std::string id_; bool init_ok_; std::string err_; bool ready_ = false;
};

DeviceInfo cpuDevice() {
    DeviceInfo d; d.name = "Mock CPU"; d.api_support = (uint32_t)DeviceInfo::APISupport::CPU; return d;
}

DeviceInfo nvidiaDevice(size_t vram_gib) {
    DeviceInfo d;
    d.name = "Mock NVIDIA GPU";
    d.vendor = DeviceInfo::Vendor::NVIDIA;
    d.api_support = (uint32_t)DeviceInfo::APISupport::CUDA | (uint32_t)DeviceInfo::APISupport::Vulkan;
    d.vram_bytes = vram_gib << 30;
    d.cuda_ok = true;
    d.vulkan_ok = true;
    d.cuda_device_index = 0;
    d.cuda_compute_major = 8;
    d.cuda_total_bytes = d.vram_bytes;
    return d;
}

DeviceInfo igpuVulkanOnly() {
    DeviceInfo d;
    d.name = "Mock iGPU";
    d.vendor = DeviceInfo::Vendor::Intel;
    d.api_support = (uint32_t)DeviceInfo::APISupport::Vulkan;
    d.vram_bytes = 1ull << 30;
    d.vulkan_ok = true;
    return d;
}

EngineOrchestrator::SelectionRequest request(const std::string& forced = "") {
    EngineOrchestrator::SelectionRequest r;
    r.model_path = "mock.gguf";
    r.model_arch = "llama";
    r.quant_type = 12;
    r.model_size_bytes = 600ull << 20;
    r.forced_backend = forced;
    return r;
}

} // namespace

int main() {
    std::cout << "== test_p26_backend_selection ==\n";
    auto& hw = HardwareDetector::instance();
    auto& orch = EngineOrchestrator::instance();
    const bool compiled = HardwareDetector::cuda_inference_compiled();
    std::cout << "  CUDA inference compiled in: " << (compiled ? "yes" : "no") << "\n";

    // ---- HardwareDetector CUDA accessors -------------------------------------
    hw.set_mock_environment({cpuDevice(), nvidiaDevice(8)}, 16ull << 30, true, true);
    AGR_CHECK(hw.has_cuda());
    AGR_CHECK(hw.best_cuda_gpu() != nullptr);
    AGR_CHECK(hw.best_cuda_gpu()->cuda_compute_major == 8);
    AGR_CHECK(hw.best_cuda_gpu()->vram_bytes == (8ull << 30));

    hw.set_mock_environment({cpuDevice(), igpuVulkanOnly()}, 16ull << 30, true, false);
    AGR_CHECK(!hw.has_cuda());
    AGR_CHECK(hw.best_cuda_gpu() == nullptr);

    // ---- Registration reflects the build configuration ---------------------------
    orch.reset();
    hw.set_mock_environment({cpuDevice(), nvidiaDevice(8)}, 16ull << 30, true, true);
    register_default_backends(orch, hw);
    bool cuda_registered = false;
    for (const auto& b : orch.backends()) if (b.backend_id == "cuda") cuda_registered = true;
    AGR_CHECK(cuda_registered == compiled);

    if (compiled) {
        // Auto selection with a CUDA-capable device must choose CUDA.
        auto sel = orch.select_backend(request());
        AGR_CHECK(sel.selected_backend == "cuda");
        // Forced CUDA is honoured.
        auto forced = orch.select_backend(request("cuda"));
        AGR_CHECK(forced.selected_backend == "cuda");
        AGR_CHECK(!forced.is_fallback);

        // No CUDA hardware -> never selected, even when forced.
        hw.set_mock_environment({cpuDevice(), igpuVulkanOnly()}, 16ull << 30, true, false);
        auto no_hw = orch.select_backend(request());
        AGR_CHECK(no_hw.selected_backend != "cuda");
        auto forced_no_hw = orch.select_backend(request("cuda"));
        AGR_CHECK(forced_no_hw.selected_backend == "hydroxl_cpu");
        AGR_CHECK(forced_no_hw.is_fallback);
        AGR_CHECK(!forced_no_hw.rejected_backends.empty());

        // Model larger than VRAM -> CUDA rejected with a VRAM reason.
        hw.set_mock_environment({cpuDevice(), nvidiaDevice(1)}, 64ull << 30, true, true);
        auto big = request();
        big.model_size_bytes = 2ull << 30;
        auto big_sel = orch.select_backend(big);
        AGR_CHECK(big_sel.selected_backend != "cuda");
        bool vram_reason = false;
        for (const auto& r : big_sel.rejected_backends)
            if (r.find("cuda") == 0 && r.find("VRAM") != std::string::npos) vram_reason = true;
        AGR_CHECK(vram_reason);
    } else {
        // CUDA-disabled build: backend absent; forced CUDA falls back with an explicit reason.
        auto sel = orch.select_backend(request());
        AGR_CHECK(sel.selected_backend != "cuda");
        auto forced = orch.select_backend(request("cuda"));
        AGR_CHECK(forced.selected_backend == "hydroxl_cpu");
        AGR_CHECK(forced.is_fallback);
        bool names_option = false;
        for (const auto& r : forced.rejected_backends)
            if (r.find("HETEROACCEL_ENABLE_CUDA") != std::string::npos) names_option = true;
        AGR_CHECK(names_option);

        // Forced CUDA with fallback disallowed must fail, not silently run elsewhere.
        auto strict = request("cuda");
        strict.allow_cpu_fallback = false;
        auto strict_sel = orch.select_backend(strict);
        AGR_CHECK(strict_sel.selected_backend.empty());
        std::string err;
        AGR_CHECK(orch.create_engine(strict_sel, strict, err) == nullptr);
        AGR_CHECK(!err.empty());
    }

    // ---- Runtime fallback: CUDA engine fails to initialize ----------------------
    // (uses test doubles registered through the public factory API)
    orch.reset();
    hw.set_mock_environment({cpuDevice(), nvidiaDevice(8)}, 16ull << 30, true, true);
    BackendCapabilities fake_cuda;
    fake_cuda.backend_id = "cuda";
    fake_cuda.requires_cuda = true;
    fake_cuda.relative_throughput = 100.0f;
    orch.register_backend(fake_cuda);
    BackendCapabilities cpu;
    cpu.backend_id = "hydroxl_cpu";
    orch.register_backend(cpu);
    orch.register_engine_factory("cuda", [] {
        return std::make_shared<FakeEngine>("cuda", false, "CUDA error: out of memory (simulated)");
    });
    orch.register_engine_factory("hydroxl_cpu", [] { return std::make_shared<FakeEngine>("hydroxl_cpu", true); });

    {
        auto req = request();
        auto sel = orch.select_backend(req);
        AGR_CHECK(sel.selected_backend == "cuda");
        std::string err;
        auto eng = orch.create_engine(sel, req, err);
        AGR_CHECK(eng != nullptr);
        AGR_CHECK(eng && eng->backend_id() == "hydroxl_cpu");
        auto rep = orch.last_creation_report();
        AGR_CHECK(rep.fell_back);
        AGR_CHECK(rep.selected_backend == "cuda");
        AGR_CHECK(rep.actual_backend == "hydroxl_cpu");
        AGR_CHECK(rep.failure_reason.find("out of memory") != std::string::npos);
    }
    {
        // Fallback disallowed: failure is surfaced with the reason.
        auto req = request();
        req.allow_cpu_fallback = false;
        auto sel = orch.select_backend(req);
        std::string err;
        AGR_CHECK(orch.create_engine(sel, req, err) == nullptr);
        AGR_CHECK(err.find("out of memory") != std::string::npos);
        AGR_CHECK(orch.last_creation_report().actual_backend.empty());
    }
    {
        // Both primary and CPU fail: both reasons are reported.
        orch.register_engine_factory("hydroxl_cpu", [] {
            return std::make_shared<FakeEngine>("hydroxl_cpu", false, "cpu load failed (simulated)");
        });
        auto req = request();
        auto sel = orch.select_backend(req);
        std::string err;
        AGR_CHECK(orch.create_engine(sel, req, err) == nullptr);
        AGR_CHECK(err.find("out of memory") != std::string::npos);
        AGR_CHECK(err.find("cpu load failed") != std::string::npos);
    }
    {
        // Healthy primary: no fallback reported.
        orch.register_engine_factory("cuda", [] { return std::make_shared<FakeEngine>("cuda", true); });
        orch.register_engine_factory("hydroxl_cpu", [] { return std::make_shared<FakeEngine>("hydroxl_cpu", true); });
        auto req = request();
        auto sel = orch.select_backend(req);
        std::string err;
        auto eng = orch.create_engine(sel, req, err);
        AGR_CHECK(eng && eng->backend_id() == "cuda");
        auto rep = orch.last_creation_report();
        AGR_CHECK(!rep.fell_back);
        AGR_CHECK(rep.actual_backend == "cuda");
    }

    orch.reset();
    hw.reset();
    AGR_TEST_MAIN_END();
}
