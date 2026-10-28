// tests/test_p22_vulkan_ops.cpp
// Phase 22: Vulkan Compute Operations Tests

#include "mini_test.h"
#include "hydroxl/VulkanTensorOps.h"
#include "hydroxl/HydroXLOps.h"
#include "orchestration/HardwareDetector.h"

#include <cmath>
#include <vector>
#include <random>
#include <iostream>

using namespace agr;
using namespace agr::hydroxl;

int main() {
    auto& hw = orchestration::HardwareDetector::instance();
    hw.detect();
    if (!hw.has_vulkan()) {
        std::cout << "[SKIP] Vulkan not available on this platform\n";
        return 77;
    }

    std::cout << "--- Phase 22: VulkanTensorOps Test Suite ---\n";

    VulkanTensorOps ops;
    std::string err;

    // Test 1: VulkanTensorOps::initialize() succeeds
    bool init_ok = ops.initialize(err);
    if (!init_ok) {
        std::cerr << "Initialization error: " << err << "\n";
    }
    AGR_CHECK(init_ok);
    AGR_CHECK(ops.isReady());
    std::cout << "[INFO] Vulkan device: " << ops.deviceName() << "\n";

    // Test 2: uploadTensor + downloadTensor roundtrip
    {
        constexpr uint32_t N = 1024;
        std::vector<float> src(N);
        for (uint32_t i = 0; i < N; ++i) {
            src[i] = static_cast<float>(i) * 0.125f - 10.0f;
        }

        bool up_ok = ops.uploadTensor("test_t2", src.data(), N);
        AGR_CHECK(up_ok);

        std::vector<float> dst(N, 0.0f);
        bool down_ok = ops.downloadTensor("test_t2", dst.data(), N);
        AGR_CHECK(down_ok);

        bool match = true;
        for (uint32_t i = 0; i < N; ++i) {
            if (std::abs(src[i] - dst[i]) > 1e-6f) {
                match = false;
                break;
            }
        }
        AGR_CHECK(match);
    }

    // Test 3: matvecHost vs CPU matvec reference (64x128 matrix, tolerance 1e-4)
    {
        constexpr uint32_t M = 64;
        constexpr uint32_t K = 128;
        std::vector<float> A(M * K);
        std::vector<float> x(K);
        std::vector<float> cpu_out(M);
        std::vector<float> gpu_out(M);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : A) v = dist(rng);
        for (auto& v : x) v = dist(rng);

        ops.uploadTensor("mat_test", A.data(), M * K);
        bool mv_ok = ops.matvecHost("mat_test", x.data(), gpu_out.data(), M, K);
        AGR_CHECK(mv_ok);

        matvec(A.data(), x.data(), cpu_out.data(), M, K);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < M; ++i) {
            float d = std::abs(gpu_out[i] - cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] Matvec max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-4f);
    }

    // Test 4: rmsnorm vs CPU reference (n=512, eps=1e-6, tolerance 1e-5)
    {
        constexpr uint32_t N = 512;
        constexpr float eps = 1e-6f;
        std::vector<float> x(N);
        std::vector<float> w(N);
        std::vector<float> cpu_out(N);
        std::vector<float> gpu_out(N);

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (auto& v : x) v = dist(rng);
        for (auto& v : w) v = dist(rng);

        ops.uploadTensor("norm_weight", w.data(), N);
        bool norm_ok = ops.rmsnorm(x.data(), "norm_weight", gpu_out.data(), N, eps);
        AGR_CHECK(norm_ok);

        rms_norm(x.data(), w.data(), cpu_out.data(), N, eps);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float d = std::abs(gpu_out[i] - cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] RMSNorm max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-5f);
    }

    // Test 5: elemwise residual_add vs CPU reference (n=512, tolerance 1e-6)
    {
        constexpr uint32_t N = 512;
        std::vector<float> a(N);
        std::vector<float> b(N);
        std::vector<float> cpu_out(N);
        std::vector<float> gpu_out(N);

        std::mt19937 rng(999);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& v : a) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        cpu_out = a;
        residual_add(cpu_out.data(), b.data(), N);

        bool add_ok = ops.elemwise(a.data(), b.data(), gpu_out.data(), N, 0); // op=0 (add)
        AGR_CHECK(add_ok);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float d = std::abs(gpu_out[i] - cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] Residual add max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-6f);
    }

    // Test 6: elemwise silu vs CPU reference (n=512, tolerance 1e-5)
    {
        constexpr uint32_t N = 512;
        std::vector<float> x(N);
        std::vector<float> cpu_out(N);
        std::vector<float> gpu_out(N);

        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
        for (auto& v : x) v = dist(rng);

        cpu_out = x;
        silu_inplace(cpu_out.data(), N);

        bool silu_ok = ops.elemwise(x.data(), nullptr, gpu_out.data(), N, 1); // op=1 (silu)
        AGR_CHECK(silu_ok);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float d = std::abs(gpu_out[i] - cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] SiLU max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-5f);
    }

    // Test 7: elemwise swiglu vs CPU reference (n=512, tolerance 1e-5)
    {
        constexpr uint32_t N = 512;
        std::vector<float> gate(N);
        std::vector<float> up(N);
        std::vector<float> cpu_out(N);
        std::vector<float> gpu_out(N);

        std::mt19937 rng(888);
        std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
        for (auto& v : gate) v = dist(rng);
        for (auto& v : up) v = dist(rng);

        swiglu(gate.data(), up.data(), cpu_out.data(), N);

        bool swiglu_ok = ops.elemwise(gate.data(), up.data(), gpu_out.data(), N, 2); // op=2 (swiglu)
        AGR_CHECK(swiglu_ok);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float d = std::abs(gpu_out[i] - cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] SwiGLU max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-5f);
    }

    // Test 8: softmax vs CPU reference (n=512, tolerance 1e-5)
    {
        constexpr uint32_t N = 512;
        std::vector<float> x_cpu(N);
        std::vector<float> x_gpu(N);

        std::mt19937 rng(555);
        std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
        for (uint32_t i = 0; i < N; ++i) {
            float v = dist(rng);
            x_cpu[i] = v;
            x_gpu[i] = v;
        }

        softmax_inplace(x_cpu.data(), N);
        bool sm_ok = ops.softmax(x_gpu.data(), N);
        AGR_CHECK(sm_ok);

        float max_diff = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            float d = std::abs(x_gpu[i] - x_cpu[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "[INFO] Softmax max diff: " << max_diff << "\n";
        AGR_CHECK(max_diff < 1e-5f);
    }

    ops.shutdown();
    AGR_CHECK(!ops.isReady());

    AGR_TEST_MAIN_END();
}
