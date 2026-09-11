#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "gpu/CpuReference.h"
#include <vector>

// CTest convention: this exit code marks the test as SKIPPED rather than
// PASSED or FAILED, so "no GPU in this environment" is never silently
// counted as a pass of the correctness check it did not actually run.
constexpr int kSkipReturnCode = 77;

int main() {
    std::cout << "== test_vulkan_vector_add_correctness ==\n";

    agr::VulkanBackend backend;
    if (!backend.initialize()) {
        std::cout << "  SKIPPED: no usable Vulkan compute device in this environment ("
                  << backend.lastError() << ")\n";
        return kSkipReturnCode;
    }

    std::cout << "  running real GPU dispatch on: " << backend.deviceName() << "\n";

    const size_t n = 4096;
    std::vector<float> a(n), b(n), cCpu(n), cGpu(n, -1.0f);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.25f;
    }
    agr::CpuReference::vectorAdd(a.data(), b.data(), cCpu.data(), n);

    agr::Buffer bufA = backend.createBuffer(n * sizeof(float));
    agr::Buffer bufB = backend.createBuffer(n * sizeof(float));
    agr::Buffer bufC = backend.createBuffer(n * sizeof(float));
    AGR_CHECK(bufA.id != 0);
    AGR_CHECK(bufB.id != 0);
    AGR_CHECK(bufC.id != 0);

    bool uploadedA = backend.upload(bufA, a.data(), n, nullptr);
    bool uploadedB = backend.upload(bufB, b.data(), n, nullptr);
    AGR_CHECK(uploadedA);
    AGR_CHECK(uploadedB);

    double execMs = -1.0;
    bool executed = backend.executeVectorAdd(bufA, bufB, bufC, static_cast<uint32_t>(n), &execMs);
    AGR_CHECK(executed);
    AGR_CHECK(execMs >= 0.0);
    std::cout << "  GPU execute time: " << execMs << " ms\n";

    bool downloaded = backend.download(bufC, cGpu.data(), n, nullptr);
    AGR_CHECK(downloaded);

    bool correct = agr::CpuReference::nearlyEqual(cCpu.data(), cGpu.data(), n);
    std::cout << "  correctness vs CPU reference: " << (correct ? "PASS" : "FAIL") << "\n";
    AGR_CHECK(correct);

    backend.destroyBuffer(bufA);
    backend.destroyBuffer(bufB);
    backend.destroyBuffer(bufC);
    backend.shutdown();

    AGR_TEST_MAIN_END();
}
