// Regression test for: vkAllocateDescriptorSets failed on second + subsequent
// executeVectorAdd calls.
//
// Root cause: the descriptor pool was created without
// VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT, making vkFreeDescriptorSets
// a no-op per the Vulkan spec.  The single pool slot was therefore never
// returned, so every call after the first exhausted the pool.
//
// This test calls executeVectorAdd N_ITERS times in a row on the same backend
// instance and verifies that:
//   (a) every dispatch succeeds,
//   (b) every dispatch produces the correct result,
//   (c) the descriptor pool is therefore being properly reused.
//
// CTest exit code 77 = SKIP (no GPU in this environment).

#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "gpu/CpuReference.h"

#include <vector>

constexpr int kSkipReturnCode = 77;
constexpr int N_ITERS = 8;      // enough iterations to expose pool exhaustion
constexpr size_t N_ELEMS = 4096;

int main() {
    std::cout << "== test_vulkan_descriptor_reuse ==\n";
    std::cout << "  Iterations: " << N_ITERS << "  Elements per iteration: " << N_ELEMS << "\n";

    agr::VulkanBackend backend;
    if (!backend.initialize()) {
        std::cout << "  SKIPPED: no usable Vulkan compute device ("
                  << backend.lastError() << ")\n";
        return kSkipReturnCode;
    }
    std::cout << "  GPU device: " << backend.deviceName() << "\n";

    // Prepare CPU reference data once; reuse across iterations.
    std::vector<float> a(N_ELEMS), b(N_ELEMS), cCpu(N_ELEMS);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.25f;
    }
    agr::CpuReference::vectorAdd(a.data(), b.data(), cCpu.data(), N_ELEMS);

    // Allocate GPU buffers once; keep them alive across all iterations.
    agr::Buffer bufA = backend.createBuffer(N_ELEMS * sizeof(float));
    agr::Buffer bufB = backend.createBuffer(N_ELEMS * sizeof(float));
    agr::Buffer bufC = backend.createBuffer(N_ELEMS * sizeof(float));
    AGR_CHECK(bufA.id != 0);
    AGR_CHECK(bufB.id != 0);
    AGR_CHECK(bufC.id != 0);

    AGR_CHECK(backend.upload(bufA, a.data(), N_ELEMS, nullptr));
    AGR_CHECK(backend.upload(bufB, b.data(), N_ELEMS, nullptr));

    for (int iter = 0; iter < N_ITERS; ++iter) {
        double execMs = -1.0;

        // This is the call that used to fail on iter >= 1 with
        // "vkAllocateDescriptorSets failed" when FREE_DESCRIPTOR_SET_BIT
        // was missing from the pool.
        bool executed = backend.executeVectorAdd(
            bufA, bufB, bufC,
            static_cast<uint32_t>(N_ELEMS),
            &execMs);

        if (!executed) {
            std::cout << "  FAILED on iteration " << iter
                      << ": " << backend.lastError() << "\n";
        }
        AGR_CHECK(executed);
        AGR_CHECK(execMs >= 0.0);

        std::vector<float> cGpu(N_ELEMS, 0.0f);
        bool downloaded = backend.download(bufC, cGpu.data(), N_ELEMS, nullptr);
        AGR_CHECK(downloaded);

        bool correct = agr::CpuReference::nearlyEqual(cCpu.data(), cGpu.data(), N_ELEMS);
        std::cout << "  iter " << iter << ": execute=" << execMs
                  << " ms  correct=" << (correct ? "yes" : "NO") << "\n";
        AGR_CHECK(correct);
    }

    backend.destroyBuffer(bufA);
    backend.destroyBuffer(bufB);
    backend.destroyBuffer(bufC);
    backend.shutdown();

    std::cout << "  All " << N_ITERS << " iterations completed successfully.\n";
    AGR_TEST_MAIN_END();
}
