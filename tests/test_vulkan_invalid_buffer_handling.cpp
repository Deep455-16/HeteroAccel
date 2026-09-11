#include "mini_test.h"
#include "gpu/VulkanBackend.h"

// Exercises Phase 2's failure-path handling for invalid/unknown buffer
// handles. These checks intentionally do NOT require a real GPU: every
// operation below must reject an unknown Buffer::id via its internal
// map lookup *before* touching the device, so this test is meaningful
// (and runs for real) with or without Vulkan hardware present.

int main() {
    std::cout << "== test_vulkan_invalid_buffer_handling ==\n";

    agr::VulkanBackend backend;
    bool initialized = backend.initialize();
    if (initialized) {
        std::cout << "  backend available: yes (" << backend.deviceName() << ")\n";
    } else {
        std::cout << "  backend available: no (" << backend.lastError() << ")\n";
    }

    agr::Buffer invalid; // id == 0, never allocated
    AGR_CHECK(invalid.id == 0);

    float dummyIn[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dummyOut[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    bool uploadResult = backend.upload(invalid, dummyIn, 4, nullptr);
    std::cout << "  upload(invalid) -> " << (uploadResult ? "true (BUG)" : "false (correct)") << "\n";
    AGR_CHECK(!uploadResult);

    bool downloadResult = backend.download(invalid, dummyOut, 4, nullptr);
    std::cout << "  download(invalid) -> " << (downloadResult ? "true (BUG)" : "false (correct)") << "\n";
    AGR_CHECK(!downloadResult);

    agr::Buffer a, b, c; // all invalid (id == 0)
    bool execResult = backend.executeVectorAdd(a, b, c, 4, nullptr);
    std::cout << "  executeVectorAdd(invalid,invalid,invalid) -> "
              << (execResult ? "true (BUG)" : "false (correct)") << "\n";
    AGR_CHECK(!execResult);

    // destroyBuffer on an invalid handle must be a safe no-op, not a crash.
    agr::Buffer toDestroy;
    backend.destroyBuffer(toDestroy);
    AGR_CHECK(toDestroy.id == 0);

    // createBuffer(0) must fail cleanly rather than allocate a zero-sized
    // resource silently.
    agr::Buffer zeroSized = backend.createBuffer(0);
    std::cout << "  createBuffer(0) -> id=" << zeroSized.id << " (expected 0)\n";
    AGR_CHECK(zeroSized.id == 0);

    backend.shutdown();
    AGR_TEST_MAIN_END();
}
