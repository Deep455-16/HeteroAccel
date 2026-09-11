#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include <vector>

constexpr int kSkipReturnCode = 77;

int main() {
    std::cout << "== test_vulkan_buffer_transfer ==\n";

    agr::VulkanBackend backend;
    if (!backend.initialize()) {
        std::cout << "  SKIPPED: no usable Vulkan compute device in this environment ("
                  << backend.lastError() << ")\n";
        return kSkipReturnCode;
    }

    // Two sizes: small (well within any single heap) and larger (more
    // likely to exercise a real allocation rather than a trivially small
    // one), both round-tripped byte-for-byte.
    for (size_t n : {size_t(16), size_t(1 << 20)}) {
        std::vector<float> original(n);
        for (size_t i = 0; i < n; ++i) original[i] = static_cast<float>(i) * 1.5f - 3.0f;

        agr::Buffer buf = backend.createBuffer(n * sizeof(float));
        AGR_CHECK(buf.id != 0);

        bool up = backend.upload(buf, original.data(), n, nullptr);
        AGR_CHECK(up);

        std::vector<float> roundTrip(n, 0.0f);
        bool down = backend.download(buf, roundTrip.data(), n, nullptr);
        AGR_CHECK(down);

        bool identical = (roundTrip == original);
        std::cout << "  n=" << n << " round-trip identical: " << (identical ? "yes" : "no") << "\n";
        AGR_CHECK(identical);

        backend.destroyBuffer(buf);
    }

    backend.shutdown();
    AGR_TEST_MAIN_END();
}
