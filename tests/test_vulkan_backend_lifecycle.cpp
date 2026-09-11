#include "mini_test.h"
#include "gpu/VulkanBackend.h"

int main() {
    std::cout << "== test_vulkan_backend_lifecycle ==\n";

    agr::VulkanBackend backend;
    bool ok = backend.initialize();
    std::cout << "  initialize() -> " << (ok ? "true" : "false") << "\n";

    if (ok) {
        std::cout << "  device: " << backend.deviceName() << "\n";
        AGR_CHECK(!backend.deviceName().empty());
        AGR_CHECK(backend.isAvailable());
    } else {
        std::cout << "  reason: " << backend.lastError() << "\n";
        // A false result must always come with a reason -- never a
        // silent, unexplained failure.
        AGR_CHECK(!backend.lastError().empty());
        AGR_CHECK(!backend.isAvailable());
    }

    // shutdown() must be safe to call regardless of init outcome, and
    // safe to call twice.
    backend.shutdown();
    backend.shutdown();
    AGR_CHECK(true);

    AGR_TEST_MAIN_END();
}
