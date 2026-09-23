// tests/test_mem_vulkan_alloc.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "mem/VulkanAllocator.h"

int main() {
    std::cout << "== test_mem_vulkan_alloc ==\n";
    
    agr::VulkanBackend backend;
    if (!backend.initialize()) {
        std::cout << "  SKIP: Vulkan not available.\n";
        return 77;
    }
    
    agr::VulkanAllocator alloc(backend);
    AGR_CHECK(alloc.used() == 0);
    
    size_t size = 1024 * 1024;
    agr::MemoryBlock b = alloc.allocate(size);
    AGR_CHECK(b.isValid());
    AGR_CHECK(b.location == agr::MemoryLocation::GPU);
    AGR_CHECK(alloc.used() == size);
    
    alloc.free(b);
    AGR_CHECK(!b.isValid());
    AGR_CHECK(alloc.used() == 0);
    
    AGR_TEST_MAIN_END();
}
