// tests/test_mem_statistics.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "mem/MemoryManager.h"

int main() {
    std::cout << "== test_mem_statistics ==\n";
    
    agr::VulkanBackend backend;
    backend.initialize(); // fine if it fails, fallback to CPU mode testing
    
    agr::MemoryManager mm(backend);
    agr::MemoryBlock b = mm.allocate(1024, agr::MemoryLocation::CPU);
    AGR_CHECK(b.isValid());
    
    agr::MemoryStats stats = mm.statistics();
    AGR_CHECK(stats.cpu_used_bytes == 1024);
    AGR_CHECK(stats.allocation_count > 0);
    
    mm.release(b);
    stats = mm.statistics();
    AGR_CHECK(stats.cpu_used_bytes == 0);
    
    AGR_TEST_MAIN_END();
}
