// tests/test_mem_invalid_ops.cpp
#include "mini_test.h"
#include "gpu/VulkanBackend.h"
#include "mem/MemoryManager.h"

int main() {
    std::cout << "== test_mem_invalid_ops ==\n";
    
    agr::VulkanBackend backend; // no init
    agr::MemoryManager mm(backend);
    
    // Invalid size
    agr::MemoryBlock b = mm.allocate(0, agr::MemoryLocation::CPU);
    AGR_CHECK(!b.isValid());
    
    // Double free (should not crash)
    agr::MemoryBlock b2 = mm.allocate(128, agr::MemoryLocation::CPU);
    AGR_CHECK(b2.isValid());
    agr::MemoryBlock copy = b2;
    
    mm.release(b2);
    AGR_CHECK(!b2.isValid());
    mm.release(copy); // double free using copied invalid id, should safely ignore
    
    AGR_TEST_MAIN_END();
}
