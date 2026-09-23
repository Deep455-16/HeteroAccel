// tests/test_mem_cpu_alloc.cpp
#include "mini_test.h"
#include "mem/CPUAllocator.h"

int main() {
    std::cout << "== test_mem_cpu_alloc ==\n";
    agr::CPUAllocator alloc;
    
    AGR_CHECK(alloc.used() == 0);
    size_t initialAvail = alloc.available();
    
    // Allocate 100 MB
    size_t size = 100 * 1024 * 1024;
    agr::MemoryBlock b = alloc.allocate(size);
    AGR_CHECK(b.isValid());
    AGR_CHECK(b.size == size);
    AGR_CHECK(b.location == agr::MemoryLocation::CPU);
    AGR_CHECK(alloc.used() == size);
    
    // Release
    alloc.free(b);
    AGR_CHECK(!b.isValid());
    AGR_CHECK(alloc.used() == 0);
    
    AGR_TEST_MAIN_END();
}
