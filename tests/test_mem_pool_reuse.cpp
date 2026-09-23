// tests/test_mem_pool_reuse.cpp
#include "mini_test.h"
#include "mem/CPUAllocator.h"
#include "mem/MemoryPool.h"

int main() {
    std::cout << "== test_mem_pool_reuse ==\n";
    
    agr::CPUAllocator alloc;
    agr::MemoryPool pool(alloc, 1024, 4);
    
    agr::MemoryBlock b1 = pool.acquire();
    agr::MemoryBlock b2 = pool.acquire();
    AGR_CHECK(pool.activeCount() == 2);
    AGR_CHECK(pool.totalCreated() == 2);
    
    pool.release(b1);
    AGR_CHECK(pool.freeCount() == 1);
    
    agr::MemoryBlock b3 = pool.acquire();
    AGR_CHECK(pool.reuseCount() == 1);
    AGR_CHECK(pool.totalCreated() == 2); // no new allocation
    
    pool.release(b2);
    pool.release(b3);
    
    AGR_TEST_MAIN_END();
}
