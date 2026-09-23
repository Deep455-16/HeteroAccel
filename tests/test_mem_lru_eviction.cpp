// tests/test_mem_lru_eviction.cpp
#include "mini_test.h"
#include "mem/MemoryTypes.h"
#include "mem/EvictionPolicy.h"
#include <chrono>
#include <thread>

int main() {
    std::cout << "== test_mem_lru_eviction ==\n";
    
    agr::EvictionPolicy policy;
    std::vector<agr::MemoryBlock> candidates(3);
    
    candidates[0].id = 1;
    candidates[0].priority = agr::MemoryPriority::NORMAL;
    candidates[0].last_accessed = std::chrono::steady_clock::now() - std::chrono::milliseconds(500); // oldest
    
    candidates[1].id = 2;
    candidates[1].priority = agr::MemoryPriority::NORMAL;
    candidates[1].last_accessed = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);
    
    candidates[2].id = 3;
    candidates[2].priority = agr::MemoryPriority::NORMAL;
    candidates[2].last_accessed = std::chrono::steady_clock::now(); // newest
    
    std::vector<uint64_t> evict = policy.selectForEviction(candidates, 1);
    AGR_CHECK(evict.size() == 1);
    AGR_CHECK(evict[0] == 1); // should pick oldest
    
    AGR_TEST_MAIN_END();
}
