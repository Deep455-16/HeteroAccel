// tests/test_mem_priority_eviction.cpp
#include "mini_test.h"
#include "mem/EvictionPolicy.h"

int main() {
    std::cout << "== test_mem_priority_eviction ==\n";
    
    agr::EvictionPolicy policy;
    std::vector<agr::MemoryBlock> candidates(4);
    
    auto now = std::chrono::steady_clock::now();
    
    candidates[0].id = 1;
    candidates[0].priority = agr::MemoryPriority::CRITICAL;
    candidates[0].last_accessed = now - std::chrono::milliseconds(900); // very old, but CRITICAL
    
    candidates[1].id = 2;
    candidates[1].priority = agr::MemoryPriority::HIGH;
    candidates[1].last_accessed = now;
    
    candidates[2].id = 3;
    candidates[2].priority = agr::MemoryPriority::LOW;
    candidates[2].last_accessed = now; // very new, but LOW
    
    candidates[3].id = 4;
    candidates[3].priority = agr::MemoryPriority::NORMAL;
    candidates[3].last_accessed = now;
    
    std::vector<uint64_t> evict = policy.selectForEviction(candidates, 1);
    AGR_CHECK(evict.size() == 1);
    AGR_CHECK(evict[0] == 3); // should pick LOW priority despite being new
    
    // Test that CRITICAL is never evicted
    evict = policy.selectForEviction(candidates, 4);
    AGR_CHECK(evict.size() == 3); // 2, 3, 4 only
    
    AGR_TEST_MAIN_END();
}
