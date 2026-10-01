#include "mini_test.h"
#include "hardware/MemoryDetector.h"

int main() {
    std::cout << "== test_memory_detector ==\n";
    agr::MemoryInfo mem = agr::MemoryDetector::detect();

    std::cout << "  detected: total_mb=" << mem.total_physical_mb
              << " available_mb=" << mem.available_physical_mb << "\n";

    AGR_CHECK(mem.total_physical_mb > 0);
    AGR_CHECK(mem.available_physical_mb <= mem.total_physical_mb);

    AGR_TEST_MAIN_END();
}
