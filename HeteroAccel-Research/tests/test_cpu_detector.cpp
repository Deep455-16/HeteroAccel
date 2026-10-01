#include "mini_test.h"
#include "hardware/CPUDetector.h"

int main() {
    std::cout << "== test_cpu_detector ==\n";
    agr::CPUInfo cpu = agr::CPUDetector::detect();

    std::cout << "  detected: vendor=" << cpu.vendor
              << " model=" << cpu.model_name
              << " logical=" << cpu.logical_processors
              << " physical=" << cpu.physical_cores
              << " arch=" << cpu.architecture << "\n";

    AGR_CHECK(!cpu.vendor.empty());
    AGR_CHECK(!cpu.model_name.empty());
    AGR_CHECK(cpu.logical_processors > 0);
    AGR_CHECK(cpu.physical_cores > 0);
    AGR_CHECK(cpu.physical_cores <= cpu.logical_processors);
    AGR_CHECK(!cpu.architecture.empty());
    AGR_CHECK(cpu.architecture != "unknown");

    AGR_TEST_MAIN_END();
}
