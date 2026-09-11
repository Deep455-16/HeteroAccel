#include "mini_test.h"
#include "hardware/GPUDetector.h"

int main() {
    std::cout << "== test_gpu_detector ==\n";
    // Primary contract for Phase 1: this call must never throw or crash,
    // regardless of whether a GPU is physically present (e.g. a headless
    // CI machine legitimately has zero GPUs).
    std::vector<agr::GPUInfo> gpus = agr::GPUDetector::detect();

    std::cout << "  detected " << gpus.size() << " GPU(s)\n";
    for (const auto& g : gpus) {
        std::cout << "    - " << g.name << " (" << agr::toString(g.vendor)
                  << ", " << agr::toString(g.kind) << ")\n";
    }

    // No assumption that gpus is non-empty -- an empty result on a
    // headless machine is the correct, honest answer.
    AGR_CHECK(gpus.size() == gpus.size()); // trivially true: call above didn't throw

    AGR_TEST_MAIN_END();
}
