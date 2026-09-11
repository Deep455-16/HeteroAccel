#include "mini_test.h"
#include "hardware/CUDADetector.h"

int main() {
    std::cout << "== test_cuda_detector ==\n";
    agr::CUDAInfo cuda = agr::CUDADetector::detect();

    std::cout << "  available=" << (cuda.available ? "true" : "false") << "\n";
    if (!cuda.available) {
        std::cout << "  reason=" << cuda.unavailable_reason << "\n";
        AGR_CHECK(!cuda.unavailable_reason.empty());
    } else {
        std::cout << "  driver_version=" << cuda.driver_version << "\n";
        AGR_CHECK(!cuda.devices.empty());
    }

    // CUDA absence must never be fatal -- process reaching this line
    // without crashing is itself the main thing being verified.
    AGR_CHECK(true);

    AGR_TEST_MAIN_END();
}
