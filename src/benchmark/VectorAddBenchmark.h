#pragma once
#include "gpu/GPUBackend.h"

#include <cstddef>
#include <string>
#include <vector>

namespace agr {

struct BenchmarkSizeResult {
    size_t element_count = 0;

    double cpu_time_ms = 0.0;

    bool gpu_ran = false;
    std::string gpu_skip_reason; // populated when gpu_ran == false

    double gpu_upload_ms = 0.0;
    double gpu_execute_ms = 0.0;
    double gpu_download_ms = 0.0;
    double gpu_total_ms = 0.0; // upload + execute + download, nothing else

    bool correctness_checked = false;
    bool correctness_passed = false;

    // cpu_time_ms / gpu_total_ms. Only meaningful when gpu_ran is true;
    // left at 0 otherwise rather than fabricating a number.
    double speedup = 0.0;
};

struct BenchmarkReport {
    std::string device_name; // empty if GPU was never available
    std::vector<BenchmarkSizeResult> results;
};

class VectorAddBenchmark {
public:
    // Runs CPU-vs-GPU vector addition across each requested element count.
    // If backend.isAvailable() is false, every result still includes the
    // real CPU timing -- only the GPU columns are marked N/A with a reason.
    static BenchmarkReport run(GPUBackend& backend, const std::vector<size_t>& sizes);

    static std::string formatText(const BenchmarkReport& report);
};

} // namespace agr
