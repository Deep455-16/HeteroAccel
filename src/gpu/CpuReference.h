#pragma once
#include <cstddef>
#include <cstdint>

namespace agr {

// Plain scalar CPU implementation. This is the ground truth Phase 2
// validates GPU results against, and the CPU side of the benchmark.
class CpuReference {
public:
    static void vectorAdd(const float* a, const float* b, float* c, size_t count);

    // Compares two float arrays with an absolute epsilon (vector addition
    // of well-scaled test data has no meaningful rounding error at
    // float32 precision, but a tiny epsilon avoids flakiness from
    // reordered FP summation between CPU and GPU).
    static bool nearlyEqual(const float* expected, const float* actual, size_t count, float epsilon = 1e-4f);
};

} // namespace agr
