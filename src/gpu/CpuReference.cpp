#include "gpu/CpuReference.h"
#include <cmath>

namespace agr {

void CpuReference::vectorAdd(const float* a, const float* b, float* c, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        c[i] = a[i] + b[i];
    }
}

bool CpuReference::nearlyEqual(const float* expected, const float* actual, size_t count, float epsilon) {
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(expected[i] - actual[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

} // namespace agr
