#include "mini_test.h"
#include "gpu/CpuReference.h"
#include <vector>

int main() {
    std::cout << "== test_cpu_reference_vector_add ==\n";

    const size_t n = 10000;
    std::vector<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = static_cast<float>(i) * 0.25f;
    }

    agr::CpuReference::vectorAdd(a.data(), b.data(), c.data(), n);

    bool allCorrect = true;
    for (size_t i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        if (c[i] != expected) { allCorrect = false; break; }
    }
    AGR_CHECK(allCorrect);

    // nearlyEqual must catch a real mismatch, not just rubber-stamp
    // anything close to correct.
    std::vector<float> wrong = c;
    wrong[n / 2] += 5.0f;
    AGR_CHECK(!agr::CpuReference::nearlyEqual(c.data(), wrong.data(), n));
    AGR_CHECK(agr::CpuReference::nearlyEqual(c.data(), c.data(), n));

    AGR_TEST_MAIN_END();
}
