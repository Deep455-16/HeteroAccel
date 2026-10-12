// tests/test_p16_cpu_ops.cpp
// Phase 16: CPU tensor operations (copy, fill, add, multiply, scale, sum, maxVal).
// Uses known values and exact/near comparison.

#include "primitives/Tensor.h"
#include "primitives/cpu/CpuOps.h"
#include "mini_test.h"
#include <iostream>
#include <cmath>

using namespace agr::primitives;

static TensorShape F32(std::initializer_list<size_t> dims) {
    return TensorShape(std::vector<size_t>(dims), DataType::FLOAT32);
}

static bool near(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

int main() {
    // ---------- fill ----------
    {
        Tensor t = Tensor::zeros(F32({4}));
        auto r = cpu::fill(t, 3.14f);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i) AGR_CHECK(near(*t.at<float>(i), 3.14f));
    }

    // fill on invalid tensor
    {
        Tensor bad;
        auto r = cpu::fill(bad, 1.0f);
        AGR_CHECK(!r.success);
    }

    // fill on wrong dtype
    {
        Tensor t = Tensor::zeros(TensorShape({4}, DataType::INT32));
        auto r = cpu::fill(t, 1.0f);
        AGR_CHECK(!r.success);
    }

    // ---------- copy ----------
    {
        Tensor src = Tensor::zeros(F32({5}));
        cpu::fill(src, 7.0f);
        Tensor dst = Tensor::zeros(F32({5}));
        auto r = cpu::copy(dst, src);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 5; ++i) AGR_CHECK(near(*dst.at<float>(i), 7.0f));
    }

    // copy shape mismatch
    {
        Tensor a = Tensor::zeros(F32({3}));
        Tensor b = Tensor::zeros(F32({4}));
        auto r = cpu::copy(a, b);
        AGR_CHECK(!r.success);
    }

    // ---------- add ----------
    {
        Tensor a = Tensor::zeros(F32({4}));
        Tensor b = Tensor::zeros(F32({4}));
        Tensor c = Tensor::zeros(F32({4}));
        cpu::fill(a, 1.0f);
        cpu::fill(b, 2.0f);
        auto r = cpu::add(c, a, b);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i) AGR_CHECK(near(*c.at<float>(i), 3.0f));
    }

    // add shape mismatch
    {
        Tensor a = Tensor::zeros(F32({3}));
        Tensor b = Tensor::zeros(F32({4}));
        Tensor c = Tensor::zeros(F32({3}));
        auto r = cpu::add(c, a, b);
        AGR_CHECK(!r.success);
    }

    // add dtype mismatch
    {
        Tensor a = Tensor::zeros(TensorShape({3}, DataType::INT32));
        Tensor b = Tensor::zeros(F32({3}));
        Tensor c = Tensor::zeros(F32({3}));
        auto r = cpu::add(c, a, b);
        AGR_CHECK(!r.success);
    }

    // ---------- multiply ----------
    {
        Tensor a = Tensor::zeros(F32({3}));
        Tensor b = Tensor::zeros(F32({3}));
        Tensor c = Tensor::zeros(F32({3}));
        cpu::fill(a, 3.0f);
        cpu::fill(b, 4.0f);
        auto r = cpu::multiply(c, a, b);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 3; ++i) AGR_CHECK(near(*c.at<float>(i), 12.0f));
    }

    // ---------- scale ----------
    {
        Tensor src = Tensor::zeros(F32({4}));
        Tensor dst = Tensor::zeros(F32({4}));
        cpu::fill(src, 5.0f);
        auto r = cpu::scale(dst, src, 2.0f);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 4; ++i) AGR_CHECK(near(*dst.at<float>(i), 10.0f));
    }

    // scale alias: dst == src
    {
        Tensor t = Tensor::zeros(F32({3}));
        cpu::fill(t, 3.0f);
        auto r = cpu::scale(t, t, -1.0f);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 3; ++i) AGR_CHECK(near(*t.at<float>(i), -3.0f));
    }

    // ---------- sum ----------
    {
        Tensor t = Tensor::zeros(F32({5}));
        float* p = t.dataAs<float>();
        for (int i = 0; i < 5; ++i) p[i] = static_cast<float>(i + 1); // 1+2+3+4+5 = 15
        double s = 0.0;
        auto r = cpu::sum(t, &s);
        AGR_CHECK(r.success);
        AGR_CHECK(std::fabs(s - 15.0) < 1e-4);
    }

    // sum null out_sum
    {
        Tensor t = Tensor::zeros(F32({3}));
        auto r = cpu::sum(t, nullptr);
        AGR_CHECK(!r.success);
    }

    // ---------- maxVal ----------
    {
        Tensor t = Tensor::zeros(F32({6}));
        float* p = t.dataAs<float>();
        p[0] = -10.0f; p[1] = 3.0f; p[2] = 7.5f; p[3] = 2.0f; p[4] = 7.5f; p[5] = -1.0f;
        float m = 0.0f;
        auto r = cpu::maxVal(t, &m);
        AGR_CHECK(r.success);
        AGR_CHECK(near(m, 7.5f));
    }

    // maxVal null out_max
    {
        Tensor t = Tensor::zeros(F32({3}));
        auto r = cpu::maxVal(t, nullptr);
        AGR_CHECK(!r.success);
    }

    // maxVal wrong dtype
    {
        Tensor t = Tensor::zeros(TensorShape({3}, DataType::INT32));
        float m = 0.0f;
        auto r = cpu::maxVal(t, &m);
        AGR_CHECK(!r.success);
    }

    // ---------- add alias (dst == a) ----------
    {
        Tensor a = Tensor::zeros(F32({3}));
        Tensor b = Tensor::zeros(F32({3}));
        cpu::fill(a, 5.0f);
        cpu::fill(b, 2.0f);
        // dst aliases a; should read a[i] before write
        auto r = cpu::add(a, a, b);
        AGR_CHECK(r.success);
        for (uint64_t i = 0; i < 3; ++i) AGR_CHECK(near(*a.at<float>(i), 7.0f));
    }

    std::cout << "test_p16_cpu_ops PASSED\n";
    AGR_TEST_MAIN_END();
}
