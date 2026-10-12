// tests/test_p16_tensor.cpp
// Phase 16: Tensor construction, move semantics, element access, views.

#include "primitives/Tensor.h"
#include "mini_test.h"
#include <iostream>
#include <cmath>

using namespace agr::primitives;

int main() {
    // Default Tensor is invalid
    {
        Tensor t;
        AGR_CHECK(!t.isValid());
        AGR_CHECK(t.dataAs<float>() == nullptr);
    }

    // Invalid shape (UNKNOWN dtype) -> invalid Tensor
    {
        TensorShape bad_shape({4}, DataType::UNKNOWN);
        Tensor t = Tensor::allocate(bad_shape);
        AGR_CHECK(!t.isValid());
    }

    // Allocate FLOAT32 1D
    {
        TensorShape shape({8}, DataType::FLOAT32);
        Tensor t = Tensor::allocate(shape);
        AGR_CHECK(t.isValid());
        AGR_CHECK(t.elementCount() == 8);
        AGR_CHECK(t.byteSize()     == 32);
        AGR_CHECK(t.dataAs<float>() != nullptr);
    }

    // zeros() initialises to zero
    {
        TensorShape shape({4}, DataType::FLOAT32);
        Tensor t = Tensor::zeros(shape);
        AGR_CHECK(t.isValid());
        const float* p = t.dataAs<const float>();
        AGR_CHECK(p != nullptr);
        for (uint64_t i = 0; i < 4; ++i) AGR_CHECK(p[i] == 0.0f);
    }

    // at<T>() bounds checking
    {
        TensorShape shape({3}, DataType::FLOAT32);
        Tensor t = Tensor::zeros(shape);
        float* p0 = t.at<float>(0);
        float* p2 = t.at<float>(2);
        float* p3 = t.at<float>(3); // out of bounds
        AGR_CHECK(p0 != nullptr);
        AGR_CHECK(p2 != nullptr);
        AGR_CHECK(p3 == nullptr);

        *p0 = 42.0f;
        AGR_CHECK(*t.at<float>(0) == 42.0f);
    }

    // Move construction
    {
        TensorShape shape({4}, DataType::FLOAT32);
        Tensor a = Tensor::zeros(shape);
        *a.at<float>(0) = 7.0f;
        Tensor b(std::move(a));
        AGR_CHECK(!a.isValid());
        AGR_CHECK(b.isValid());
        AGR_CHECK(*b.at<float>(0) == 7.0f);
    }

    // Move assignment
    {
        TensorShape shape({2}, DataType::FLOAT32);
        Tensor a = Tensor::zeros(shape);
        *a.at<float>(1) = 99.0f;
        Tensor b;
        b = std::move(a);
        AGR_CHECK(!a.isValid());
        AGR_CHECK(b.isValid());
        AGR_CHECK(*b.at<float>(1) == 99.0f);
    }

    // Overflow shape -> invalid Tensor (not allocated)
    {
        TensorShape huge({(size_t)1 << 32, (size_t)1 << 32}, DataType::FLOAT32);
        Tensor t = Tensor::allocate(huge);
        AGR_CHECK(!t.isValid());
    }

    // TensorView basic usage
    {
        TensorShape shape({5}, DataType::FLOAT32);
        Tensor t = Tensor::zeros(shape);
        *t.at<float>(3) = 13.0f;

        TensorView v(t);
        AGR_CHECK(v.isValid());
        AGR_CHECK(v.elementCount() == 5);
        AGR_CHECK(*v.at<float>(3) == 13.0f);
        AGR_CHECK(v.at<float>(5) == nullptr); // out of bounds
    }

    // TensorView on invalid Tensor
    {
        Tensor bad;
        TensorView v(bad);
        AGR_CHECK(!v.isValid());
    }

    std::cout << "test_p16_tensor PASSED\n";
    AGR_TEST_MAIN_END();
}
