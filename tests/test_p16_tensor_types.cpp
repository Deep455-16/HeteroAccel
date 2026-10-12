// tests/test_p16_tensor_types.cpp
// Phase 16: Tensor shape, dtype, element count, byte size, overflow tests.

#include "primitives/TensorTypes.h"
#include "mini_test.h"
#include <iostream>
#include <cstdint>

using namespace agr::primitives;

int main() {
    // elementSize
    AGR_CHECK(elementSize(DataType::FLOAT32) == 4);
    AGR_CHECK(elementSize(DataType::FLOAT16) == 2);
    AGR_CHECK(elementSize(DataType::INT32)   == 4);
    AGR_CHECK(elementSize(DataType::INT16)   == 2);
    AGR_CHECK(elementSize(DataType::INT8)    == 1);
    AGR_CHECK(elementSize(DataType::UINT8)   == 1);
    AGR_CHECK(elementSize(DataType::BOOL)    == 1);
    AGR_CHECK(elementSize(DataType::UNKNOWN) == 0);

    // toString
    AGR_CHECK(std::string(toString(DataType::FLOAT32)) == "FLOAT32");
    AGR_CHECK(std::string(toString(DataType::UNKNOWN)) == "UNKNOWN");

    // Empty TensorShape
    TensorShape s0;
    AGR_CHECK(!s0.isValid()); // UNKNOWN dtype
    AGR_CHECK(s0.elementCount() == 0);

    // Valid scalar shape {1}
    TensorShape s1({1}, DataType::FLOAT32);
    AGR_CHECK(s1.isValid());
    AGR_CHECK(s1.rank() == 1);
    AGR_CHECK(s1.elementCount() == 1);
    AGR_CHECK(s1.byteSize() == 4);
    AGR_CHECK(!s1.isEmpty());

    // 2D shape
    TensorShape s2({4, 8}, DataType::FLOAT32);
    AGR_CHECK(s2.elementCount() == 32);
    AGR_CHECK(s2.byteSize()     == 128);

    // Zero-dim is empty
    TensorShape s_zero({0, 8}, DataType::FLOAT32);
    AGR_CHECK(s_zero.isEmpty());
    AGR_CHECK(s_zero.elementCount() == 0);

    // Overflow: enormously large shape should return UINT64_MAX
    // 2^32 * 2^32 dims would overflow; simulate with large values
    TensorShape s_overflow({(size_t)1 << 32, (size_t)1 << 32}, DataType::FLOAT32);
    AGR_CHECK(s_overflow.elementCount() == UINT64_MAX);
    AGR_CHECK(s_overflow.byteSize()     == UINT64_MAX);

    // Equality
    TensorShape sa({2, 3}, DataType::FLOAT32);
    TensorShape sb({2, 3}, DataType::FLOAT32);
    TensorShape sc({2, 4}, DataType::FLOAT32);
    AGR_CHECK(sa == sb);
    AGR_CHECK(sa != sc);

    // describe()
    std::string desc = sa.describe();
    AGR_CHECK(desc.find("FLOAT32") != std::string::npos);
    AGR_CHECK(desc.find("2") != std::string::npos);

    std::cout << "test_p16_tensor_types PASSED\n";
    AGR_TEST_MAIN_END();
}
