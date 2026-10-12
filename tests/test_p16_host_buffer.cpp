// tests/test_p16_host_buffer.cpp
// Phase 16: HostBuffer allocation, bounds-checking, move semantics, fill, read/write.

#include "primitives/HostBuffer.h"
#include "mini_test.h"
#include <iostream>
#include <cstring>

using namespace agr::primitives;

int main() {
    // Default construction — empty/invalid
    {
        HostBuffer b;
        AGR_CHECK(b.capacity() == 0);
        AGR_CHECK(b.data() == nullptr);
    }

    // Basic allocation
    {
        HostBuffer b(256);
        AGR_CHECK(b.capacity() == 256);
        AGR_CHECK(b.isValid());
        AGR_CHECK(b.data() != nullptr);
    }

    // Zero-size is valid
    {
        HostBuffer b(0);
        AGR_CHECK(b.capacity() == 0);
        AGR_CHECK(b.isValid()); // zero-size is valid
    }

    // fill + read back
    {
        HostBuffer b(16);
        b.fill(0xAB);
        uint8_t val = 0;
        b.read(0, &val, 1);
        AGR_CHECK(val == 0xAB);
        b.read(15, &val, 1);
        AGR_CHECK(val == 0xAB);
    }

    // write + read roundtrip
    {
        HostBuffer b(64);
        float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        AGR_CHECK(b.write(0, src, sizeof(src)));
        float dst[4] = {};
        AGR_CHECK(b.read(0, dst, sizeof(dst)));
        AGR_CHECK(dst[0] == 1.0f && dst[1] == 2.0f && dst[2] == 3.0f && dst[3] == 4.0f);
    }

    // Bounds check: out-of-bounds read/write returns false
    {
        HostBuffer b(8);
        float dummy[4] = {};
        AGR_CHECK(!b.write(6, dummy, sizeof(float) * 2)); // 6+8 > 8
        AGR_CHECK(!b.read(8, dummy, 1));                   // offset == capacity
    }

    // data() bounds check returns nullptr
    {
        HostBuffer b(8);
        AGR_CHECK(b.data(4, 5) == nullptr); // 4+5 > 8
        AGR_CHECK(b.data(0, 8) != nullptr); // exactly capacity
        AGR_CHECK(b.data(8, 0) == nullptr); // offset at boundary, size 0 but offset > cap
    }

    // Move construction
    {
        HostBuffer a(128);
        void* original_ptr = a.data();
        HostBuffer b(std::move(a));
        AGR_CHECK(a.capacity() == 0);
        AGR_CHECK(a.data() == nullptr);
        AGR_CHECK(b.capacity() == 128);
        AGR_CHECK(b.data() == original_ptr);
    }

    // Move assignment
    {
        HostBuffer a(64);
        HostBuffer b(32);
        b = std::move(a);
        AGR_CHECK(b.capacity() == 64);
        AGR_CHECK(a.capacity() == 0);
    }

    // Self-move-assignment (should not crash)
    {
        HostBuffer a(64);
        a = std::move(a);
        // After self-move, state is unspecified but must not crash.
    }

    std::cout << "test_p16_host_buffer PASSED\n";
    AGR_TEST_MAIN_END();
}
