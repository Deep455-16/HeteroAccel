// tests/test_p17_kv_cache.cpp
// Phase 17: Unit tests for the HydroXL KV cache.

#include "mini_test.h"
#include "hydroxl/HydroXLKVCache.h"
#include <cstring>
#include <iostream>

using namespace agr::hydroxl;

int main() {
    // Not allocated initially
    {   HydroXLKVCache c;
        AGR_CHECK(!c.isAllocated());
        AGR_CHECK(c.memoryBytes() == 0);
    }

    // Valid allocation
    {   HydroXLKVCache c;
        AGR_CHECK(c.allocate(2, 2, 16, 8));
        AGR_CHECK(c.isAllocated());
        AGR_CHECK(c.nLayers()   == 2);
        AGR_CHECK(c.nKVHeads()  == 2);
        AGR_CHECK(c.maxSeqLen() == 16);
        AGR_CHECK(c.headDim()   == 8);
        AGR_CHECK(c.currentSeqLen() == 0);
    }

    // Zero-dimension rejection
    {   HydroXLKVCache c;
        AGR_CHECK(!c.allocate(0, 2, 16, 8));
        AGR_CHECK(!c.allocate(2, 0, 16, 8));
        AGR_CHECK(!c.allocate(2, 2, 0,  8));
        AGR_CHECK(!c.allocate(2, 2, 16, 0));
        AGR_CHECK(!c.isAllocated());
    }

    // Memory footprint
    {   HydroXLKVCache c;
        c.allocate(2, 2, 16, 8);
        // 2 * 2 * 16 * 8 * 2 * 4 bytes = 4096
        size_t expected = 2UL * 2 * 16 * 8 * 2 * sizeof(float);
        AGR_CHECK(c.memoryBytes() == expected);
    }

    // Write and read back
    {   HydroXLKVCache c;
        AGR_CHECK(c.allocate(1, 1, 8, 4));
        float k[4] = {1.f,2.f,3.f,4.f}, v[4] = {5.f,6.f,7.f,8.f};
        AGR_CHECK(c.writeKV(0, 0, k, v));
        const float* kr = c.kLayer(0);
        const float* vr = c.vLayer(0);
        AGR_CHECK(kr && vr);
        AGR_CHECK(kr[0]==1.f && kr[1]==2.f && kr[2]==3.f && kr[3]==4.f);
        AGR_CHECK(vr[0]==5.f && vr[1]==6.f && vr[2]==7.f && vr[3]==8.f);
    }

    // Multiple positions
    {   HydroXLKVCache c;
        c.allocate(1, 1, 4, 2);
        float k0[2]={1.f,2.f}, v0[2]={3.f,4.f};
        float k1[2]={5.f,6.f}, v1[2]={7.f,8.f};
        AGR_CHECK(c.writeKV(0, 0, k0, v0));
        AGR_CHECK(c.writeKV(0, 1, k1, v1));
        const float* kr = c.kLayer(0);
        // pos 0 → offset 0, pos 1 → offset 2
        AGR_CHECK(kr[0]==1.f && kr[1]==2.f);
        AGR_CHECK(kr[2]==5.f && kr[3]==6.f);
    }

    // Out-of-bounds write
    {   HydroXLKVCache c;
        c.allocate(1, 1, 4, 2);
        float k[2]={}, v[2]={};
        AGR_CHECK(!c.writeKV(0, 4, k, v)); // pos >= max_seq
        AGR_CHECK(!c.writeKV(1, 0, k, v)); // layer >= n_layers
    }

    // advanceSeq and overflow
    {   HydroXLKVCache c;
        c.allocate(1, 1, 4, 2);
        AGR_CHECK(c.currentSeqLen() == 0);
        AGR_CHECK(c.advanceSeq()); AGR_CHECK(c.advanceSeq());
        AGR_CHECK(c.advanceSeq()); AGR_CHECK(c.advanceSeq());
        AGR_CHECK(c.currentSeqLen() == 4);
        AGR_CHECK(!c.advanceSeq()); // overflow
    }

    // reset clears seq_len and zeros data
    {   HydroXLKVCache c;
        c.allocate(1, 1, 4, 2);
        float k[2]={9.f,9.f}, v[2]={9.f,9.f};
        c.writeKV(0, 0, k, v);
        c.advanceSeq();
        c.reset();
        AGR_CHECK(c.currentSeqLen() == 0);
        const float* kr = c.kLayer(0);
        AGR_CHECK(kr[0]==0.f && kr[1]==0.f);
    }

    // Bad layer index → null
    {   HydroXLKVCache c;
        c.allocate(2, 1, 4, 2);
        AGR_CHECK(c.kLayer(2) == nullptr);
        AGR_CHECK(c.vLayer(2) == nullptr);
    }

    AGR_TEST_MAIN_END()
}
