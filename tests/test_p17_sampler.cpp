// tests/test_p17_sampler.cpp
// Phase 17: Unit tests for HydroXL logits sampler.

#include "mini_test.h"
#include "hydroxl/HydroXLSampler.h"

#include <iostream>
#include <vector>

using namespace agr::hydroxl;

int main() {
    // Greedy (temperature=0) picks argmax
    {   SamplerParams p; p.temperature = 0.0f; p.seed = 42;
        HydroXLSampler s(p);
        float logits[5] = {1.f,5.f,3.f,2.f,4.f};
        AGR_CHECK(s.sample(logits, 5, {}) == 1);
    }

    // null logits → invalid
    {   SamplerParams p; HydroXLSampler s(p);
        AGR_CHECK(s.sample(nullptr, 10, {}) == TOKEN_INVALID);
    }

    // zero vocab → invalid
    {   SamplerParams p; HydroXLSampler s(p);
        float logits[4] = {1.f,2.f,3.f,4.f};
        AGR_CHECK(s.sample(logits, 0, {}) == TOKEN_INVALID);
    }

    // same seed → same result
    {   SamplerParams p; p.temperature = 0.8f; p.top_k = 10; p.seed = 1337;
        float logits[10]; for (int i=0;i<10;i++) logits[i]=(float)i;
        HydroXLSampler s1(p), s2(p);
        AGR_CHECK(s1.sample(logits, 10, {}) == s2.sample(logits, 10, {}));
    }

    // result always in range [0, vocab_size)
    {   SamplerParams p; p.temperature = 1.0f; p.top_k = 0; p.seed = 42;
        HydroXLSampler s(p);
        float logits[20]; for (int i=0;i<20;i++) logits[i]=(float)(i%5);
        for (int t=0;t<20;t++) { auto tok = s.sample(logits, 20, {}); AGR_CHECK(tok>=0 && tok<20); }
    }

    // top_k=1 → always pick the best token
    {   SamplerParams p; p.temperature = 1.0f; p.top_k = 1; p.seed = 0;
        HydroXLSampler s(p);
        float logits[5] = {0.f,10.f,0.f,0.f,0.f};
        for (int i=0;i<5;i++) AGR_CHECK(s.sample(logits, 5, {}) == 1);
    }

    // repeat penalty: penalised token loses when another is close
    {   SamplerParams p; p.temperature = 0.0f; p.repeat_penalty = 10.0f;
        HydroXLSampler s(p);
        float logits[3] = {1.f, 2.f, 3.f}; // greedy → 2
        AGR_CHECK(s.sample(logits, 3, {}) == 2);
        // penalise token 2: logit 3.0/10 = 0.3 < 2.0 → now picks 1
        AGR_CHECK(s.sample(logits, 3, {2}) == 1);
    }

    // reset reproduces the same sequence
    {   SamplerParams p; p.temperature = 0.9f; p.seed = 777;
        float logits[8]; for (int i=0;i<8;i++) logits[i]=(float)i;
        HydroXLSampler s(p);
        std::vector<TokenID> seq1, seq2;
        for (int i=0;i<5;i++) seq1.push_back(s.sample(logits, 8, {}));
        s.reset(777);
        for (int i=0;i<5;i++) seq2.push_back(s.sample(logits, 8, {}));
        AGR_CHECK(seq1 == seq2);
    }

    AGR_TEST_MAIN_END()
}
