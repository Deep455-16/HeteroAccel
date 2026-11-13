// tests/test_p29_real_model_equiv.cpp
// Phase 29: real-model numerical equivalence of HydroXL (threaded matvec) vs the
// llama.cpp CPU reference on Qwen2.5-0.5B Q4_K_M. Needed because the benchmark
// showed greedy generations diverge after a few tokens; this test establishes
// whether that is genuine near-tie noise (expected: llama.cpp multiplies Q4_K
// weights with int8-quantised activations, HydroXL uses f32 activations) or a bug.
#include "mini_test.h"
#include "llama.h"

#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLKVCache.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace agr::hydroxl;

static bool llamaLogits(const std::string& path, const std::vector<int32_t>& toks_in, std::vector<float>& out) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    ggml_backend_dev_t none[] = {nullptr};
    mp.devices = none;
    llama_model* m = llama_model_load_from_file(path.c_str(), mp);
    if (!m) return false;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 256; cp.n_threads = 4; cp.n_threads_batch = 4;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;   // remove KV-cache rounding from the comparison
    llama_context* c = llama_init_from_model(m, cp);
    if (!c) { llama_model_free(m); return false; }
    std::vector<llama_token> t(toks_in.begin(), toks_in.end());
    bool ok = llama_decode(c, llama_batch_get_one(t.data(), (int32_t)t.size())) == 0;
    if (ok) {
        int n = llama_vocab_n_tokens(llama_model_get_vocab(m));
        const float* l = llama_get_logits_ith(c, -1);
        ok = l != nullptr;
        if (ok) out.assign(l, l + n);
    }
    llama_free(c);
    llama_model_free(m);
    return ok;
}

int main() {
    std::cout << "== test_p29_real_model_equiv ==\n";
    const char* mp = std::getenv("HETEROACCEL_MODEL_PATH");
    std::ifstream probe(mp ? mp : "", std::ios::binary);
    if (!mp || !probe.good()) {
        std::cout << "[SKIP] HETEROACCEL_MODEL_PATH not set/readable\n";
        AGR_TEST_MAIN_END();
    }
    probe.close();
    std::string path = mp;

    HydroXLModel model;
    AGR_CHECK(model.load(path));
    HydroXLTokenizer tok;
    {
        GGUFLoader g;
        AGR_CHECK(g.open(path));
        AGR_CHECK(tok.loadFromGGUF(g));
    }
    const auto& hp = model.hparams();
    HydroXLKVCache kv;
    AGR_CHECK(kv.allocate(hp.n_layers, hp.n_kv_heads, 256, hp.n_head_dim));
    HydroXLForwardPass fp;
    fp.attach(&model, &kv);

    std::vector<int32_t> ids;
    for (auto t : tok.encode("The capital of France is", tok.addsBos())) ids.push_back((int32_t)t);
    AGR_CHECK(ids.size() >= 4);

    // Teacher-forced prefixes: the prompt, then the shared continuation " Paris. It".
    std::vector<int32_t> cont = {12095, 13, 1084};
    std::vector<int32_t> seq = ids;
    for (int c : cont) seq.push_back(c);

    std::vector<float> hx;
    size_t checked = 0;
    for (size_t pos = 0; pos < seq.size(); ++pos) {
        AGR_CHECK(fp.forward((uint32_t)seq[pos], (uint32_t)pos, hx));
        if (pos + 1 < ids.size()) continue;   // compare at every position from the end of the prompt on

        std::vector<int32_t> prefix(seq.begin(), seq.begin() + pos + 1);
        std::vector<float> ref;
        AGR_CHECK(llamaLogits(path, prefix, ref));
        AGR_CHECK(ref.size() == hx.size());
        if (ref.size() != hx.size()) break;

        double dot = 0, na = 0, nb = 0, maxd = 0;
        for (size_t i = 0; i < hx.size(); ++i) {
            dot += (double)hx[i] * ref[i]; na += (double)hx[i] * hx[i]; nb += (double)ref[i] * ref[i];
            maxd = std::max(maxd, (double)std::fabs(hx[i] - ref[i]));
        }
        double cosv = dot / (std::sqrt(na) * std::sqrt(nb));
        int t_hx  = (int)(std::max_element(hx.begin(), hx.end()) - hx.begin());
        int t_ref = (int)(std::max_element(ref.begin(), ref.end()) - ref.begin());
        // llama.cpp's own margin between its top-1 and the token HydroXL picked.
        double gap = (double)ref[t_ref] - (double)ref[t_hx];
        std::cout << "  pos " << pos << ": cosine=" << cosv << " max|d|=" << maxd
                  << " top1 hx=" << t_hx << " llama=" << t_ref << " llama_gap_to_hx_choice=" << gap << "\n";

        AGR_CHECK(cosv > 0.999);
        // Differing argmax is only acceptable when llama.cpp itself sees the two as a near tie.
        if (t_hx != t_ref) AGR_CHECK(gap < 0.5);
        ++checked;
    }
    AGR_CHECK(checked >= 3);
    AGR_TEST_MAIN_END();
}
