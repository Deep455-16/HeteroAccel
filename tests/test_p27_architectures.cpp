// tests/test_p27_architectures.cpp
// Phase 27: Universal Model Architecture & Tokenizer Reference Suite for HydroXL.
//
// Validates numerical equivalence of HydroXL forward-pass logits against the
// trusted upstream llama.cpp reference engine across all supported architecture
// families:
//   1. Llama / Llama 3  ("llama")   — NORM RoPE, GQA, SwiGLU
//   2. Mistral          ("mistral", "mistral3") — NORM RoPE, GQA, sliding window, SwiGLU
//   3. Qwen2 / Qwen2.5  ("qwen2")   — NEOX RoPE (1e6 base), QKV biases, SwiGLU, GPT-2 BPE
//   4. Phi-3            ("phi3")    — NEOX RoPE, fused attn_qkv & fused ffn_up, SwiGLU
//   5. Gemma            ("gemma")   — NEOX RoPE, sqrt(n_embd) embed scale, GeGLU, tied LM head
//   6. Gemma 2          ("gemma2")  — NEOX RoPE, sqrt(n_embd) embed scale, GeGLU,
//                                     post-attention/post-FFN norms, attn & final logit softcapping

#include "mini_test.h"
#include "gguf_fixture.h"

#include "llama.h"

#include "hydroxl/GGUFLoader.h"
#include "hydroxl/HydroXLForwardPass.h"
#include "hydroxl/HydroXLKVCache.h"
#include "hydroxl/HydroXLModel.h"
#include "hydroxl/HydroXLTokenizer.h"
#include "orchestration/HydroXLEngineAdapter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace agr::hydroxl;

namespace {

struct LogitComparison {
    double cosine_sim = 0.0;
    double max_abs_diff = 0.0;
    int    top1_hydroxl = -1;
    int    top1_llama   = -1;
};

// Run multi-token prompt through llama.cpp CPU reference and return final logits
bool runLlamaCppReference(const std::string& gguf_path,
                          const std::vector<int32_t>& prompt_tokens,
                          std::vector<float>& out_logits) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    ggml_backend_dev_t cpu_only_devs[] = { nullptr };
    mparams.devices = cpu_only_devs;

    llama_model* model = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model) return false;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        llama_model_free(model);
        return false;
    }

    std::vector<llama_token> toks(prompt_tokens.begin(), prompt_tokens.end());
    llama_batch batch = llama_batch_get_one(toks.data(), static_cast<int32_t>(toks.size()));
    if (llama_decode(ctx, batch) != 0) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float* logits_ptr = llama_get_logits_ith(ctx, -1);
    if (!logits_ptr || n_vocab <= 0) {
        llama_free(ctx);
        llama_model_free(model);
        return false;
    }

    out_logits.assign(logits_ptr, logits_ptr + n_vocab);
    llama_free(ctx);
    llama_model_free(model);
    return true;
}

// Run multi-token prompt through HydroXL native forward pass and return final logits
bool runHydroXLForward(const std::string& gguf_path,
                       const std::vector<int32_t>& prompt_tokens,
                       std::vector<float>& out_logits) {
    HydroXLModel model;
    if (!model.load(gguf_path)) {
        std::cerr << "  [HydroXL] load failed: " << model.lastError() << "\n";
        return false;
    }

    const auto& hp = model.hparams();
    HydroXLKVCache kv;
    if (!kv.allocate(hp.n_layers, hp.n_kv_heads, 128, hp.n_head_dim)) {
        return false;
    }

    HydroXLForwardPass fp;
    fp.attach(&model, &kv);

    for (size_t pos = 0; pos < prompt_tokens.size(); ++pos) {
        if (!fp.forward(static_cast<uint32_t>(prompt_tokens[pos]),
                        static_cast<uint32_t>(pos), out_logits)) {
            std::cerr << "  [HydroXL] forward failed: " << fp.lastError() << "\n";
            return false;
        }
    }
    return true;
}

LogitComparison compareLogits(const std::vector<float>& hx,
                              const std::vector<float>& ref) {
    LogitComparison res;
    if (hx.size() != ref.size() || hx.empty()) return res;

    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < hx.size(); ++i) {
        double a = hx[i];
        double b = ref[i];
        dot += a * b;
        na  += a * a;
        nb  += b * b;
        double d = std::fabs(a - b);
        if (d > res.max_abs_diff) res.max_abs_diff = d;
    }
    res.cosine_sim = (na > 0.0 && nb > 0.0) ? (dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0;
    res.top1_hydroxl = static_cast<int>(std::max_element(hx.begin(), hx.end()) - hx.begin());
    res.top1_llama   = static_cast<int>(std::max_element(ref.begin(), ref.end()) - ref.begin());
    return res;
}

void verifyArchitectureAgainstReference(const std::string& label,
                                        const fixture::TinyArchSpec& spec,
                                        const std::string& llama_ref_arch = "") {
    std::string hx_path  = "test_p27_hx_"  + label + ".gguf";
    std::string ref_path = "test_p27_ref_" + label + ".gguf";

    fixture::writeTinyArchModel(hx_path, spec);

    fixture::TinyArchSpec ref_spec = spec;
    if (!llama_ref_arch.empty()) {
        ref_spec.arch = llama_ref_arch;
    }
    fixture::writeTinyArchModel(ref_path, ref_spec);

    // Multi-token prompt (positions 0, 1, 2, 3) exercises RoPE, GQA, KV cache, norms, FFN
    std::vector<int32_t> prompt = {1, 260, 261, 262};
    std::vector<float> hx_logits, ref_logits;

    bool hx_ok  = runHydroXLForward(hx_path, prompt, hx_logits);
    bool ref_ok = runLlamaCppReference(ref_path, prompt, ref_logits);

    AGR_CHECK(hx_ok);
    AGR_CHECK(ref_ok);

    if (hx_ok && ref_ok) {
        LogitComparison cmp = compareLogits(hx_logits, ref_logits);
        std::cout << "  [" << label << "] cosine=" << cmp.cosine_sim
                  << " max_abs_diff=" << cmp.max_abs_diff
                  << " top1_hx=" << cmp.top1_hydroxl
                  << " top1_llama=" << cmp.top1_llama << "\n";
        AGR_CHECK(cmp.cosine_sim > 0.9999);
        // llama.cpp stores KV cache in FP16 by default, introducing ~1e-3 rounding on gemma2
        AGR_CHECK(cmp.max_abs_diff < 2e-3);
        AGR_CHECK(cmp.top1_hydroxl == cmp.top1_llama);
    }

    // Also verify end-to-end generation through HydroXLCpuEngineAdapter
    {
        agr::orchestration::HydroXLCpuEngineAdapter adapter;
        std::string err;
        AGR_CHECK(adapter.initialize(hx_path, err));
        agr::orchestration::InferenceRequest req;
        req.prompt_token_ids = {1, 260, 261};
        req.max_new_tokens   = 4;
        req.temperature      = 0.0f;
        auto inf_res = adapter.run(req);
        AGR_CHECK(inf_res.success);
        AGR_CHECK(inf_res.output_tokens > 0);
        adapter.shutdown();
    }

    std::remove(hx_path.c_str());
    std::remove(ref_path.c_str());
}

} // namespace

int main() {
    std::cout << "=== Phase 27: Universal Architecture & Tokenizer Validation ===\n";
    llama_backend_init();

    // 1. Llama / Llama 3 (NORM RoPE, GQA, SwiGLU)
    {
        fixture::TinyArchSpec sp;
        sp.arch      = "llama";
        sp.tok_model = "llama";
        sp.seed      = 2710;
        verifyArchitectureAgainstReference("llama", sp);
    }

    // 2a. Mistral ("mistral3" native in both HydroXL and llama.cpp)
    {
        fixture::TinyArchSpec sp;
        sp.arch        = "mistral3";
        sp.tok_model   = "llama";
        sp.sliding_win = 64;
        sp.seed        = 2720;
        verifyArchitectureAgainstReference("mistral3", sp);
    }

    // 2b. Mistral ("mistral" identifier in HydroXL vs identical Llama/Mistral graph in llama.cpp)
    {
        fixture::TinyArchSpec sp;
        sp.arch        = "mistral";
        sp.tok_model   = "llama";
        sp.sliding_win = 64;
        sp.seed        = 2725;
        verifyArchitectureAgainstReference("mistral", sp, "llama");
    }

    // 3. Qwen2 / Qwen2.5 (NEOX RoPE, freq_base=1e6, QKV biases, SwiGLU, GPT-2 BPE)
    {
        fixture::TinyArchSpec sp;
        sp.arch      = "qwen2";
        sp.tok_model = "gpt2";
        sp.rms_eps   = 1e-6f;
        sp.rope_base = 1000000.0f;
        sp.qkv_bias  = true;
        sp.seed      = 2730;
        verifyArchitectureAgainstReference("qwen2", sp);
    }

    // 4. Phi-3 (NEOX RoPE, fused attn_qkv.weight AND fused ffn_up.weight [n_embd, 2*n_ff], SwiGLU)
    {
        fixture::TinyArchSpec sp;
        sp.arch         = "phi3";
        sp.tok_model    = "llama";
        sp.rms_eps      = 1e-5f;
        sp.rope_base    = 10000.0f;
        sp.fused_qkv    = true;
        sp.fused_ffn_up = true;
        sp.seed         = 2740;
        verifyArchitectureAgainstReference("phi3", sp);
    }

    // 5. Gemma (NEOX RoPE, sqrt(n_embd) embedding scale, GeGLU, tied LM head)
    {
        fixture::TinyArchSpec sp;
        sp.arch         = "gemma";
        sp.tok_model    = "llama";
        sp.rms_eps      = 1e-6f;
        sp.rope_base    = 10000.0f;
        sp.tied_lm_head = true;
        sp.seed         = 2750;
        verifyArchitectureAgainstReference("gemma", sp);
    }

    // 6. Gemma 2 (NEOX RoPE, sqrt(n_embd) embedding scale, GeGLU, post_attention_norm,
    //             post_ffw_norm, attn_logit_softcapping=50, final_logit_softcapping=30)
    {
        fixture::TinyArchSpec sp;
        sp.arch          = "gemma2";
        sp.tok_model     = "llama";
        sp.rms_eps       = 1e-6f;
        sp.rope_base     = 10000.0f;
        sp.tied_lm_head  = true;
        sp.post_norms    = true;
        sp.attn_softcap  = 50.0f;
        sp.final_softcap = 30.0f;
        sp.sliding_win   = 64;
        sp.seed          = 2760;
        verifyArchitectureAgainstReference("gemma2", sp);
    }

    // 7. SentencePiece ("llama") Tokenizer encode/decode & byte-fallback verification
    {
        const char* spm_path = "test_p27_spm_tok.gguf";
        fixture::TinyArchSpec sp;
        sp.arch      = "llama";
        sp.tok_model = "llama";
        fixture::writeTinyArchModel(spm_path, sp);

        GGUFLoader loader;
        AGR_CHECK(loader.open(spm_path));
        HydroXLTokenizer tok;
        AGR_CHECK(tok.loadFromGGUF(loader));
        AGR_CHECK(tok.tokenizerType() == TokenizerType::LLAMA);
        AGR_CHECK(tok.addsBos());
        AGR_CHECK(tok.addsSpacePrefix());

        // "hello world" -> BOS(1), "▁hello", "▁world"
        auto ids = tok.encode("hello world", true);
        AGR_CHECK(ids.size() == 3);
        AGR_CHECK(ids[0] == 1); // BOS
        AGR_CHECK(tok.decode(ids, true) == "hello world");

        // Byte fallback for uppercase / punctuation not in tiny lowercase vocab
        auto fb_ids = tok.encode("A!", false);
        AGR_CHECK(!fb_ids.empty());
        AGR_CHECK(tok.decode(fb_ids, true) == "A!");

        std::remove(spm_path);
    }

    // 8. Unsupported architecture rejected cleanly
    {
        const char* bad_arch_path = "test_p27_unsupported_arch.gguf";
        fixture::TinyArchSpec sp;
        sp.arch = "mamba_ssm_unsupported";
        fixture::writeTinyArchModel(bad_arch_path, sp);

        HydroXLModel m;
        AGR_CHECK(!m.load(bad_arch_path));
        AGR_CHECK(m.lastError().find("Unsupported architecture") != std::string::npos);

        std::remove(bad_arch_path);
    }

    llama_backend_free();
    AGR_TEST_MAIN_END()
}
