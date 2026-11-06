# Supported Model Architectures & Quantization Matrix (Phase 27)

This document defines the exact, test-validated model architectures and GGML quantization formats supported by the native **HydroXL** inference engine and the **HeteroAccel** orchestration layer.

Every architecture and quantization entry listed as **Supported** below is validated against the bundled upstream `llama.cpp` / `ggml` reference implementation in automated CTest suites (`test_p19_dequant_ref`, `test_p27_quant_decoders`, `test_p27_architectures`).

---

## 1. Supported Model Architectures (HydroXL Native Engine)

| Architecture Family | GGUF `general.architecture` | Normalization | RoPE Variant & Default `freq_base` | Attention / GQA & Softcapping | FFN Activation & Tensor Layout | Tokenizer Family (`tokenizer.ggml.model`) | Reference Validation Test |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama / Llama 2 / Llama 3** | `llama` | Pre-attn & pre-FFN RMSNorm ($\epsilon = 10^{-5}$) | `NORM` (interleaved consecutive pairs), `10000.0` (`500000.0` in Llama 3 metadata) | MHA / GQA (`head_count_kv`) | `SwiGLU` (`ffn_gate`, `ffn_up`, `ffn_down`) | `llama` (SentencePiece BPE) or `gpt2` (Byte-level BPE) | `test_p27_architectures` (`[llama]`) |
| **Mistral / Ministral** | `mistral`, `mistral3` | Pre-attn & pre-FFN RMSNorm ($\epsilon = 10^{-5}$) | `NORM` (interleaved consecutive pairs), `10000.0` | GQA + optional sliding window (`attention.sliding_window`) | `SwiGLU` (`ffn_gate`, `ffn_up`, `ffn_down`) | `llama` (SentencePiece BPE) or `gpt2` | `test_p27_architectures` (`[mistral3]`, `[mistral]`) |
| **Qwen2 / Qwen2.5** | `qwen2`, `qwen2_vl` | Pre-attn & pre-FFN RMSNorm ($\epsilon = 10^{-6}$) | `NEOX` (half-split pairs), `1000000.0` | GQA + QKV biases (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) | `SwiGLU` (`ffn_gate`, `ffn_up`, `ffn_down`) | `gpt2` (Byte-level BPE) | `test_p27_architectures` (`[qwen2]`) |
| **Phi-3** | `phi3`, `phi2` | Pre-attn & pre-FFN RMSNorm ($\epsilon = 10^{-5}$) | `NEOX` (half-split pairs), `10000.0`, partial `rope.dimension_count` supported | MHA / GQA, supports both fused `blk.{l}.attn_qkv.weight` and split `attn_q/k/v` | `SwiGLU`, supports both fused `blk.{l}.ffn_up.weight` (`[n_embd, 2*n_ff]`) and split `ffn_gate` + `ffn_up` | `llama` (SentencePiece BPE) | `test_p27_architectures` (`[phi3]`) |
| **Gemma** | `gemma` | Pre-attn & pre-FFN RMSNorm ($\epsilon = 10^{-6}$), embedding scaled by $\sqrt{n_{\text{embd}}}$ | `NEOX` (half-split pairs), `10000.0` | MHA / GQA, tied LM head (`token_embd.weight`) | `GeGLU` ($\text{GELU}(\text{gate}) \odot \text{up}$) | `llama` (`add_space_prefix = false`) | `test_p27_architectures` (`[gemma]`) |
| **Gemma 2** | `gemma2` | Pre-attn, **post-attn** (`post_attention_norm`), pre-FFN, **post-FFN** (`post_ffw_norm`) RMSNorm ($\epsilon = 10^{-6}$), embedding scaled by $\sqrt{n_{\text{embd}}}$ | `NEOX` (half-split pairs), `10000.0` | GQA + sliding window + attention logit softcapping (`attn_logit_softcapping`) + final logit softcapping (`final_logit_softcapping`), tied LM head | `GeGLU` ($\text{GELU}(\text{gate}) \odot \text{up}$) | `llama` (`add_space_prefix = false`) | `test_p27_architectures` (`[gemma2]`) |

---

## 2. Supported GGML Quantization Types (HydroXL Dequantization Engine)

All 13 prioritized GGML tensor types are implemented in [`src/hydroxl/HydroXLDequant.cpp`](../src/hydroxl/HydroXLDequant.cpp) with strict block-size divisibility validation, disk byte-bounds checking, overflow protection, and bit-exact / $10^{-6}$ relative tolerance equivalence against upstream `ggml-quants.c`.

| GGML Type Name | GGML Type ID | Elements per Block (`block_size`) | Bytes per Block (`block_bytes`) | Formula / Unpacking Summary | HydroXL CPU Support | HydroXL Vulkan Support | Reference Validation Test |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| **`F32`** | `0` | `1` | `4` | IEEE 754 FP32 direct copy | Supported | Supported | `test_p27_quant_decoders` |
| **`F16`** | `1` | `1` | `2` | Exact IEEE 754 FP16 $\to$ FP32 conversion | Supported | Supported | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q4_0`** | `2` | `32` | `18` | `fp16 d` (2B) + 16B nibbles: $x = (q - 8) \cdot d$ | Supported | Supported (dequant on load) | `test_p27_quant_decoders` |
| **`Q4_1`** | `3` | `32` | `20` | `fp16 d` (2B) + `fp16 m` (2B) + 16B nibbles: $x = q \cdot d + m$ | Supported | Supported (dequant on load) | `test_p27_quant_decoders` |
| **`Q5_0`** | `6` | `32` | `22` | `fp16 d` (2B) + `uint32 qh` (4B) + 16B nibbles: $x = ((q_4 \mid q_{h}) - 16) \cdot d$ | Supported | Supported (dequant on load) | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q5_1`** | `7` | `32` | `24` | `fp16 d` (2B) + `fp16 m` (2B) + `uint32 qh` (4B) + 16B nibbles: $x = (q_4 \mid q_h) \cdot d + m$ | Supported | Supported | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q8_0`** | `8` | `32` | `34` | `fp16 d` (2B) + 32 $\times$ `int8 qs`: $x = d \cdot q$ | Supported | Supported | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q2_K`** | `10` | `256` | `84` | 16B 4-bit scales/mins + 64B 2-bit quants + `fp16 d, dmin` | Supported | Supported (dequant on load) | `test_p27_quant_decoders` |
| **`Q3_K`** | `11` | `256` | `110` | 32B `hmask` + 64B 2-bit `qs` + 12B packed 6-bit scales + `fp16 d` | Supported | Supported (dequant on load) | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q4_K`** | `12` | `256` | `144` | `fp16 d, dmin` + 12B packed 6-bit scales/mins + 128B 4-bit `qs` | Supported | Supported (dequant on load; auto-orchestrated to CPU for large K-quant models) | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q5_K`** | `13` | `256` | `176` | `fp16 d, dmin` + 12B packed 6-bit scales/mins + 32B `qh` + 128B `ql` | Supported | Supported (dequant on load) | `test_p27_quant_decoders` |
| **`Q6_K`** | `14` | `256` | `210` | 128B `ql` + 64B `qh` + 16B `int8 scales` + `fp16 d` | Supported | Supported (dequant on load) | `test_p19_dequant_ref`, `test_p27_quant_decoders` |
| **`Q8_K`** | `15` | `256` | `292` | `fp32 d` (4B) + 256B `int8 qs` + 32B `int16 bsums` | Supported | Supported (dequant on load) | `test_p27_quant_decoders` |

> **Note on GGUF File Presets (`Q4_K_M`, `Q4_K_S`, `Q5_K_M`, `Q3_K_M`, `Q6_K`, etc.):**
> GGUF model filenames such as `*-Q4_K_M.gguf` or `*-Q5_K_M.gguf` are mixed-quantization presets that assign per-tensor GGML types (for example, `Q4_K` for most projection matrices and `Q6_K` or `Q8_0` for `attn_v.weight`, `ffn_down.weight`, and `output.weight`). Because `HydroXLModel::loadTensor` inspects and decodes every tensor by its individual `GGMLType` descriptor, all standard K-quant preset combinations (`Q2_K`, `Q3_K_S/M/L`, `Q4_K_S/M`, `Q5_K_S/M`, `Q6_K`, `Q8_0`) are supported out of the box.
