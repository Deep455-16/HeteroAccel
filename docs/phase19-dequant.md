# HeteroAccel — Phase 19: Quantization, GGUF Parsing & Tokenizer

## Quantization Support

Phase 19 extends the HydroXL dequantization layer (`src/hydroxl/HydroXLDequant.cpp`) to cover all GGUF K-quant and legacy formats required for real Qwen2.5 model files.

### Supported Formats

| Format | GGML type id | Block size (elems) | Block bytes | Notes |
|--------|-------------|-------------------|------------|-------|
| F32    | 0           | —                 | 4×n        | No quantization |
| F16    | 1           | —                 | 2×n        | IEEE 754 half-precision |
| Q4_0   | 2           | 32                | 18         | Legacy; parsed but rarely used |
| Q4_1   | 3           | 32                | 20         | Legacy |
| Q5_0   | 6           | 32                | 22         | 5-bit, zero-point |
| Q5_1   | 7           | 32                | 24         | 5-bit, additive minimum |
| Q8_0   | 8           | 32                | 34         | 8-bit, FP16 scale |
| Q2_K   | 10          | 256               | 84         | Parsed; not dequantized by HydroXL |
| **Q3_K** | **11**    | **256**           | **110**    | ✓ Full dequant (Phase 17+19) |
| **Q4_K** | **12**    | **256**           | **144**    | ✓ Full dequant — primary format |
| Q5_K   | 13          | 256               | 176        | Parsed; not dequantized |
| **Q6_K** | **14**    | **256**           | **210**    | ✓ Full dequant (Phase 19) |

> **Note:** The `_S`, `_M`, `_L` suffixes (Q3_K_M, Q4_K_M, etc.) are quantization preset names. All variants of the same K-quant family share the same binary block format and GGML type integer. HydroXL decodes all Q4_K_M and Q4_K_S tensors identically because they have the same on-disk layout.

### Reference Testing

Dequantization correctness is verified by `tests/test_p19_dequant_ref.cpp`, which:

1. Generates deterministic pseudo-random input floats (`std::mt19937`).
2. Quantizes them with the **real GGML** `quantize_row_*_ref()` functions from the bundled llama.cpp dependency.
3. Decodes with the **real GGML** `dequantize_row_*()` (reference) and HydroXL decoders.
4. Requires `max |hydroxl − reference| ≤ 1e-6 × max(1, max|reference|)`.

This is a genuine independent reference comparison — HydroXL decoders are not self-certified.

## GGUF Parsing

`src/hydroxl/GGUFLoader.cpp` is a self-contained GGUF v2/v3 binary parser with no llama.cpp dependency for metadata reading. It handles:

- All scalar metadata types (UINT8–FLOAT64, STRING, BOOL)
- String and typed arrays (stored as `std::vector<std::string>` or skipped for unsupported element types)
- Tensor name, n_dims, shape (uint64 per dim), type, and per-data-section offset
- Data section alignment padding (`general.alignment`, default 32)
- Byte-size computation for all supported GGML types
- EOF bounds checking on tensor data reads
- Rejection of: bad magic, unsupported version, unknown scalar type, unknown array element type, tensor offset outside file

## Tokenizer

`src/hydroxl/HydroXLTokenizer.cpp` implements **GPT-2 byte-level BPE** as used by Qwen2.5.

### Algorithm

1. **Pre-tokenization**: Qwen2 split rules — letter/digit/punctuation runs, contractions, space-prefixed words, digit groups of up to three.
2. **Byte→Unicode mapping**: 256 bytes mapped to unique Unicode codepoints (the GPT-2 table).
3. **BPE merges**: Lowest-rank adjacent pair repeatedly merged until no merge applies.
4. **Vocabulary lookup**: Merged symbols are looked up in the token→ID map.

### Special tokens

- `eos_token_id` and `bos_token_id` are read from GGUF metadata.
- Qwen2 models set `bos_token_id = 0xFFFFFFFF` (no BOS) — `addsBos()` returns `false`.
- Control tokens (type 3) are excluded from `decode()` when `skip_special = true`.

### Known Limitation

The pre-tokenizer classifies every non-ASCII codepoint ≥ U+0080 as a letter. This matches reference behaviour for ASCII, Latin script, and CJK text but may differ for non-ASCII punctuation or emoji adjacent to letters.

### Supported tokenizer family

Only `tokenizer.ggml.model == "gpt2"` (byte-level BPE) is supported. Files reporting `"llama"`, `"rwkv"`, or other families are rejected with a clear error message.

## Verification

Run with the bundled GGML library:

```powershell
# From repository root
$env:HETEROACCEL_MODEL_PATH = "D:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
ctest --test-dir build -C Release -R "test_p17|test_p19" --output-on-failure
```
