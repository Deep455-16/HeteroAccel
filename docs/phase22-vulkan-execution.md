# Phase 22 — HydroXL Vulkan Tensor Execution Pipeline

## Summary

Phase 22 replaces the `HydroXLVulkanEngineAdapter` stub with a fully functional Vulkan
compute backend for real HydroXL inference. The implementation is vendor-neutral: it
works on any Vulkan-capable device (Intel, AMD, NVIDIA, or other). CPU inference remains
operational as the reference implementation and automatic fallback.

**Measured on Intel Iris Xe (unified memory, Intel i5-1235U, 16 GB RAM):**

| Backend        | tok/s | Notes                              |
|----------------|-------|------------------------------------|
| CPU (HydroXL)  | 2.07  | Reference baseline                 |
| Vulkan (HydroXL) | 5.20 | ~2.5× speedup, output bit-identical |

Output tokens are **bit-identical** between CPU and Vulkan execution at temperature=0.

---

## Architecture

### Execution Split

| Operation            | Device | Rationale                                        |
|----------------------|--------|--------------------------------------------------|
| Embedding lookup     | CPU    | Table lookup; trivial on host                    |
| QKV projections      | GPU    | Dominant matmul; large weight matrices           |
| RMSNorm              | GPU    | Parallelises naturally across hidden dim         |
| Attention scores     | CPU    | Seq-length–dependent KV cache access             |
| RoPE                 | CPU    | Complex trigonometry; CPU is sufficient          |
| FFN gate/up/down     | GPU    | Largest matmuls in the model                     |
| SwiGLU               | GPU    | Element-wise; zero host–device roundtrip         |
| Residual add         | GPU    | Fused with surrounding ops on device             |
| LM head              | GPU    | Largest output matmul                            |

### Quantization Transparency

HydroXL dequantizes all model weights to float32 on the CPU during `HydroXLModel::load()`.
The Vulkan backend uploads these float32 buffers at `initialize()` time and reuses them
across every forward pass. This means the Vulkan backend supports **all GGUF quantization
formats** (Q4_K_M, Q8_0, F16, F32, etc.) without format-specific GPU shader code.

---

## New Files

### Shaders (`shaders/`)

| File                    | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| `hydroxl_matvec.comp`   | Matrix-vector multiply. One work-group per output row; 256 threads with shared-memory reduction. |
| `hydroxl_rmsnorm.comp`  | RMS normalization. Parallel max+sum reduction then per-element scale+weight broadcast. |
| `hydroxl_elemwise.comp` | Element-wise ops: residual add (op=0), SiLU (op=1), SwiGLU (op=2). Push-constant op selector. |
| `hydroxl_softmax.comp`  | In-place softmax. Two-pass max + sum reductions then divide. |

### Source (`src/hydroxl/`)

| File                    | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| `VulkanTensorOps.h`     | Public API with `#if AGR_HAVE_VULKAN` guard; no-op stub for CPU-only builds. |
| `VulkanTensorOps.cpp`   | 565 lines. `initialize()`, `loadPipeline()`, `runPipeline()`, `uploadTensor()`, `downloadTensor()`, `matvecHost()`, `rmsnorm()`, `elemwise()`, `softmax()`, `shutdown()`. |

### Modified (`src/orchestration/`)

| File                         | Change                                                        |
|------------------------------|---------------------------------------------------------------|
| `HydroXLEngineAdapter.h`     | `HydroXLVulkanEngineAdapter` now owns `VulkanTensorOps vk_ops_`, scratch buffers, and `forward_token()`. |
| `HydroXLEngineAdapter.cpp`   | Full `initialize()` (weight upload) and `run()` (hybrid CPU/GPU forward pass). |
| `OrchestratorInit.cpp`       | Updated `hydroxl_vulkan` capabilities: all quant types, `min_vram_bytes=0` (unified memory), `relative_throughput=2.5`. |

### Modified (`src/gpu/`)

| File              | Change                                                             |
|-------------------|--------------------------------------------------------------------|
| `VulkanBackend.h` | Added `device()`, `computeQueue()`, `commandPool()`, `vkBufferOf()` public accessors for use by `VulkanTensorOps`. |

### Tests (`tests/`)

| File                        | Tests                                                       |
|-----------------------------|-------------------------------------------------------------|
| `test_p22_vulkan_ops.cpp`   | Upload/download round-trip, matvec accuracy (<1e-4), RMSNorm (<1e-5), residual add (<1e-6), SiLU (<1e-5), SwiGLU (<1e-5), softmax (<1e-5). |
| `test_p22_vulkan_engine.cpp`| initialize(), backend_id(), run() success/gpu_used/output_tokens, greedy determinism (two runs identical), CPU vs Vulkan token parity, shutdown(). |

---

## Build

Phase 22 shaders compile to SPIR-V at build time using `glslangValidator` or `glslc`
(whichever is found in the Vulkan SDK). The `agr_shaders` CMake target builds all `.comp`
files in `shaders/`. `agr_hydroxl` depends on `agr_shaders` and links `VulkanTensorOps.cpp`.

```cmake
# CMakeLists.txt (Phase 22 additions)
set(AGR_ALL_SHADERS
    vector_add.comp
    hydroxl_matvec.comp
    hydroxl_rmsnorm.comp
    hydroxl_elemwise.comp
    hydroxl_softmax.comp
)
```

If neither shader compiler is found, the build succeeds with a CMake warning and Vulkan
shaders are not compiled. The engine gracefully falls back to the CPU backend.

---

## Test Results (Phase 22 baseline — Intel Iris Xe)

```
test_p22_vulkan_ops:
  [INFO] Vulkan device: Intel(R) Iris(R) Xe Graphics
  ok: init_ok
  ok: upload/download round-trip
  ok: matvec max diff 1.19e-06  (<1e-4)
  ok: rmsnorm max diff 2.38e-07 (<1e-5)
  ok: residual add max diff 0   (<1e-6)
  ok: silu max diff 4.77e-07    (<1e-5)
  ok: swiglu max diff 9.54e-07  (<1e-5)
  ok: softmax max diff 1.86e-09 (<1e-5)

test_p22_vulkan_engine (Qwen2.5-0.5B-Instruct-Q4_K_M):
  [INFO] Vulkan output 1: 5.20 tok/s
  [INFO] CPU output:      2.07 tok/s
  [INFO] CPU vs Vulkan tokens match: YES
  ok: all assertions pass
```

---

## Limitations and Known Issues

- **Garbled / non-English output** on this model-tokenizer combination persists from Phase 19. The output text is bit-identical between CPU and Vulkan backends, confirming this is a pre-existing tokenizer or model-loading issue, not a Vulkan regression.
- **KV cache is host-resident.** Attention scores and RoPE still run on CPU. A future phase could move the KV cache and attention to GPU for fully device-resident inference.
- **No persistent performance profile.** Benchmark numbers above are from one run; a future phase should accumulate benchmark history and feed it to the orchestrator selection policy.
- **Shader recompilation** is required if the Vulkan SDK is updated or if the SPIR-V target version changes.

---

## Recommended Phase 23 Improvements

1. **Vulkan KV cache and attention** — move `k_buf_`, `v_buf_`, and causal attention to GPU to eliminate per-token host-device transfers.
2. **Persistent performance profile** — log tok/s measurements to a profile file; use them in `EngineOrchestrator` selection.
3. **CUDA engine adapter** — guarded by `#ifdef HETEROACCEL_CUDA_ENABLED`; mirrors `HydroXLVulkanEngineAdapter` using CUDA kernels.
4. **Plugin-to-orchestrator auto-wiring** — automatically register plugin-provided engines into the orchestrator at startup.
5. **Context-aware memory estimation** — estimate peak runtime memory (model weights + KV cache + activations) before selecting a backend, to avoid OOM on low-memory devices.
