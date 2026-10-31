# Phase 24 — Native Streaming and Intelligent Runtime Resource Management

## Summary

HeteroAccel Phase 24 delivers production-ready incremental token streaming, cooperative mid-generation cancellation, context-aware memory estimation, persistent multi-backend performance profiles, standard CMake packaging, and pure-Python ctypes bindings.

---

## Deliverables & Architecture

### 1. Native Incremental Token Streaming

- **Direct Loop Emission**: Replaced previous simulated word-splitting with genuine token-by-token emission inside the transformer autoregressive generation loops (`HydroXLCpuEngineAdapter::run_stream` and `HydroXLVulkanEngineAdapter::run_stream`).
- **Engine Streaming Abstraction**:
  - `IInferenceEngine::StreamCallback`: `std::function<bool(const std::string& token_text, int token_id)>`. Returning `false` immediately stops generation.
  - `virtual bool run_stream(...)`: Supported by both CPU and Vulkan HydroXL engine adapters (`supports_streaming() == true`).
  - `run()` delegates directly to `run_stream(req, nullptr, res)` for unified code reuse.

### 2. Mid-Generation Cooperative Cancellation

- **Callback-Driven Cancellation**: The C callback `HaTokenCallback` returns `0` to continue streaming, or non-zero to immediately abort generation.
- **Asynchronous Flag Cancellation**: `ha_session_cancel(session)` or `ha_infer_cancel(session)` safely sets an atomic cancellation flag checked before and during each token generation step.
- Clean session cleanup on abort: intermediate output text and token IDs are preserved and returned with error code `HA_ERR_CANCELLED`.

### 3. Context-Aware Memory Estimation (`MemoryEstimator`)

- Calculates memory requirements prior to loading and allocation:
  - **Weights**: Exact dequantized `float32` byte size based on transformer layer count, hidden dimension, MLP intermediate size, and vocabulary size.
  - **KV Cache**: Sequence length $\times$ layer count $\times 2 \times$ key-value heads $\times$ head dimension $\times \text{sizeof(float32)}$.
  - **Activations**: Working scratch buffers for layer activations, attention projections, SwiGLU, and logits.
  - **Headroom**: Configurable safety factor (default 10%).
- **System Memory Probe**: Integrated with Win32 `GlobalMemoryStatusEx` to query actual available physical RAM.
- **Unified Memory Awareness**: Correctly accounts for shared system/GPU memory on integrated GPUs (e.g. Intel Iris Xe) where VRAM and RAM share the physical pool.

### 4. Persistent Performance Profiles (`PerfProfile`)

- Stores verified tokens/second throughput and peak RSS per `(backend_id, model_hash, quant_type, n_layers, n_embd)`.
- Updates records with running averages over repeated runs.
- Serializes to an extensible key-value format at `%LOCALAPPDATA%/HeteroAccel/perf_profile.txt` (or platform equivalent).
- Resilient parser ignores corrupted, truncated, or incomplete lines without crashing.
- Integrated into `EngineOrchestrator` to select the fastest historically proven backend.

### 5. Packaging & Distribution

- **CMake Install Target**: Generates `HeteroAccelConfig.cmake`, `HeteroAccelConfigVersion.cmake`, and `HeteroAccelTargets.cmake`.
- **pkg-config**: Generates `heteroaccel.pc` with compiler and linker flags.
- **External Consumer Verification**: Verified via `tests/external_consumer` consuming installed targets via `find_package(HeteroAccel)`.
- **Python Bindings**: Pure Python (`ctypes`) wrapper in `python/heteroaccel.py` exposing full lifecycle, memory estimation, streaming inference, and cancellation.

---

## Public API Reference

### C Interface (`include/heteroaccel/heteroaccel.h`)

```c
// Native token streaming callback
typedef int (*HaTokenCallback)(const char* token_text, int token_id, void* user_data);

// Streaming inference execution
HA_API HaError ha_infer_stream(HaSession session, const HaInferParams* params,
                               HaTokenCallback cb, void* user_data);

// Cooperative cancellation
HA_API HaError ha_session_cancel(HaSession session);
HA_API HaError ha_infer_cancel(HaSession session);

// Memory estimation
typedef struct {
    size_t      weights_bytes;
    size_t      kv_cache_bytes;
    size_t      activations_bytes;
    size_t      overhead_bytes;
    size_t      total_estimated;
    int         unified_memory;
    const char* notes;
} HaMemoryEstimate;

HA_API HaError ha_model_memory_estimate(HaModel model, uint32_t max_seq_len, HaMemoryEstimate* out);

// Persistent profiling control
HA_API HaError ha_runtime_perf_reset(HaRuntime rt);
HA_API HaError ha_runtime_perf_path(HaRuntime rt, char* buf, size_t buf_size);
```

### C++17 RAII SDK (`include/heteroaccel/HeteroAccel.hpp`)

```cpp
ha::Runtime rt;
ha::Model model = rt.load_model("path/to/model.gguf");

// Estimate memory
ha::MemoryEstimate est = model.memory_estimate(2048);
std::cout << "Estimated RAM: " << (est.total_estimated / (1024 * 1024)) << " MB\n";

// Streaming with lambda callback
ha::Session session = model.create_session();
session.infer_stream("Explain GPU execution:", [](const std::string& piece, int token_id) {
    std::cout << piece << std::flush;
    return true; // continue
});
```

---

## Test Verification

All 115 tests pass with 100% pass rate (`ctest --test-dir build -C Release`):

| Test | Suite | Execution Time | Status |
|---|---|---|---|
| `test_p24_perf_profile` | Phase 24 | 0.26 s | **PASSED** |
| `test_p24_memory_est` | Phase 24 | 0.23 s | **PASSED** |
| `test_p24_streaming` | Phase 24 | 8.60 s | **PASSED** |
| `test_p24_install` | Phase 24 | 0.04 s | **PASSED** |
| `test_p24_python_bindings` | Phase 24 | 0.49 s | **PASSED** |
| Tests 1–110 (Phases 1–23) | Full Regression | ~185 s | **ALL PASSED** |
