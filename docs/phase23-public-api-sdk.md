# Phase 23 — Stable Public API and Developer SDK

## Summary

HeteroAccel Phase 23 provides a production-grade, ABI-stable C public API and a modern C++17 header-only RAII developer SDK. The SDK exposes full inference capabilities, model lifecycle management, streaming token generation, backend introspection, and transparent hardware acceleration (Vulkan GPU compute with automated CPU fallback) for external consumers and applications.

---

## Deliverables & Architecture

### Dual Interface Design

1. **C Public API (`include/heteroaccel/heteroaccel.h`)**:
   - Pure C99 ABI-stable interface with `extern "C"` linkage.
   - Opaque handle types (`HaRuntime`, `HaModel`, `HaSession`).
   - Platform export macros (`HETEROACCEL_API`, `__declspec(dllexport)` on MSVC/Windows, `__attribute__((visibility("default")))` on GCC/Clang).
   - Uniform error code return convention (`HaErrorCode`) with descriptive string lookup via `ha_error_string()`.
   - Streaming token callback interface with cancellation support (`ha_infer_stream` and `ha_infer_cancel`).
   - Hardware backend enumeration and capability introspection (`ha_backend_count`, `ha_backend_get`, `ha_backend_is_available`).

2. **C++17 RAII Wrapper (`include/heteroaccel/HeteroAccel.hpp`)**:
   - Header-only convenience wrapper in the `ha` namespace.
   - Move-only RAII classes (`ha::Runtime`, `ha::Model`, `ha::Session`) guaranteeing leak-free cleanup.
   - Type-safe exception handling (`ha::Error` inheriting from `std::runtime_error`) wrapping `HaErrorCode`.
   - High-level ergonomic convenience methods (e.g. `session.infer("prompt", 32)`).
   - Standard C++ closures for streaming callbacks (`std::function<bool(const std::string& token)>`).

3. **SDK Implementation (`src/sdk/HeteroAccelSDK.cpp`)**:
   - Implements all C API functions, backed by internal runtime components (`HardwareDetector`, `HydroXLModel`, `HydroXLVulkanEngineAdapter`, and `HydroXLCpuEngineAdapter`).
   - Manages handle lifecycles safely with atomic cancellation flags (`std::atomic<bool>`).
   - Performs automatic hardware dispatch: initializes Vulkan GPU acceleration when available and memory constraints permit; falls back seamlessly to multi-threaded CPU inference.

4. **CMake Integration & Packaging**:
   - `agr_sdk`: Static library target (`lib/agr_sdk.lib` or `lib/libagr_sdk.a`).
   - `heteroaccel`: Shared library target (`bin/heteroaccel.dll` + `lib/heteroaccel.lib` or `lib/libheteroaccel.so`).
   - Standard install rules exporting headers, libraries, and runtime binaries.

---

## C API Specification

### Handle Types and Structures

```c
typedef struct HaRuntime_T* HaRuntime;
typedef struct HaModel_T*   HaModel;
typedef struct HaSession_T* HaSession;

typedef struct {
    int max_tokens;       /* Maximum new tokens to generate (default: 32) */
    float temperature;    /* Sampling temperature, 0.0f = greedy */
    float top_p;          /* Nucleus sampling threshold (e.g. 0.95f) */
    int seed;             /* RNG seed (-1 = random device) */
} HaInferParams;

typedef struct {
    char text[4096];            /* Generated text output */
    int tokens_generated;       /* Total tokens generated */
    double tokens_per_second;   /* Generation speed */
    double latency_ms;          /* Total wall-clock latency */
    char backend_used[64];      /* Backend executed: "hydroxl_vulkan" or "hydroxl_cpu" */
} HaResult;

typedef struct {
    char name[64];              /* Backend name identifier */
    char device_name[128];      /* Device description (e.g. GPU model) */
    bool is_available;          /* Hardware and dependencies detected */
    float relative_throughput;  /* Relative performance weight */
} HaBackendInfo;
```

### Core Functions

- **Lifecycle & Error Handling**:
  - `HaErrorCode ha_runtime_create(HaRuntime* out_runtime)`
  - `void ha_runtime_destroy(HaRuntime runtime)`
  - `const char* ha_error_string(HaErrorCode code)`
  - `const char* ha_version_string(void)`

- **Model Management**:
  - `HaErrorCode ha_model_load(HaRuntime runtime, const char* path, HaModel* out_model)`
  - `void ha_model_unload(HaModel model)`
  - `int64_t ha_model_get_param_count(HaModel model)`
  - `int32_t ha_model_get_context_length(HaModel model)`
  - `const char* ha_model_get_architecture(HaModel model)`
  - `const char* ha_model_get_quant_type(HaModel model)`

- **Session and Inference**:
  - `HaErrorCode ha_session_create(HaModel model, const char* backend_name, HaSession* out_session)`
  - `void ha_session_destroy(HaSession session)`
  - `HaErrorCode ha_infer(HaSession session, const char* prompt, const HaInferParams* params, HaResult* out_result)`
  - `HaErrorCode ha_infer_stream(HaSession session, const char* prompt, const HaInferParams* params, ha_stream_callback_t cb, void* user_data, HaResult* out_result)`
  - `HaErrorCode ha_infer_cancel(HaSession session)`

- **Backend Introspection**:
  - `int ha_backend_count(void)`
  - `HaErrorCode ha_backend_get(int index, HaBackendInfo* out_info)`
  - `bool ha_backend_is_available(const char* backend_name)`

---

## C++ Modern RAII Wrapper Usage

```cpp
#include <heteroaccel/HeteroAccel.hpp>
#include <iostream>

int main() {
    try {
        // 1. Initialize Runtime
        ha::Runtime runtime;
        std::cout << "HeteroAccel v" << runtime.version_string() << "\n";

        // 2. Query available backends
        for (const auto& b : runtime.backends()) {
            std::cout << "Backend: " << b.name << " (" << b.device_name 
                      << ") Available: " << (b.is_available ? "YES" : "NO") << "\n";
        }

        // 3. Load Model
        ha::Model model(runtime, "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf");
        std::cout << "Model architecture: " << model.architecture()
                  << ", Context: " << model.context_length() << " tokens\n";

        // 4. Create Session (auto-selects Vulkan GPU acceleration if available)
        ha::Session session(model);

        // 5. Synchronous inference
        ha::InferParams params;
        params.max_tokens = 32;
        params.temperature = 0.0f; // greedy

        ha::Result res = session.infer("What is artificial intelligence?", params);
        std::cout << "Generated: " << res.text << "\n";
        std::cout << "Speed: " << res.tokens_per_second << " tok/s on " << res.backend_used << "\n";

        // 6. Streaming inference
        session.infer_stream("Explain GPU computing in one sentence:", [](const std::string& token) {
            std::cout << token << std::flush;
            return true; // continue streaming
        });
    } catch (const ha::Error& e) {
        std::cerr << "Inference error: " << e.what() << " (code: " << static_cast<int>(e.code()) << ")\n";
        return 1;
    }
    return 0;
}
```

---

## C API Consumer Example

```c
#include <heteroaccel/heteroaccel.h>
#include <stdio.h>

int main(void) {
    HaRuntime runtime = NULL;
    HaModel model = NULL;
    HaSession session = NULL;

    if (ha_runtime_create(&runtime) != HA_SUCCESS) {
        fprintf(stderr, "Failed to create runtime\n");
        return 1;
    }

    if (ha_model_load(runtime, "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", &model) != HA_SUCCESS) {
        fprintf(stderr, "Failed to load model\n");
        ha_runtime_destroy(runtime);
        return 1;
    }

    if (ha_session_create(model, "auto", &session) != HA_SUCCESS) {
        fprintf(stderr, "Failed to create session\n");
        ha_model_unload(model);
        ha_runtime_destroy(runtime);
        return 1;
    }

    HaInferParams params = {
        .max_tokens = 16,
        .temperature = 0.0f,
        .top_p = 0.95f,
        .seed = 42
    };

    HaResult result;
    if (ha_infer(session, "The capital of France is", &params, &result) == HA_SUCCESS) {
        printf("Result: %s\n", result.text);
        printf("Speed: %.2f tok/s via %s\n", result.tokens_per_second, result.backend_used);
    }

    ha_session_destroy(session);
    ha_model_unload(model);
    ha_runtime_destroy(runtime);
    return 0;
}
```

---

## Verification & Performance

### Test Suite Execution (110 / 110 Passing)

All 105 existing baseline tests plus the 5 newly implemented Phase 23 test suites pass completely:

| Test Name | Focus Area | Verification Criteria |
|-----------|------------|-----------------------|
| `test_p23_sdk_init` | Lifecycle & Versioning | Runtime create/destroy, major/minor/patch numbers, error string lookups |
| `test_p23_sdk_model` | Model Operations | GGUF loading, architecture/context/quant introspection, unload, invalid path handling |
| `test_p23_sdk_infer` | Inference Execution | Sync inference, token streaming callback, cancellation token, invalid arg handling |
| `test_p23_sdk_backends` | Hardware Discovery | Backend enumeration, capability retrieval, Vulkan & CPU presence check |
| `test_p23_sdk_concurrent` | Multi-Threading | Independent concurrent sessions across multiple threads without data races |

### Hardware Benchmark (Real Model: Qwen2.5-0.5B-Instruct-Q4_K_M)

Measured on Intel Core i5-1235U with integrated Intel Iris Xe GPU:

| Backend | Mode | Measured tok/s | Latency / token |
|---------|------|----------------|-----------------|
| `hydroxl_vulkan` | Vulkan GPU compute | ~7.23 tok/s | ~138 ms |
| `hydroxl_cpu` | Multi-threaded CPU fallback | ~2.10 tok/s | ~476 ms |
