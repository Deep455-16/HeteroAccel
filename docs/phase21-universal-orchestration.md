# Phase 21: Universal Hardware-Aware Engine Orchestration

## Overview

Phase 21 introduces a hardware-independent orchestration layer that dynamically selects and coordinates the best compatible inference backend for each workload. The orchestration layer sits between the user-facing API and the backend engines, evaluating hardware capabilities, model requirements, and observed performance history to make evidence-based backend selection decisions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User / CLI / Tests                  │
└─────────────────────────┬───────────────────────────────┘
                          │ SelectionRequest
                          ▼
┌─────────────────────────────────────────────────────────┐
│              EngineOrchestrator (agr_orchestration)     │
│  • register_backend()     • select_backend()            │
│  • create_engine()        • record_performance()        │
│  • print_diagnostics()                                  │
└──────┬─────────────────────┬───────────────────────────-┘
       │                     │
       ▼                     ▼
┌──────────────┐    ┌─────────────────────┐
│HardwareDetec-│    │BackendCapabilities  │
│tor           │    │Registry             │
│ CPU/Vulkan/  │    │ hydroxl_cpu         │
│ CUDA/RAM     │    │ hydroxl_vulkan      │
└──────────────┘    │ cuda (guarded)      │
                    │ plugin:<name>       │
                    └─────────────────────┘
                          │
              ┌───────────┴────────────┐
              ▼                        ▼
   ┌─────────────────┐      ┌──────────────────────┐
   │HydroXLCpuEngine │      │HydroXLVulkanEngine   │
   │Adapter          │      │Adapter               │
   │(IInferenceEngine│      │(returns fallback     │
   │ impl)           │      │ until shader pipeline│
   └─────────────────┘      │ is complete)         │
                            └──────────────────────┘
```

---

## Components

### `src/orchestration/HardwareDetector.h/.cpp`
Runtime hardware capability registry. Detects:
- **System RAM** via `GlobalMemoryStatusEx` (Windows) / `MemoryDetector`
- **CPU devices** via existing `CPUDetector`
- **Vulkan GPUs** via existing `VulkanDetector` — enumerates all physical devices, extracts VRAM (DEVICE_LOCAL heap), queue families, driver version, and vendor ID
- **DXGI GPU fallback** via `GPUDetector` when Vulkan is unavailable
- **CUDA devices** (guarded by `#ifdef HETEROACCEL_CUDA_ENABLED`) via `CUDADetector`

Vendor detection covers: Intel (0x8086), NVIDIA (0x10DE), AMD (0x1002), ARM (0x13B5), Apple (0x106B), plus string-based fallback.

**Mock environment support** (`set_mock_environment`) enables deterministic unit testing without real hardware dependency.

**Idempotent**: calling `detect()` twice resets and re-enumerates cleanly.

### `src/orchestration/BackendCapabilities.h`
Descriptor for each registered backend. Fields:
- `backend_id` — unique string key (`"hydroxl_cpu"`, `"hydroxl_vulkan"`, `"cuda"`, `"plugin:<name>"`)
- `supported_quant_types` — list of GGML type IDs; empty = wildcard (all types accepted)
- `supported_arch` — model architecture strings; empty = wildcard; matching is case-insensitive
- `requires_cuda` / `requires_vulkan` — hardware gate flags
- `min_vram_bytes` / `min_ram_bytes` — memory gates
- `relative_throughput` — normalized heuristic throughput (1.0 = baseline CPU)
- `init_cost_ms` — estimated cold-start latency
- `priority` — tie-breaking weight

`is_available(hw)` evaluates all gates against the live `HardwareDetector`.

### `src/orchestration/EngineOrchestrator.h/.cpp`
Core orchestration logic. Thread-safe via `std::mutex`.

**Backend registration** (`register_backend`): idempotent — re-registers replace existing entries by ID.

**Engine factory registration** (`register_engine_factory`): associates a backend ID with a `std::function<shared_ptr<IInferenceEngine>()>` factory.

**Backend selection algorithm** (`select_backend`):
1. If `forced_backend` is set: verify it is available + compatible; fall back to CPU if not and `allow_cpu_fallback`
2. Filter backends: `is_available(hw)` && `supports_quant(type)` && `supports_arch(arch)` && sufficient RAM/VRAM (model_size × 1.2 headroom)
3. Score each candidate:
   ```
   throughput = observed_tok_s  (if perf_history exists)  else  relative_throughput
   init_cost_factor = 1.0 + (init_cost_ms / 1000)
   if prefer_latency: init_cost_factor *= 2.0
   ram_headroom_factor = min(2.0, max(1.0, avail_ram / needed / 2))
   score = (throughput × ram_headroom_factor) / init_cost_factor + priority × 0.001
   ```
4. Select highest score; populate `reason` and `rejected_backends`
5. Final CPU fallback if no candidate passed and `allow_cpu_fallback`

**Engine creation** (`create_engine`): instantiates engine via factory, calls `initialize(model_path)`. On failure, retries with `hydroxl_cpu` factory if `allow_cpu_fallback`.

**Performance recording** (`record_performance`): stores max observed tok/s and peak RSS per backend ID. Used to boost scoring in future selections.

### `src/orchestration/IInferenceEngine.h`
Common interface for all backends:
```cpp
class IInferenceEngine {
    virtual bool initialize(const std::string& model_path, std::string& error) = 0;
    virtual InferenceResult run(const InferenceRequest& req) = 0;
    virtual void shutdown() = 0;
    virtual std::string backend_id() const = 0;
    virtual bool is_ready() const = 0;
};
```

### `src/orchestration/HydroXLEngineAdapter.h/.cpp`
Two concrete `IInferenceEngine` implementations:

**`HydroXLCpuEngineAdapter`**: Full working adapter.
- `initialize()`: loads `HydroXLModel`, opens GGUF tokenizer, allocates `HydroXLKVCache`, attaches `HydroXLForwardPass`
- `run()`: runs prefill loop over prompt tokens, then greedy/sampled generation via `HydroXLSampler`, measures tok/s
- `shutdown()`: releases all owned resources; mutex-protected

**`HydroXLVulkanEngineAdapter`**: Documented stub.
- `initialize()` returns `false` with an honest error: "HydroXL Vulkan compute pipeline for tensor weights is not ready (fallback required)"
- Triggers the designed CPU fallback path in `create_engine()`
- Placeholder for when the Vulkan compute shader pipeline for weight tensors is complete

### `src/orchestration/OrchestratorInit.h/.cpp`
`register_default_backends()` — registers at program startup:

| Backend | Quant Types | Architectures | Requires | Throughput | Priority |
|---------|-------------|---------------|----------|------------|----------|
| `hydroxl_cpu` | 0,1,2,6,7,8,10,11,12,13,14,15,16,17 | llama, qwen2, mistral, phi, gemma | nothing | 1.0× | 10 |
| `hydroxl_vulkan` | 0,1,7,8 | llama, qwen2 | Vulkan | 1.5× (heuristic) | 20 |
| `cuda` | all | all | CUDA SDK | 5.0× (heuristic) | 30 |

---

## Hardware Compatibility Matrix

| Hardware | CPU Backend | Vulkan Backend | CUDA Backend |
|----------|-------------|----------------|--------------|
| Intel Iris Xe (this machine) | ✅ Verified | ✅ Detected, init returns fallback | ❌ Not available |
| NVIDIA GPU | ✅ | ✅ (Vulkan capable) | ✅ (when CUDA SDK present) |
| AMD GPU | ✅ | ✅ (Vulkan capable) | ❌ (no ROCm path yet) |
| ARM / Apple | ✅ | ⚠️ Vulkan support varies | ❌ |
| CPU-only server | ✅ | ❌ | ❌ |

---

## Fallback Behavior

```
User calls create_engine(sel, req, error)
    │
    ├── Factory found? NO → try hydroxl_cpu factory (if allow_cpu_fallback)
    │
    └── Factory found? YES → engine->initialize(model_path)
            │
            ├── Success → return engine
            │
            └── Failure → try hydroxl_cpu factory (if allow_cpu_fallback)
                    │
                    ├── Success → return CPU fallback engine
                    └── Failure → return nullptr + error string
```

All fallback events are reported in `SelectionResult::is_fallback` and logged to `error_out`.

---

## How to Add a New Backend

1. Create a class that implements `IInferenceEngine` (see `HydroXLEngineAdapter.h`)
2. Build it into `agr_orchestration` or a separate library
3. Call `orch.register_backend(caps)` with a `BackendCapabilities` descriptor
4. Call `orch.register_engine_factory("my_backend", []() { return make_shared<MyBackend>(); })`
5. The orchestrator will automatically evaluate and score it during `select_backend()`

For plugin-based backends (Phase 18/19 plugin SDK), the plugin loader can call `register_engine_factory` after ABI negotiation succeeds.

---

## Benchmark Results (Phase 21, Intel i5-1235U, 16 GB RAM, Intel Iris Xe)

**Orchestration overhead:**

| Operation | Measured Latency |
|-----------|-----------------|
| `HardwareDetector::detect()` | ~1 ms (Vulkan enumeration) |
| `select_backend()` (no history) | < 50 µs |
| `select_backend()` (with history) | < 50 µs |
| `create_engine()` + `initialize()` | ~1,656 ms (model load dominates) |

**End-to-end test (`test_p21_e2e_orchestration`, real Qwen2.5-0.5B Q4_K_M):**
- Selected backend: `hydroxl_cpu` (Q4_K type 12 not in Vulkan's shader list)
- Engine init: measured in test (9.18 sec total test time including inference)
- Inference correctness: ✅ greedy deterministic, pass 1 == pass 2
- Benchmark history update: ✅ `used_benchmark_history=true` on second selection

**Concurrency:**
- 10 threads × 50 selections each = 500 concurrent selections: all succeeded (no deadlock, no crash)

---

## Test Suite

| Test | Coverage | Result |
|------|----------|--------|
| `test_p21_hardware_detector` | detect(), RAM, Vulkan, best_gpu(), idempotency, print_summary | ✅ PASSED |
| `test_p21_orchestrator` | selection, forced override, quant rejection, RAM rejection, fallback, perf history, concurrency (500 selections), diagnostics | ✅ PASSED |
| `test_p21_backend_capabilities` | CPU/Vulkan/CUDA mock environments, VRAM gate, quant/arch matching, wildcard, scoring ordering | ✅ PASSED |
| `test_p21_e2e_orchestration` | Real model load via orchestrator, inference, perf recording, determinism, shutdown | ✅ PASSED (skips if model absent) |

**Full regression: 103/103 tests passed — 0 failed, 0 regressions vs Phase 20.**

---

## Known Limitations

1. **HydroXL Vulkan backend**: `HydroXLVulkanEngineAdapter::initialize()` returns `false` — the Vulkan compute shader pipeline for weight tensor operations is not yet complete. Vulkan is detected and the backend is registered, but inference always falls back to CPU. This is documented behavior, not a regression.

2. **AMD/Apple hardware**: Vendor detection is implemented; Vulkan enumeration will work on AMD/Apple Vulkan drivers, but the Vulkan inference pipeline limitation (point 1) means CPU is used regardless.

3. **CUDA backend**: Compiled under `#ifdef HETEROACCEL_CUDA_ENABLED`. Registered and scored correctly when CUDA is present, but not tested on this machine (no NVIDIA GPU). The CUDA path requires a future `CudaEngineAdapter` implementation.

4. **Plugin-to-orchestrator wiring**: The Phase 18/19 PluginLoader discovers plugins and validates ABI, but does not yet automatically call `EngineOrchestrator::register_engine_factory()`. Plugin engines must be manually wired at startup.

5. **Multi-GPU**: `best_gpu()` picks the GPU with the most VRAM. Heterogeneous multi-GPU scheduling (splitting a model across two GPUs) is not implemented.

---

## Phase 22 Recommendations

1. **Complete HydroXL Vulkan tensor pipeline** — implement weight-loading and matmul compute shaders so `HydroXLVulkanEngineAdapter` can pass `initialize()` and run inference on Intel Iris Xe / AMD / NVIDIA Vulkan devices.

2. **Plugin-to-orchestrator auto-wiring** — after `PluginLoader::plugin_discover()` succeeds and ABI is validated, automatically call `EngineOrchestrator::register_engine_factory()` so plugin backends participate in scoring.

3. **Persistent performance profile** — serialize `perf_history_` to a JSON file so benchmark history survives across process restarts, enabling warm selection decisions from the first run.

4. **CUDA engine adapter** — implement `CudaEngineAdapter` using llama.cpp's CUDA backend so NVIDIA GPU inference is reachable through the orchestration layer without bypassing HydroXL.

5. **Context-aware memory estimation** — replace the fixed `model_size × 1.2` heuristic with a model-aware estimate that accounts for KV cache size (`n_layers × n_kv_heads × n_head_dim × n_ctx × 2 × dtype_size`) to make RAM/VRAM gate decisions more accurate for long-context workloads.
