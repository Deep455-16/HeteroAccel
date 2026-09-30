# HeteroAccel

> **Hardware-adaptive heterogeneous compute runtime that automatically selects CPU, Vulkan, or CUDA based on runtime conditions, telemetry, and capabilities. Includes full model/layer streaming and prefetch infrastructure for running models larger than accelerator memory.**

HeteroAccel inspects the host machine at startup, discovers all available compute backends, scores them against workload requirements using an adaptive cost model, and dynamically schedules the best execution path — while streaming model layers between Disk → RAM → Accelerator on demand.

```cpp
// Register model and layers
uint64_t modelId = modelManager.registerModel("my-model");
modelManager.registerResource(layerWeights);

// Execute — HeteroAccel handles streaming, scheduling, and prefetch
TaskResult result = modelManager.executeLayer(layerId);
```

The runtime handles: hardware detection, capability analysis, memory management, streaming, prefetching, transfer telemetry, and adaptive feedback.

---

## Architecture

```
                         APPLICATION
                              |
                       HeteroAccel API
                              |
                       ModelManager
                              |
                       LayerManager
                              |
               +──────────────┴──────────────+
               |                             |
        ResidencyManager            PrefetchEngine
               |                             |
               +──────────────┬──────────────+
                              |
                      StreamingEngine
                       |            |
              StorageBackend    MemoryManager (Phase 4)
              (Disk I/O)              |
                              Phase 5 Scheduler
                                      |
                  +───────────────────┼───────────────────+
                  |                   |                   |
                 CPU               Vulkan              CUDA
```

### Streaming Pipeline

```
SSD / Disk
    ↓  StorageBackend (IStorageBackend)
System RAM  →  WARM residency
    ↓  MemoryManager + Scheduler
Execution Device  →  HOT residency
    ↓  Execution
(WARM on demand, DISK on eviction)
```

### Residency State Machine

```
DISK ──LOADING──> WARM ──LOADING──> HOT
 ^                  |                |
 |    EVICTING <────┘    EVICTING <──┘
 └────────────────────────────────────┘
```

| State    | Meaning |
|----------|---------|
| `DISK`   | On persistent storage only |
| `COLD`   | Known to runtime, not in memory |
| `WARM`   | In system RAM, ready for promotion |
| `HOT`    | On execution device (GPU/CPU) |
| `LOADING`| Being transferred |
| `EVICTING`| Being removed from a tier |

---

## Phases

### Phase 1 — Hardware Detection ✅
- CPU, RAM, Vulkan, CUDA detection — graceful across all hardware.

### Phase 2 — Vulkan Compute Backend ✅
- Native Vulkan vector-add compute pipeline, buffer management, descriptor pool reuse.

### Phase 3 — LLM Inference via llama.cpp ✅
- Integrated `llama.cpp` (static, Vulkan-enabled), automatic GPU layer offloading.

### Phase 4 — Unified Memory + Backend Discovery ✅
- Unified `MemoryManager` with `CPUAllocator`, `VulkanAllocator`, `CUDAAllocator`.
- `BackendManager` discovers CPU, Vulkan, CUDA as `ComputeDevice` objects.
- Free-list slab cache, LRU eviction, utilisation-based pressure monitoring.

### Phase 5 — Adaptive Heterogeneous Scheduler ✅
- `AdaptiveScheduler` dispatches workloads via `CostModel` + `PerformanceHistory` (EMA).
- Asynchronous `CPUWorker`, `VulkanWorker`, `CUDAWorker` with fallback chains.

### Phase 6 — Model / Layer Manager, Streaming & Prefetch ✅
- **`IStorageBackend` / `FileStorageBackend`**: portable disk I/O abstraction.
- **`ResourceResidencyManager`**: explicit DISK→COLD→WARM→HOT state machine.
- **`StreamingEngine`**: async tier transitions using Phase 4 MemoryManager.
- **`LayerManager`**: registers generic `ModelLayer` nodes with dependency graph.
- **`PrefetchEngine`**: memory-pressure-aware look-ahead prefetch (configurable distance).
- **`ModelManager`**: top-level API — register models, resources, execute layers.

---

## Build

```powershell
# Configure
cmake -S . -B build -A x64

# Build Release
cmake --build build --config Release -j 4

# Run all tests
ctest --test-dir build -C Release --output-on-failure
```

**Requirements:** MSVC / VS BuildTools 2022+, CMake ≥ 3.20, Vulkan SDK 1.3+
*(CUDA is optional and gracefully bypassed if unavailable)*

---

## CLI Commands

```powershell
# Hardware discovery + automatic backend selection (Phase 4)
.\build\Release\adaptive-gpu.exe devices

# Adaptive Scheduler diagnostic/benchmark (Phase 5)
.\build\Release\adaptive-gpu.exe scheduler

# Memory manager stats (Phase 4)
.\build\Release\adaptive-gpu.exe memory

# Vulkan vector-add benchmark (Phase 2)
.\build\Release\adaptive-gpu.exe benchmark vulkan

# LLM inference — requires model file (Phase 3)
.\build\Release\adaptive-gpu.exe llm --model C:\models\model.gguf --prompt "Hello"
```

### `devices` output on Intel Iris Xe

```
HeteroAccel Hardware Configuration
===================================

CPU
  Name:    Intel Core i5-1235U
  Memory:  15.7 GB  |  Cores: 10  |  Status: AVAILABLE  |  Score: 120

Vulkan
  Name:    Intel Iris Xe Graphics
  API:     1.3  |  Memory: 7.9 GB  |  Status: AVAILABLE  |  Score: 379

CUDA
  Status:  UNAVAILABLE — No NVIDIA driver or GPU found

Backend Decision (automatic)
  Primary: Vulkan (Intel Iris Xe Graphics)
  CPU Fallback: ENABLED
```

---

## Tests

**43/43 CTest tests pass** (3 skipped — LLM tests require `HETEROACCEL_MODEL_PATH`).

| Group | Tests | Coverage |
|-------|-------|----------|
| Phase 1: Hardware | 6 | CPU, RAM, GPU, Vulkan, CUDA, JSON |
| Phase 2: Vulkan compute | 6 | lifecycle, vector-add, buffers, descriptor reuse |
| Phase 2: CPU reference | 2 | vector-add, benchmark formatting |
| Phase 3: LLM | 5 | availability, model path, CPU/Vulkan infer, mode |
| Phase 4: Memory | 9 | CPU alloc, Vulkan alloc, transfers, pool, eviction |
| Phase 4: Backends | 6 | CPU detect, Vulkan detect, CUDA graceful, enum, auto-select, fallback |
| Phase 5: Scheduler | 6 | cost model, OOM rejection, residency bias, feedback loop, fallback, CPU-only |
| Phase 6: Model/Layer | 4 | model manager, layer dependencies, residency transitions, prefetch engine |

To run LLM tests with a real model:
```powershell
$env:HETEROACCEL_MODEL_PATH = "C:\models\model.gguf"
ctest --test-dir build -C Release --output-on-failure
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Hardware detection (CPU, RAM, Vulkan, CUDA) |
| 2 | ✅ | Vulkan compute backend |
| 3 | ✅ | LLM inference (llama.cpp + Vulkan) |
| 4 | ✅ | Unified memory + backend capability discovery |
| 5 | ✅ | Adaptive Heterogeneous Scheduler (Cost Models, Telemetry, Async Workers) |
| 6 | ✅ | Model / Layer Manager, Streaming & Prefetch Engine |
| 7 | planned | llama.cpp integration — layer-by-layer streaming inference |
| 8 | planned | Auto-tuner — ML-driven prefetch distance and scheduling |

---

## Hardware Tested

| Component | Detail |
|-----------|--------|
| CPU | Intel Core i5-1235U (10 cores) |
| GPU | Intel Iris Xe (integrated, unified memory) |
| RAM | ~16 GB |
| Vulkan | SDK 1.4 — available |
| CUDA | unavailable (no NVIDIA hardware) |
| OS | Windows 11 |
| Compiler | MSVC 19.50 / VS BuildTools |

NVIDIA systems (CPU + Vulkan + CUDA) are fully supported via Phase 5 automatic selection — no code changes needed.

---

## Design Principle

> HeteroAccel automatically detects hardware, evaluates capabilities using a real-time cost model, and selects the optimal execution backend. CUDA and Vulkan are implementation details hidden behind the runtime. Phase 6 adds the infrastructure to execute models larger than accelerator memory by streaming layers between Disk → RAM → Accelerator on demand, with a configurable prefetch engine that overlaps I/O with computation without tying itself to any specific model format or inference framework.
