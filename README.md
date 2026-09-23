# HeteroAccel

> **Hardware-adaptive heterogeneous compute runtime that automatically selects CPU, Vulkan, or CUDA based on runtime conditions, telemetry, and capabilities.**

HeteroAccel inspects the host machine at startup, discovers all available compute backends, scores them against workload requirements using an adaptive cost model, and dynamically schedules the best execution path.

`cpp
// What the user writes:
agr::TaskHandle handle = scheduler.schedule(workload);
agr::TaskResult result = scheduler.wait(handle);
`

The runtime handles the rest: hardware detection, capability analysis, memory management, transfer telemetry, and adaptive feedback.

---

## Architecture

`
                         HeteroAccel
                              |
                    Hardware Discovery
                              |
                    BackendManager
                              |
              +---------------+---------------+
              |                               |
        MemoryManager                  AdaptiveScheduler
              |                               |
        Unified Alloc                 +-------+-------+
       Transfers & Evict              |       |       |
                              CostModel PerformanceHistory
                                      |
                              +-------+-------+
                              |       |       |
                            CPU    Vulkan   CUDA
                           Worker  Worker  Worker
                              \       |       /
                               \      |      /
                                +-----v-----+
                                  Execution
`

### Automatic Selection Flow

1. **Workload Definition:** Defines input/output memory, operation scale, and backend-specific execution callbacks.
2. **Cost Evaluation:** CostModel evaluates feasible backends, factoring in:
   - Compute capability
   - Available memory
   - Required data transfers (residency bias)
   - Queue penalties (crude load representation)
3. **Execution:** The AdaptiveScheduler submits the workload to the winning backend's asynchronous IWorker.
4. **Adaptive Feedback:** Upon completion, the PerformanceHistory uses Exponential Moving Average (EMA) to update its internal ops/sec and bandwidth models, making future scheduling smarter.

---

## Phases

### Phase 1 — Hardware Detection ✅
- CPU, RAM, Vulkan, CUDA detection.
- Graceful, crash-free CUDA discovery via dynamic load.

### Phase 2 — Vulkan Compute Backend ✅
- Native Vulkan vector-add compute pipeline.
- Buffer management and descriptor pool reuse.

### Phase 3 — LLM Inference via llama.cpp ✅
- Integrated llama.cpp (static, Vulkan-enabled).
- Automatic Vulkan detection from logs.

### Phase 4 — Unified Memory + Backend Discovery ✅
- Unified MemoryManager with CPUAllocator, VulkanAllocator, CUDAAllocator.
- Free-list slab cache, LRU eviction, utilisation-based pressure monitoring.
- BackendManager unifies CPU, Vulkan, CUDA into ComputeDevice models.

### Phase 5 — Adaptive Heterogeneous Scheduler ✅
- Workload abstraction supporting generic, asynchronous backend callbacks.
- CostModel that evaluates residency penalties and capacity limits.
- PerformanceHistory providing EMA-smoothed telemetry for ops/sec and bandwidth.
- Asynchronous execution workers (CPUWorker, VulkanWorker, CUDAWorker).
- AdaptiveScheduler orchestrates the flow and guarantees safe fallbacks.

---

## Build

`powershell
# Configure
cmake -S . -B build -A x64

# Build Release
cmake --build build --config Release -j 4

# Run all tests
ctest --test-dir build -C Release --output-on-failure
`

**Requirements:** MSVC / VS BuildTools 2022+, CMake ≥ 3.20, Vulkan SDK 1.3+
*(CUDA is optional and gracefully bypassed if unavailable)*

---

## CLI Commands

`powershell
# Adaptive Scheduler diagnostic/benchmark (Phase 5)
.\build\Release\adaptive-gpu.exe scheduler

# Hardware discovery + capability scores
.\build\Release\adaptive-gpu.exe devices

# Memory manager stats
.\build\Release\adaptive-gpu.exe memory

# Vulkan vector-add benchmark
.\build\Release\adaptive-gpu.exe benchmark vulkan

# LLM inference
.\build\Release\adaptive-gpu.exe llm --model C:\models\Qwen.gguf --prompt "What is Vulkan?"
`

### scheduler output — Intel Iris Xe machine

`
HeteroAccel Scheduler Benchmark (Phase 5)
=========================================

Workload: Generic Compute Tensor
Input:    100 MB
Output:   10 MB
Compute:  5000000 ops

Candidate Devices
-----------------
Vulkan
  Estimated compute: 2000 ms
  Transfer cost:     0 ms
  Memory penalty:    10 ms
  Total estimate:    2010 ms

CPU
  Estimated compute: 10000 ms
  Transfer cost:     0 ms
  Memory penalty:    2 ms
  Total estimate:    15003 ms

Selected:
  Vulkan

Actual execution:
  Status:  SUCCESS
  Compute: 16.9342 ms
  Total:   16.9342 ms
`

---

## Tests

**39/39 CTest tests pass** (3 skipped if HETEROACCEL_MODEL_PATH is missing).
Tests cover hardware detection, Vulkan execution, memory pooling/eviction, backend selection, cost modelling, residency bias, OOM rejection, feedback loop updates, and graceful failure fallbacks.

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ | Hardware detection (CPU, RAM, Vulkan, CUDA) |
| 2 | ✅ | Vulkan compute backend |
| 3 | ✅ | LLM inference (llama.cpp + Vulkan) |
| 4 | ✅ | Unified memory + backend capability discovery |
| 5 | ✅ | Adaptive Heterogeneous Scheduler (Cost Models, Telemetry, Async Workers) |
| 6 | planned | Layer/model placement, prefetch, streaming, MoE routing |

