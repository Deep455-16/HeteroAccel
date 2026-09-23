# HeteroAccel

> **Hardware-adaptive heterogeneous compute runtime that automatically selects CPU, Vulkan, or CUDA. The user never chooses a backend.**

HeteroAccel inspects the host machine at startup, discovers all available compute backends (CPU, Vulkan GPU, CUDA), scores them against workload requirements, and selects the best execution path automatically.

```cpp
// What the user writes:
runtime.execute(task);

// What HeteroAccel decides internally (Intel Iris Xe machine):
//  CPU     detected : 12th Gen Intel Core i5-1235U  score=120
//  Vulkan  detected : Intel Iris Xe Graphics         score=379  <- selected
//  CUDA    absent   : no NVIDIA driver               score=0
//
//  Primary: Vulkan  |  Fallback: CPU
```

Backends are implementation details. The runtime decides.

---

## Architecture

```
                         HeteroAccel
                              |
                    Hardware Discovery
                              |
                    Capability Analysis
                              |
                    BackendManager
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
             CPU           Vulkan           CUDA
              |               |               |
              +---------------+---------------+
                              |
                    DeviceSelector (scoring model)
                              |
                    MemoryManager (unified alloc)
                              |
                    Workload Execution
```

### Automatic Selection Flow

```
All Candidates
      |
  +---+---+
  CPU  Vulkan  CUDA
   |     |       |
score  score   score
   |     |       |
  +---+---+-------+
          |
     Best Candidate
      (or CPU fallback)
```

**On Intel Iris Xe (this machine):**
```
CPU     available  score ~120
Vulkan  available  score ~379  <- winner
CUDA    absent     score   0
```

**On NVIDIA RTX system:**
```
CPU     available  score ~120
Vulkan  available  score ~379
CUDA    available  score ~550  <- winner
```

---

## Design Principle

> HeteroAccel automatically detects the host CPU and available accelerator backends,
> evaluates their capabilities using a scoring model, and selects an appropriate
> execution path. **CUDA and Vulkan are backend implementations hidden behind the
> hardware-adaptive runtime layer.**
>
> The long-term goal:
> `Any supported hardware → detection → capability analysis → auto selection → heterogeneous scheduling → maximum local performance`

**The user never writes:**
```cpp
if (cudaAvailable)   useCUDA();
else if (vulkan)     useVulkan();
```
That logic lives inside HeteroAccel.

---

## Phases

### Phase 1 — Hardware Detection ✅
- CPU: vendor, model, physical/logical cores, ISA features
- RAM: total and available physical memory
- GPU: integrated/dedicated classification
- Vulkan: loader, devices, compute queues, memory heaps
- CUDA: runtime driver discovery via dynamic load — graceful, never crashes

### Phase 2 — Vulkan Compute Backend ✅
- Native Vulkan vector-add compute pipeline
- Buffer management for unified (Iris Xe) and discrete GPU memory
- GLSL → SPIR-V shader compilation
- CPU vs GPU benchmark

### Phase 3 — LLM Inference via llama.cpp ✅
- `llama.cpp` integrated via CMake FetchContent (static, Vulkan-enabled)
- CPU-only and Vulkan GPU-offloaded inference modes
- Automatic Vulkan detection from llama.cpp logs
- Honest telemetry: tokens/sec, load time, GPU layers offloaded
- Validated: Qwen2.5-0.5B Q4_K_M — 25/25 layers to Intel Iris Xe via Vulkan

### Phase 4 — Unified Memory + Automatic Backend Selection ✅

#### Memory Infrastructure (`agr_mem`)
| Component | Role |
|---|---|
| `CPUAllocator` | `std::malloc` with peak tracking |
| `VulkanAllocator` | Reuses `VulkanBackend`, queries Vulkan heap sizes |
| `CUDAAllocator` | Graceful stub; real cudaMalloc in Phase 5 |
| `TransferManager` | CPU↔GPU data movement + bandwidth telemetry |
| `MemoryPool` | Free-list slab cache for block reuse |
| `ResidencyManager` | Per-block location state tracking |
| `PressureMonitor` | Utilisation-based pressure (NORMAL/WARNING/HIGH/CRITICAL) |
| `EvictionPolicy` | LRU + priority scoring (CRITICAL never evicted) |
| `MemoryManager` | Unified facade: `allocate()`, `release()`, `move()`, `statistics()` |

`MemoryLocation::ACCELERATOR` lets HeteroAccel pick the right accelerator automatically.

#### Automatic Backend Selection (`agr_backend`)
| Component | Role |
|---|---|
| `ComputeDevice` | Hardware-independent device descriptor (CPU / Vulkan / CUDA) |
| `BackendManager` | Discovers all backends at startup |
| `DeviceSelector` | Scoring model → automatic selection; CPU always the fallback |

---

## Build

```powershell
# Configure (downloads llama.cpp ~2 GB, takes a few minutes)
cmake -S . -B build -A x64

# Build Release (5-15 minutes first time)
cmake --build build --config Release -j 4

# Run all tests
ctest --test-dir build -C Release --output-on-failure
```

**Requirements:** MSVC / VS BuildTools 2022+, CMake ≥ 3.20, Vulkan SDK 1.3+

CUDA is **optional** — builds and runs perfectly without NVIDIA hardware.

---

## CLI Commands

```powershell
# Hardware discovery + automatic backend selection
.\build\Release\adaptive-gpu.exe devices

# Full hardware info (JSON)
.\build\Release\adaptive-gpu.exe hardware
.\build\Release\adaptive-gpu.exe hardware --json

# Vulkan vector-add benchmark
.\build\Release\adaptive-gpu.exe benchmark vulkan

# LLM inference (auto GPU offload)
.\build\Release\adaptive-gpu.exe llm --model C:\models\Qwen.gguf --prompt "What is Vulkan?"

# LLM CPU-only
.\build\Release\adaptive-gpu.exe llm --model C:\models\Qwen.gguf --cpu

# LLM CPU vs Vulkan benchmark
.\build\Release\adaptive-gpu.exe benchmark llm --model C:\models\Qwen.gguf

# Memory manager stats
.\build\Release\adaptive-gpu.exe memory
```

### `devices` output — Intel Iris Xe machine

```
HeteroAccel Hardware Configuration
===================================

Vulkan
  Name:    Intel(R) Iris(R) Xe Graphics
  API:     1.3
  Memory:  7.87 GB
  Status:  AVAILABLE
  Score:   378.655

CPU
  Name:    12th Gen Intel(R) Core(TM) i5-1235U
  Vendor:  GenuineIntel
  Memory:  15.73 GB
  Cores:   10
  Status:  AVAILABLE
  Score:   120

CUDA
  Name:    CUDA
  Status:  UNAVAILABLE
  Reason:  CUDA driver library not found -- no NVIDIA driver installed
  Score:   0

Backend Decision (automatic)
  Primary Accelerator: Vulkan (Intel(R) Iris(R) Xe Graphics)
  CPU Fallback:        ENABLED

HeteroAccel selects backends automatically.
CUDA and Vulkan are implementation details hidden from the caller.
```

---

## Tests

**33/33 CTest tests pass** (3 skipped — LLM tests need `HETEROACCEL_MODEL_PATH`).

| Group | Tests | Coverage |
|---|---|---|
| Phase 1: Hardware | 6 | CPU, RAM, GPU, Vulkan, CUDA, JSON |
| Phase 2: Vulkan compute | 6 | lifecycle, vector-add, buffers, descriptor reuse |
| Phase 2: CPU reference | 2 | vector-add, benchmark formatting |
| Phase 3: LLM | 5 | availability, model path, CPU infer, Vulkan infer, mode |
| Phase 4: Memory | 9 | CPU alloc, Vulkan alloc, transfers, pool, LRU, eviction |
| Phase 4: Backend selection | 6 | CPU detect, Vulkan detect, CUDA graceful, enum, auto-select, fallback |

Run LLM tests with a model:
```powershell
$env:HETEROACCEL_MODEL_PATH = "C:\models\Qwen.gguf"
ctest --test-dir build -C Release --output-on-failure
```

---

## Hardware Tested

| Component | Detail |
|---|---|
| CPU | Intel Core i5-1235U (10 cores) |
| GPU | Intel Iris Xe (integrated, unified memory) |
| RAM | ~16 GB |
| Vulkan | SDK 1.4.357 — available |
| CUDA | unavailable (no NVIDIA hardware) |
| OS | Windows 11 |
| Compiler | MSVC 19.50 / VS BuildTools |
| CMake | 4.3.3 |

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | ✅ | Hardware detection (CPU, RAM, Vulkan, CUDA) |
| 2 | ✅ | Vulkan compute backend |
| 3 | ✅ | LLM inference (llama.cpp + Vulkan) |
| 4 | ✅ | Unified memory + automatic backend selection |
| 5 | planned | Dynamic heterogeneous scheduling |
| 6 | planned | Layer/model placement, prefetch, streaming |

---

See [docs/phase4-memory.md](docs/phase4-memory.md) for the full memory subsystem API.
