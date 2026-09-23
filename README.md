# Adaptive GPU Runtime

Universal Adaptive GPU Acceleration Runtime (HeteroAccel). A heterogeneous, hardware-aware local AI execution runtime that dynamically distributes AI computation across CPU and GPU.

Currently in **Phase 3**: Real LLM Inference via `llama.cpp` and Vulkan GPU offloading.

## Architecture

```
User Application / CLI
       â”‚
HeteroAccel Runtime
       â”‚
llama.cpp Adapter (LlamaCppEngine)
       â”‚
llama.cpp (via FetchContent)
       â”‚
ggml_vulkan backend
       â”‚
Intel Iris Xe (or other Vulkan GPU)
```

## Phases

### Phase 1: Hardware Detection (Verified)
- CPU detection (vendor, architecture, physical/logical cores)
- RAM detection
- GPU classification (Integrated/Dedicated, VRAM, Shared system memory)
- Vulkan detection (loader, devices, compute queues)
- CUDA detection (with honest unavailable reporting)

### Phase 2: Vulkan Compute Backend (Verified)
- Native Vulkan vector-addition backend
- Buffer management (handles Iris Xe unified memory efficiently without staging)
- `glslangValidator` shader compilation
- CPU vs GPU benchmark

### Phase 3: Real LLM Inference (Verified)
- Integration of `llama.cpp` (Vulkan-enabled, shallow-cloned at configure time)
- Loading external GGUF models
- CPU-only inference reference mode
- Vulkan GPU offloaded inference mode
- Honest metrics (tokens/sec, load time, generation time, backend, actual GPU layers)

---

## Build Requirements (Windows)

- Visual Studio 2022 or 2026 (MSVC)
- CMake >= 3.20
- Vulkan SDK (must include `glslangValidator`)
- **~2 GB free disk space** (llama.cpp source + build artifacts + GGUF model)

```powershell
# 1. Configure (will download llama.cpp, takes a few minutes)
cmake -S . -B build -A x64

# 2. Build Release (llama.cpp is large, takes 5-15 mins)
cmake --build build --config Release

# 3. Test (model-dependent tests will cleanly SKIP if no model is found)
ctest --test-dir build -C Release --output-on-failure
```

---

## Running Phase 3 (LLM Inference)

### 1. Get a GGUF Model
We do **not** commit models to the repository. You must download one.
Recommended for Intel Iris Xe testing:
- **Qwen2.5-0.5B-Instruct-Q4_K_M.gguf** (~394 MB)
- **TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf** (~669 MB)

Place the model anywhere on your machine (e.g., `C:\models\Qwen.gguf`).

### 2. Basic Inference (Auto GPU Offload)
```powershell
.\build\Release\adaptive-gpu.exe llm `
    --model "C:\models\Qwen.gguf" `
    --prompt "Explain what Vulkan is in one sentence."
```
*The runtime will automatically intercept llama.cpp logs to confirm whether the Vulkan backend was genuinely used and on which device.*

### 3. CPU Reference Mode
```powershell
.\build\Release\adaptive-gpu.exe llm `
    --model "C:\models\Qwen.gguf" `
    --cpu `
    --prompt "Explain what Vulkan is in one sentence."
```

### 4. CPU vs Vulkan Benchmark
Runs a side-by-side inference comparison using the same model and prompt.
```powershell
.\build\Release\adaptive-gpu.exe benchmark llm `
    --model "C:\models\Qwen.gguf"
```
> **Note on Integrated GPUs**: It is completely normal for a Vulkan-offloaded run on an integrated GPU to be slower than the CPU run for small models. CPU-to-integrated-GPU synchronization and kernel launch overhead often outweigh the parallelization benefit for small workloads. This runtime accurately measures and reports this reality rather than fabricating fake speedups.

### 5. Using the Environment Variable
To avoid passing `--model` every time, set the environment variable:
```powershell
$env:HETEROACCEL_MODEL_PATH = "C:\models\Qwen.gguf"

# Now the tests that require a model will automatically run instead of skipping
ctest --test-dir build -C Release --output-on-failure

# And CLI commands work without --model
.\build\Release\adaptive-gpu.exe llm --prompt "Hello"
```

---

## Phase 1 & 2 Commands

```powershell
.\build\Release\adaptive-gpu.exe hardware
.\build\Release\adaptive-gpu.exe hardware --json
.\build\Release\adaptive-gpu.exe benchmark vulkan
```


## Phase 4: Unified Memory Management

Adds the `agr_mem` library — a cross-backend memory manager for CPU and Vulkan GPU memory.

### Features
- **CPUAllocator** — `std::malloc`-backed, with peak tracking
- **VulkanAllocator** — reuses the existing `VulkanBackend`, queries heap sizes from `vkGetPhysicalDeviceMemoryProperties`
- **TransferManager** — CPU↔GPU data movement with bandwidth telemetry
- **MemoryPool** — free-list slab cache (4 MB default) for CPU and GPU blocks
- **ResidencyManager** — tracks block location state per block ID
- **PressureMonitor** — utilisation-based pressure levels (NORMAL/WARNING/HIGH/CRITICAL)
- **EvictionPolicy** — LRU + priority scoring (CRITICAL never evicted)
- **MemoryManager** — unified facade: `allocate()`, `release()`, `move()`, `statistics()`

### CLI
```bash
adaptive-gpu memory     # Show memory manager stats + transfer telemetry
```

### Tests
27/27 CTest tests pass (9 new Phase 4 tests + 18 existing Phase 1–3 tests).

See [docs/phase4-memory.md](docs/phase4-memory.md) for full API docs.
