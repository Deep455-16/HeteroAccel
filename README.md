# HeteroAccel

> **A hardware-adaptive heterogeneous compute runtime for LLM inference, built as an orchestration and scheduling layer over `llama.cpp`.**

HeteroAccel automatically evaluates system capabilities (CPU, Vulkan, CUDA), memory pressure, and historical performance to dynamically configure and execute LLM inference workloads. It is designed for resource-constrained edge devices and integrated GPUs, gracefully falling back to CPU or adjusting thread/layer configurations to ensure stability.

## Current Engineering State (Phase 9 Complete)

This project is an **engineering implementation**, not a theoretical research prototype. The capabilities described below reflect the actual executing code. 

The system operates via an explicit boundary:
1. **HeteroAccel** manages the workload queue, measures memory pressure, consults historical performance, enforces safety constraints (Phase 9), and auto-tunes configurations (Phase 8).
2. **`llama.cpp`** (integrated via FetchContent) executes the actual tensor operations and token generation on the hardware. 

### Key Features

* **Hardware Auto-Detection (Phases 1-3):** Safely detects CPU characteristics, Vulkan availability (e.g., Intel Iris Xe), and CUDA availability without crashing on missing drivers.
* **Memory & Residency Management (Phases 4-6):** Tracks estimated memory pressure. If the system enters a `CRITICAL` state, HeteroAccel enforces CPU-only execution. Note: Phase 6 streaming/residency abstractions exist as structural readiness but do not perform mid-generation tensor migration, respecting the limitations of the underlying `llama.cpp` API.
* **Inference Orchestration (Phase 7):** Integrates directly with `llama.cpp`, providing both synchronous and streaming generation interfaces (`adaptive-gpu run` and `adaptive-gpu chat`).
* **Performance History & Auto-Tuning (Phase 8):** Records metrics (TTFT, tokens/sec) for specific hardware/model combinations. The AutoTuner uses this history to safely explore thread configurations and roll back if regressions occur.
* **Execution Policy & Workload Management (Phase 9):** 
  * The **Execution Policy Engine** evaluates the actual model file size, hardware capabilities, and current memory pressure to dictate the allowed execution envelope (e.g., `max_gpu_layers`).
  * The **AutoTuner** is strictly constrained by the Policy Engine (e.g., it will never attempt GPU execution if the Policy Engine mandates `CPU_FALLBACK`).
  * The **Workload Registry** provides thread-safe tracking of concurrent requests.
  * **Mid-Generation Cancellation:** Thread-safe workload cancellation is wired directly into the `llama.cpp` decoding loop via an atomic cancel flag, stopping generation immediately upon user abort without process termination or unsafe thread kills.
  * **Concurrency Model (intentional):** The runtime mutex is released before entering the token-generation loop. Different backend/model instances can therefore overlap in time. However, requests sharing the same `LlamaCppBackend` instance are serialized by its internal mutex — intentional, because `llama_context` is not thread-safe. HeteroAccel does **not** claim unrestricted same-model concurrent generation.
  * **GPU-Layer Control:** HeteroAccel controls execution through the standard `llama.cpp` API (`n_gpu_layers`, `n_threads`, `n_ctx`, `n_batch`). HeteroAccel does **not** implement custom per-tensor device placement; `n_gpu_layers` is the mechanism used to allocate work between CPU and GPU.

## Architecture Flow

```text
Application / CLI Request
        ↓
Workload Registry (Concurrency & Cancellation tracking)
        ↓
Execution Policy Engine (Safety Constraints: Memory, Model Size)
        ↓
AutoTuner (Performance Optimization within constraints)
        ↓
Dynamic Model Reloading (if constraints require migration)
        ↓
LlamaCppBackend / LlamaCppEngine
        ↓
llama.cpp (CPU / Vulkan / CUDA)
```

## CLI Usage

The runtime provides a built-in CLI for testing and inference.

* **Single prompt:** `adaptive-gpu run <model.gguf> "Hello, world!"`
* **Interactive chat:** `adaptive-gpu chat <model.gguf>`
* **Status & Telemetry:** `adaptive-gpu status`
* **Simulated Workloads:** `adaptive-gpu workloads` (Note: Currently displays workloads active within the current process instance).

## Building and Testing

Requirements: CMake 3.20+, C++17, and a supported compiler (MSVC, GCC, Clang).

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 4
ctest -C Release --output-on-failure
```

The test suite includes 57 rigorous unit and integration tests covering hardware detection, memory pressure emulation, AutoTuner constraint enforcement, mid-generation cancellation, and end-to-end `llama.cpp` integration. Tests requiring actual `.gguf` weights are gracefully skipped if no model is provided, preventing false failures.
