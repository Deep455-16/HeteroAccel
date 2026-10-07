 # HeteroAccel

> **A hardware-adaptive heterogeneous compute runtime for local AI inference, built as an orchestration, scheduling, memory-management, and optimization layer over inference engines such as `llama.cpp`.**

HeteroAccel provides a common runtime layer between local AI applications/inference engines and heterogeneous hardware.

It automatically evaluates available hardware, memory pressure, workload requirements, and historical performance to determine how a workload should execute on the current machine.

The goal is simple:

> **Applications should not need to manually optimize themselves for every CPU, GPU, memory configuration, or operating system. HeteroAccel provides the adaptive execution layer underneath them.**

HeteroAccel is designed for local and privacy-preserving AI workloads where models, prompts, documents, and inference can remain on the user's machine.

---

## What HeteroAccel Is

HeteroAccel is an **adaptive execution and optimization runtime**.

It is not itself an LLM, RAG system, AI IDE, or inference engine.

Instead, it sits between an application/inference engine and the available hardware:

```text
┌──────────────────────────────────────────────────────────┐
│                    AI APPLICATIONS                       │
│                                                          │
│  AI IDE │ Local Chat │ RAG │ Agents │ VLM/VLA │ Tools   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│              INFERENCE / COMPUTE ENGINE                  │
│                                                          │
│       llama.cpp │ Colibrì │ Other Compatible Engines     │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                       HETEROACCEL                       │
│                                                          │
│ Hardware Detection                                       │
│ Model Analysis & Execution Planning                      │
│ Execution Policy                                         │
│ Adaptive Scheduling                                      │
│ Memory & Residency Management                             │
│ Streaming / Prefetch Infrastructure                      │
│ Performance Profiling                                    │
│ Auto-Tuning                                              │
│ Workload Management & Cancellation                       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                     LOCAL HARDWARE                       │
│                                                          │
│ CPU │ GPU │ NPU* │ RAM │ VRAM │ NVMe / SSD              │
└──────────────────────────────────────────────────────────┘

*NPU support depends on the connected inference/backend implementation.
```

### Core Principle

HeteroAccel should be changed when a **generic hardware/runtime capability** is missing.

Application-specific functionality should remain in the application or inference engine.

| Requirement                  | Where it belongs         |
| ---------------------------- | ------------------------ |
| AI IDE interface             | AI IDE                   |
| RAG retrieval                | RAG system               |
| Agent/tool orchestration     | Agent layer              |
| New model architecture       | Inference engine         |
| Tensor/kernel implementation | Inference backend/engine |
| Hardware detection           | **HeteroAccel**          |
| CPU/GPU workload scheduling  | **HeteroAccel**          |
| Memory pressure management   | **HeteroAccel**          |
| RAM/VRAM residency           | **HeteroAccel**          |
| Adaptive execution policy    | **HeteroAccel**          |
| Runtime profiling            | **HeteroAccel**          |
| Hardware-aware autotuning    | **HeteroAccel**          |
| Workload cancellation        | **HeteroAccel**          |

This allows HeteroAccel to act as reusable infrastructure rather than becoming tightly coupled to one application.

---

## Current Engineering State

**Phase 10 — Production Hardening Complete**

The current implementation includes Phases 1–10.

Latest validated state:

* Release build succeeds
* **58/58 tests passed**
* 0 failed
* 0 skipped
* Real CPU LLM inference validated
* Real Vulkan LLM inference validated
* CPU/Vulkan mode distinction validated
* llama.cpp integration validated
* Final repository state clean

The current implementation is an engineering runtime, not merely a theoretical architecture.

---

## What HeteroAccel Actually Does

The runtime is responsible for the following execution pipeline:

```text
Application Request
        ↓
Workload Registration
        ↓
Hardware / Runtime State
        ↓
Execution Policy
        ↓
Performance History
        ↓
Auto-Tuning
        ↓
Scheduling
        ↓
Memory / Residency Management
        ↓
Inference Backend
        ↓
llama.cpp
        ↓
CPU / Vulkan / CUDA
```

HeteroAccel determines the **execution envelope and configuration**.

The inference engine performs the actual model computation.

For the current llama.cpp integration:

```text
HeteroAccel
    ↓
LlamaCppBackend
    ↓
LlamaCppEngine
    ↓
llama.cpp
    ↓
CPU / Vulkan / CUDA
```

HeteroAccel does not replace llama.cpp's tensor kernels.

Instead, it controls and optimizes the environment in which the inference engine executes.

---

## Major Capabilities

### Hardware Auto-Detection

HeteroAccel detects available compute resources including:

* CPU characteristics
* Vulkan availability
* CUDA availability
* available memory information
* accelerator capabilities

It can therefore adapt its execution strategy to different machines.

Example:

```text
Intel CPU + Intel Iris Xe
        ↓
HeteroAccel
        ↓
CPU / Vulkan
```

Another machine:

```text
AMD/Intel CPU + NVIDIA GPU
        ↓
HeteroAccel
        ↓
CPU / CUDA
```

CPU-only machine:

```text
CPU only
   ↓
HeteroAccel
   ↓
CPU execution
```

### Adaptive Scheduling

HeteroAccel evaluates workloads using factors such as:

* hardware capability
* current device load
* memory availability
* estimated compute cost
* transfer cost
* historical performance
* workload priority
* execution constraints

The scheduler therefore does not simply attempt to maximize GPU utilization.

Its objective is to choose an execution strategy appropriate for the workload and current machine state.

### Memory and Residency Management

HeteroAccel maintains a unified abstraction for resources that may reside in:

```text
DISK
 ↓
CPU/RAM
 ↓
GPU/VRAM
```

The runtime tracks resource residency and memory pressure.

It supports concepts such as:

* FREE
* RESIDENT
* LOADING
* EVICTING
* CRITICAL/HIGH/NORMAL/LOW priorities
* CPU/GPU residency
* memory pressure detection
* eviction
* memory pools
* transfer management

If the system reaches a critical memory condition, the execution policy can force CPU-only execution to maintain stability.

#### Important Limitation

The current Phase 6 streaming/residency system provides the runtime infrastructure for model/resource streaming and prefetching, but it does **not** claim arbitrary mid-generation tensor migration inside llama.cpp.

Actual inference placement is currently controlled through the supported llama.cpp configuration mechanisms.

### llama.cpp Integration

HeteroAccel currently provides a real llama.cpp execution path.

The main interface is:

```text
HeteroRuntime
      ↓
LlamaCppBackend
      ↓
LlamaCppEngine
      ↓
llama.cpp
```

HeteroAccel can configure parameters such as:

```text
n_gpu_layers
n_threads
n_ctx
n_batch
```

The current integration therefore allows HeteroAccel to control how llama.cpp uses CPU and supported GPU execution.

#### Important Limitation

HeteroAccel does **not** currently implement arbitrary per-tensor device placement.

For llama.cpp, `n_gpu_layers` is the primary mechanism used to control CPU/GPU allocation.

### Performance Profiling

HeteroAccel records runtime information including:

* time to first token (TTFT)
* tokens per second
* queue time
* execution duration
* hardware/model combinations
* performance history

This information is used to improve future decisions.

### Auto-Tuning

The AutoTuner uses historical performance information to explore configurations such as:

```text
n_threads
n_gpu_layers
```

while respecting the limits imposed by the Execution Policy Engine.

If a configuration causes a regression or failure, the tuner can roll back rather than blindly continuing exploration.

The policy engine always has authority over the allowed execution envelope.

### Execution Policy

The Execution Policy Engine evaluates:

* model size
* hardware capabilities
* current memory pressure
* active workloads
* historical performance

It can select strategies such as:

```text
FULL_RESIDENT
PARTIAL_RESIDENT
STREAMING
CPU_FALLBACK
MEMORY_PRESSURE
```

The policy prevents performance tuning from violating system safety constraints.

For example:

```text
Policy:
CPU fallback required

        ↓

AutoTuner cannot override policy

        ↓

CPU-only execution
```

### Workload Management

HeteroAccel maintains a workload registry for:

* active workloads
* workload priority
* workload classification
* cancellation
* concurrent workload tracking

Cancellation is propagated into the llama.cpp generation loop through a thread-safe atomic cancellation mechanism.

This allows a generation to be stopped without terminating the entire application.

### Concurrency

Different backend/model instances can overlap in time where supported.

Requests sharing the same `LlamaCppBackend` instance are intentionally serialized because the underlying llama.cpp context is not thread-safe.

HeteroAccel therefore does **not** claim unrestricted concurrent generation using one shared llama.cpp context.

---

## How to Use HeteroAccel Today

There are currently two primary ways to use HeteroAccel.

### Method 1 — Use the Built-in CLI

This is the easiest way to test HeteroAccel.

#### Step 1 — Build

Requirements:

* CMake 3.20+
* C++17 compiler
* MSVC, GCC, or Clang
* supported Vulkan/CUDA environment when GPU execution is desired

```bash
mkdir build
cd build

cmake ..
cmake --build . --config Release -j 4
```

#### Step 2 — Run the Test Suite

```bash
ctest -C Release --output-on-failure
```

A successful current build should report:

```text
58/58 tests passed
```

#### Step 3 — Obtain a Compatible GGUF Model

HeteroAccel's current end-to-end LLM path uses llama.cpp-compatible GGUF models.

For example:

```text
Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

Place the model somewhere accessible to the machine.

Example:

```text
C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

#### Step 4 — Run a Local Prompt

```bash
adaptive-gpu run <model.gguf> "Explain what heterogeneous computing is."
```

Example:

```bash
adaptive-gpu run C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf "Explain heterogeneous computing."
```

This gives you an actual local LLM inference workload passing through:

```text
CLI
 ↓
HeteroAccel
 ↓
LlamaCppBackend
 ↓
llama.cpp
 ↓
CPU / Vulkan
```

#### Step 5 — Run Interactive Chat

```bash
adaptive-gpu chat <model.gguf>
```

For example:

```bash
adaptive-gpu chat C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

You can then interact with the local model through the HeteroAccel runtime.

#### Step 6 — Inspect Runtime Status

```bash
adaptive-gpu status
```

This allows you to inspect runtime/hardware information exposed by HeteroAccel.

#### Step 7 — Inspect Workloads

```bash
adaptive-gpu workloads
```

This displays workloads tracked by the current process instance.

It is primarily useful for observing the runtime's workload-management system.

#### Step 8 — Inspect Profiling Information

```bash
adaptive-gpu profile --show
```

This exposes the performance profile information collected by HeteroAccel.

---

## Testing CPU vs GPU/Vulkan Execution

One of the simplest demonstrations of HeteroAccel is comparing the same model under different execution configurations.

### CPU Mode

Use:

```text
n_gpu_layers = 0
```

The model executes through the CPU path.

### Vulkan Mode

Use a configuration with GPU layers enabled, for example:

```text
n_gpu_layers = 99
```

when supported by the model/hardware configuration.

On a Vulkan-capable system, HeteroAccel can therefore execute through the Vulkan path.

You can compare:

```text
CPU
 ↓
tokens/sec
latency
memory
```

against:

```text
Vulkan
 ↓
tokens/sec
latency
memory
```

This is currently one of the most direct demonstrations of HeteroAccel's adaptive execution concept.

---

## How Another Developer Can Use HeteroAccel

A developer does not have to use the CLI.

The intended architecture allows another application to integrate with the runtime API.

Conceptually:

```text
Developer Application
        ↓
HeteroAccel Runtime API
        ↓
Execution Policy
        ↓
Scheduler
        ↓
Memory / Residency
        ↓
Inference Backend
```

The application provides the workload.

HeteroAccel handles the hardware/runtime decisions.

This makes HeteroAccel suitable as an infrastructure layer underneath other local AI software.

---

## Who Can Use HeteroAccel?

### Students

Students can use HeteroAccel to:

* run local GGUF models
* experiment with CPU/GPU inference
* study heterogeneous computing
* study adaptive scheduling
* study memory management
* benchmark local inference

The CLI is the simplest entry point.

### AI/ML Developers

AI developers can integrate HeteroAccel underneath a local inference application.

Example:

```text
Local AI Application
        ↓
HeteroAccel
        ↓
llama.cpp
        ↓
Local Hardware
```

This can reduce the amount of hardware-specific execution logic that the application itself needs to manage.

### AI Application Developers

Applications such as:

* local AI assistants
* RAG systems
* AI coding tools
* AI IDEs
* autonomous agents
* VLM/VLA applications

can potentially use HeteroAccel as their local execution layer.

The application remains responsible for its own functionality.

HeteroAccel remains responsible for hardware/resource execution.

### Researchers

Researchers can use HeteroAccel to investigate:

* heterogeneous scheduling
* CPU/GPU execution
* memory pressure
* adaptive inference
* performance profiling
* runtime autotuning
* local AI execution

The runtime also provides a foundation for future benchmarking and research experiments.

### Developers Building Inference Engines

Future inference engines can potentially use HeteroAccel as a hardware/resource orchestration layer.

For example:

```text
Colibrì
  ↓
HeteroAccel
  ↓
CPU / GPU / RAM / NVMe
```

In this architecture:

**Colibrì answers:**

> How should this model be executed?

**HeteroAccel answers:**

> Given this machine and its current resources, how should the available hardware and memory be used to execute that workload?

This separation keeps the inference engine and hardware runtime independently reusable.

---

## Using HeteroAccel as the Runtime / "Code Processor"

HeteroAccel should not be described as a compiler or a traditional source-code processor.

Its role is better understood as an **execution processor/runtime layer**.

For an application, the flow is:

```text
Application
     ↓
Create workload
     ↓
HeteroAccel evaluates:
     ├── Hardware
     ├── Memory
     ├── Workload
     ├── Historical performance
     └── Execution constraints
     ↓
Execution Policy
     ↓
Scheduler
     ↓
Backend
     ↓
Actual computation
```

Therefore, an application developer does not need to manually write:

```text
if NVIDIA:
    use CUDA

elif Intel Vulkan:
    use Vulkan

elif CPU:
    use CPU
```

for every application-level workload.

Instead, the application can target the HeteroAccel runtime interface, while HeteroAccel handles the supported hardware-specific execution path underneath.

---

## Example: Building an AI IDE on HeteroAccel

A future AI IDE could look like:

```text
┌───────────────────────────────┐
│            AI IDE             │
│                               │
│ Code editor                   │
│ Chat                          │
│ Agent                         │
│ RAG                           │
└───────────────┬───────────────┘
                ↓
        Inference Engine
                ↓
           HeteroAccel
                ↓
     CPU / GPU / Memory
```

The IDE does not need to become responsible for:

* detecting every GPU
* monitoring memory pressure
* deciding GPU-layer limits
* maintaining performance history
* managing execution policy
* implementing workload cancellation
* implementing hardware-specific scheduling

Those concerns belong to the runtime layer.

---

## Example: Building a Local RAG System

A local RAG application could use:

```text
Documents
   ↓
Local RAG Database
   ↓
Retrieval
   ↓
Inference Engine
   ↓
HeteroAccel
   ↓
Local Hardware
```

HeteroAccel does not perform the retrieval itself.

Instead, once the RAG system has generated an inference workload, HeteroAccel can provide the adaptive execution layer underneath the inference engine.

---

## Example: Building an Agent System

A local agent can use:

```text
User
 ↓
Agent
 ├── Tools
 ├── Memory
 ├── RAG
 └── Inference Engine
          ↓
      HeteroAccel
          ↓
      Hardware
```

The agent decides **what to do**.

The inference engine decides **how the model computes**.

HeteroAccel decides **how the available local execution resources should be managed**.

---

## Future Colibrì Integration

HeteroAccel is intentionally designed so that a future inference engine such as Colibrì does not require a completely different hardware-management system.

The intended architecture is:

```text
AI Applications
       ↓
     Colibrì
       ↓
  HeteroAccel
       ↓
CPU / GPU / NPU / RAM / NVMe
```

The important principle is:

> **Do not modify HeteroAccel simply because a new application exists.**

Instead:

1. Build the application/inference engine.
2. Integrate it with the existing HeteroAccel runtime.
3. Identify any genuinely missing generic runtime capability.
4. Add that capability to HeteroAccel only if it is reusable across workloads.
5. All future applications can then benefit from the same capability.

This prevents HeteroAccel from becoming permanently coupled to one AI application.

---

## What HeteroAccel Can and Cannot Do Today

### Currently Supported

* Hardware detection
* CPU execution
* Vulkan execution
* CUDA architecture/support where available
* llama.cpp integration
* GPU-layer configuration
* Adaptive scheduling
* Memory pressure management
* Resource residency abstractions
* Streaming/prefetch infrastructure
* Performance profiling
* Historical performance tracking
* Auto-tuning
* Execution policies
* Workload tracking
* Workload cancellation
* Safe fallback behavior
* Local GGUF inference
* CLI-based testing
* Developer runtime integration

### Not Currently Claimed

HeteroAccel does **not** currently claim to:

* make a multi-trillion-parameter model run on any laptop automatically
* provide arbitrary per-tensor placement inside llama.cpp
* perform arbitrary mid-generation tensor migration
* guarantee identical performance across different hardware
* replace an inference engine
* replace an operating system
* automatically make every existing AI application use HeteroAccel
* provide a complete end-user AI application by itself

These capabilities may require future integration with inference engines and additional runtime capabilities.

---

## Recommended Current Demonstration

The simplest demonstration of the current project is:

```text
1. Clone HeteroAccel
        ↓
2. Build Release
        ↓
3. Run 58 tests
        ↓
4. Provide a GGUF model
        ↓
5. Run:
   adaptive-gpu run <model.gguf> "Hello"
        ↓
6. Run interactive chat
        ↓
7. Inspect status
        ↓
8. Inspect profiling
        ↓
9. Compare CPU and Vulkan execution
```

This demonstrates the current system without requiring a future application such as Colibrì.

---

## Architecture Philosophy

HeteroAccel follows one central design principle:

> **Applications should describe workloads; the runtime should manage heterogeneous execution.**

The application should focus on:

```text
What does the user want?
```

The inference engine should focus on:

```text
How is the model executed?
```

HeteroAccel should focus on:

```text
How should the available local hardware and resources
be managed to execute that workload efficiently and safely?
```

This separation allows HeteroAccel to become a reusable infrastructure layer for multiple local AI systems rather than a component tied to a single application.

---

## Building and Testing

Requirements:

* CMake 3.20+
* C++17
* MSVC, GCC, or Clang
* Vulkan SDK for Vulkan execution/testing
* CUDA toolkit/driver environment for CUDA execution where applicable
* GGUF model for end-to-end LLM tests

Build:

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release -j 4
```

Run tests:

```bash
ctest -C Release --output-on-failure
```

Current validated result:

```text
58/58 tests passed
0 failed
0 skipped
```

---

## Project Status

**Current status: Phase 10 complete.**

HeteroAccel is currently a functional developer-stage adaptive local inference runtime with real llama.cpp integration.

The next stage is not another mandatory HeteroAccel phase.

Instead, the runtime can now be used as infrastructure for:

* local AI applications
* local RAG systems
* AI coding tools
* AI IDEs
* agent systems
* future inference engines such as Colibrì
* heterogeneous-computing research and benchmarking

HeteroAccel should only be extended when a future workload exposes a **generic execution capability that the runtime genuinely does not yet provide**.

---

