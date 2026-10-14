# HeteroAccel

### Hardware-Adaptive AI Inference Runtime with Native Heterogeneous Execution and HydroXL

HeteroAccel is a hardware-adaptive runtime designed to make local AI inference more flexible across heterogeneous computing environments. It combines hardware capability detection, memory management, adaptive scheduling, execution planning, model residency and streaming, heterogeneous execution graphs, and native execution primitives into a unified runtime architecture.

With Phase 17, the project introduces **HydroXL**, HeteroAccel's native inference-engine subsystem, laying the groundwork for tighter integration between inference execution and hardware-aware runtime scheduling.

The project is being developed incrementally, with an emphasis on modularity, measurable execution behavior, correctness, resource awareness, and extensibility.

> **Development status:** Phases 1–17 have been implemented according to the project's development roadmap. Build success, test results, hardware support, model compatibility, and performance must be verified against the current repository and target machine before making release-level claims.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Core Objectives](#core-objectives)
- [Architecture](#architecture)
- [Implementation Status](#implementation-status)
- [Phase 11–17 Highlights](#phase-11-17-highlights)
- [HydroXL](#hydroxl)
- [Hardware and Backend Support](#hardware-and-backend-support)
- [Requirements](#requirements)
- [Build Instructions](#build-instructions)
- [Running HeteroAccel](#running-heteroaccel)
- [Testing and Validation](#testing-and-validation)
- [Design Principles](#design-principles)
- [Current Limitations](#current-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Project Overview

Local AI inference is constrained by available RAM, GPU memory, compute capability, memory bandwidth, and the characteristics of the inference engine. A runtime that assumes a single fixed hardware configuration may fail to use available resources effectively or may attempt workloads that exceed the machine's capabilities.

HeteroAccel addresses these challenges by introducing a runtime layer that evaluates the host environment, inspects model requirements, plans execution, manages model residency, and coordinates execution through a modular architecture.

The intended design separates three major responsibilities:

1. **Runtime intelligence:** Hardware detection, resource management, scheduling, profiling, and execution planning.
2. **Execution infrastructure:** Device and backend abstractions, tensors, buffers, execution graphs, and data movement.
3. **Inference engines:** Pluggable engines that execute supported model workloads, including the native HydroXL subsystem and external engine adapters.

The long-term goal is to provide an extensible runtime that applications can integrate without having to implement hardware-aware execution management independently.

## Core Objectives

- Detect relevant CPU, memory, GPU, and backend capabilities.
- Represent hardware capabilities through reusable abstractions.
- Integrate inference engines behind a common execution interface.
- Manage memory allocation and resource pressure.
- Schedule workloads according to available resources and execution policies.
- Collect performance telemetry and support profiling-based tuning.
- Inspect supported model metadata and estimate execution requirements.
- Generate execution plans before attempting model execution.
- Manage model residency, file-backed data access, streaming, prefetching, and eviction.
- Represent heterogeneous workloads using an execution graph.
- Provide native tensor, buffer, and execution primitives.
- Develop HydroXL as a native inference-engine subsystem.
- Establish a foundation for future engine plugins, SDKs, and external integrations.

These objectives describe the project's architecture and direction. They do not imply that every model, operation, accelerator, or workload is already supported.

---

## Architecture

HeteroAccel is organized into cooperating subsystems.

### 1. Hardware and Capability Detection

Responsible for identifying the host's available computing resources and the capabilities exposed by supported backends.

### 2. Memory Management and Scheduling

Coordinates resource-aware workload execution, memory availability, and scheduling decisions. The objective is to make execution decisions based on actual resource constraints rather than assuming that every model can remain fully resident.

### 3. Execution Engine Abstraction

Provides a common interface through which the runtime can communicate with inference engines. The architecture separates runtime orchestration from engine-specific implementation details.

### 4. Model Inspection and Capability Analysis

Inspects supported model formats and compares estimated model requirements with the capabilities of the current machine.

### 5. Execution Planning

Creates a structured execution plan from model requirements, runtime configuration, and available capabilities. Planning results should distinguish between detected facts, estimates, unsupported requirements, and execution constraints.

### 6. Model Residency and Streaming

Manages model data residency and file-backed access. It provides the foundation for controlled loading, region extraction, prefetching, and eviction where supported.

### 7. Heterogeneous Execution Graph

Represents execution as a graph of operations and dependencies, providing a foundation for coordinating supported work across execution components.

### 8. Native Execution Primitives

Introduces the lower-level abstractions needed for native execution, including tensor and buffer representations, supported operations, backend interfaces, and synchronization or transfer mechanisms.

### 9. HydroXL Native Inference Engine

Provides the native inference-engine subsystem introduced in Phase 17. It is intended to develop alongside HeteroAccel's execution infrastructure rather than replace the runtime's hardware-awareness and orchestration responsibilities.

---

## Implementation Status

The development roadmap through Phase 17 is as follows.

| Phase | Component | Purpose |
|---|---|---|
| 1 | Hardware Detection | Detect available hardware and system resources. |
| 2 | Backend and Device Abstraction | Represent computing devices and execution backends. |
| 3 | LLM Engine Integration | Establish the initial inference-engine integration. |
| 4 | Memory Manager | Manage runtime memory resources. |
| 5 | Adaptive Scheduler | Coordinate workloads according to resource constraints. |
| 6 | Model Streaming Infrastructure | Establish the initial infrastructure for model data access and streaming. |
| 7 | HeteroAccel Runtime | Integrate runtime execution, backend selection, and command-line workflows. |
| 8 | Profiling and Auto-Tuning | Collect execution metrics and maintain performance history for tuning decisions. |
| 9 | Execution Policies | Represent workload execution strategies and resource-pressure policies. |
| 10 | Production Hardening | Improve robustness, input validation, error handling, and lifecycle management. |
| 11 | Universal Execution Engine Interface | Introduce a common interface for supported inference engines. |
| 12 | Model Inspection and Capability Analysis | Inspect model requirements and evaluate hardware compatibility. |
| 13 | Execution Planner | Generate structured execution plans for model workloads. |
| 14 | Model Residency and Streaming | Add model residency management, region extraction, prefetching, and eviction infrastructure. |
| 15 | Heterogeneous Execution Graph | Introduce a graph-based representation for execution and dependencies. |
| 16 | Native HeteroAccel Execution Primitives | Establish native tensor, buffer, operation, and backend foundations. |
| 17 | Native HydroXL Inference Engine | Introduce HydroXL as a native inference-engine subsystem. |

---

## Phase 11–17 Highlights

### Phase 11 — Universal Execution Engine Interface

Introduced a common engine-facing abstraction intended to decouple runtime orchestration from engine-specific inference implementations.

The reported components include:

- `IExecutionEngine`
- `ExecutionContext`
- `ExecutionResult`
- `EngineCapability`
- `EngineIdentity`
- `ExecutionEngineRegistry`
- `LlamaCppExecutionEngine`

The runtime can use the common interface to interact with registered engines while keeping engine-specific execution behavior behind the appropriate implementation.

### Phase 12 — Model Inspection and Capability Analysis

Introduced model-inspection and capability-analysis components for evaluating model requirements against the current system.

The reported components include:

- `ModelRequirements`
- `IModelInspector`
- `GGUFInspector`
- `CapabilityAnalyzer`
- `CapabilityReport`

The command-line workflow includes model analysis through:

```powershell
adaptive-gpu analyze <model.gguf>
```

The purpose is to provide information about a model's requirements and the detected environment before attempting execution. Analysis output should not be treated as proof that every operation or the entire model can execute successfully on a particular backend.

### Phase 13 — Execution Planner

Introduced a planning layer that produces a structured representation of the intended model execution strategy.

The reported components include:

- `ModelExecutionPlan`
- `ExecutionPlanner`
- `PlannerConfig`
- `PlanReason`

The command-line workflow includes:

```powershell
adaptive-gpu plan <model.gguf>
```

The planner provides a foundation for explaining execution decisions and identifying constraints before loading or executing a model.

### Phase 14 — Model Residency and Streaming

Introduced infrastructure for managing model data residency and controlled access to model regions.

The reported components include:

- `ModelResidencyManager`
- `GGUFRegionExtractor`
- `PlanResidencyAdapter`
- File-backed model data access
- Residency tracking
- Transfer and prefetch infrastructure
- LRU-style eviction
- Telemetry and cancellation support

This phase establishes infrastructure for managing model data under constrained memory conditions.

Streaming model data does not eliminate the compute, memory-bandwidth, latency, or working-memory requirements of inference. Actual execution behavior depends on the model architecture, supported operations, implementation, hardware, and workload.

### Phase 15 — Heterogeneous Execution Graph

Introduced the graph-based execution foundation for representing operations and their dependencies.

The execution graph is intended to support more structured coordination between runtime components and execution backends. Its practical capabilities depend on the implemented graph operations, dependency handling, backend integrations, and validation tests.

### Phase 16 — Native HeteroAccel Execution Primitives

Introduced the native execution foundations needed to reduce dependence on engine-specific abstractions at the runtime level.

This phase establishes the direction for:

- Tensor and buffer representations.
- Native operation execution.
- Backend abstractions.
- Data transfer and synchronization mechanisms.
- Integration with the heterogeneous execution graph.

Actual supported operations and accelerator execution paths must be verified against the implementation and tests. A declared backend or abstraction does not by itself establish that every operation executes natively on that device.

### Phase 17 — Native HydroXL Inference Engine

Introduced HydroXL as a native inference-engine subsystem within HeteroAccel.

The objective is to establish an engine that can progressively use HeteroAccel's native execution infrastructure, rather than relying exclusively on external inference engines.

The Phase 17 development history includes the HydroXL inference-engine implementation and subsequent dequantization-related changes.

HydroXL should be evaluated based on its implemented model support, tensor operations, quantization and dequantization correctness, execution paths, numerical accuracy, memory behavior, and reproducible performance measurements.

Phase 17 establishes a foundation for continued development. It should not be interpreted as proof of complete compatibility with all GGUF models, all quantization formats, every transformer architecture, or every available accelerator.

---

## HydroXL

HydroXL is the native inference-engine subsystem being developed alongside the HeteroAccel runtime.

The intended division of responsibility is:

- **HeteroAccel:** Hardware awareness, capability analysis, execution planning, resource management, residency, scheduling, and orchestration.
- **HydroXL:** Native inference execution and the engine-specific operations needed to run supported models.
- **External engines:** Optional integrations that can provide alternative execution implementations where useful.

Keeping these responsibilities separate allows the runtime to evolve independently from the inference engine.

### HydroXL Development Priorities

Future development should prioritize:

1. Correct tensor layouts and numerical operations.
2. Correct dequantization and quantized-weight handling for explicitly supported formats.
3. Reliable execution of supported model architectures.
4. Memory-efficient weight and activation handling.
5. Integration with HeteroAccel's execution planning and residency infrastructure.
6. Profiling of actual execution time, memory use, and data movement.
7. Reference-based correctness tests and reproducible benchmarks.
8. Incremental support for additional operators, model architectures, and backends.

HydroXL's supported capabilities should be documented explicitly as implementation and testing progress.

---

## Hardware and Backend Support

HeteroAccel is designed around hardware-aware execution, but hardware detection, backend availability, and successful inference are separate concerns.

A backend can be detected without supporting every operation required by a particular model. Likewise, a device's reported shared memory is not equivalent to dedicated high-bandwidth VRAM.

### Development Hardware

The development environment has included the following configuration:

- **CPU:** Intel Core i5-1235U
- **System memory:** Approximately 16 GB
- **Integrated graphics:** Intel Iris Xe
- **Dedicated GPU memory:** Approximately 128 MB reported
- **Shared graphics memory:** Approximately 8 GB reported
- **Vulkan:** Available in the reported development environment
- **CUDA:** Not available on the reported Intel-only machine

These details describe the development machine, not a universal minimum or maximum hardware requirement.

### Backend Verification

Before describing a backend as supported for a workload, verify:

- The backend is detected correctly.
- The relevant device and execution capabilities are available.
- The required operations are implemented.
- The model can execute correctly.
- Results are numerically validated.
- Performance and memory use are measured.

CUDA support requires a compatible NVIDIA GPU, drivers, toolkit or runtime dependencies, and a working CUDA backend implementation. It cannot be demonstrated on the reported Intel-only development machine.

Vulkan availability alone does not guarantee that the entire inference workload executes on the GPU.

---

## Requirements

The exact dependencies depend on the current source tree and build configuration. The development environment has included:

- Windows 10/11
- CMake
- Visual Studio Build Tools and a compatible MSVC compiler
- Git
- Vulkan SDK for Vulkan-enabled builds
- The dependencies required by the selected inference engine and backend

A GGUF model is needed for model-analysis and inference workflows that use GGUF.

Example model path used during development:

```text
C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
```

This is an example path, not a model bundled with the repository.

---

## Build Instructions

The following is a general Windows build workflow. Verify the actual CMake targets and configuration options in the current repository before running it.

### 1. Clone the repository

```powershell
git clone https://github.com/Deep455-16/HeteroAccel.git
cd HeteroAccel
```

If the repository is already present locally:

```powershell
cd C:\Users\rav62\Downloads\adaptive-gpu-runtime
git status
git pull
```

Review local changes before pulling or switching branches.

### 2. Configure the project

```powershell
cmake -S . -B build
```

Add backend-specific configuration flags only when they are defined by the project's current CMake configuration.

### 3. Build

```powershell
cmake --build build --config Release
```

### 4. Run tests

```powershell
ctest --test-dir build -C Release --output-on-failure
```

If the build uses a different configuration, generator, or test directory, adjust the commands accordingly.

---

## Running HeteroAccel

The following examples illustrate the intended CLI workflows. Confirm the available commands using the current executable's help output.

### View available commands

```powershell
adaptive-gpu --help
```

If the executable is not on `PATH`, run it from its actual build output directory.

### Analyze a model

```powershell
adaptive-gpu analyze "C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```

This workflow is intended to inspect the model and report relevant requirements and capability information.

### Generate an execution plan

```powershell
adaptive-gpu plan "C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```

This workflow is intended to generate an execution plan based on the model and the detected environment.

### Run inference

The runtime has included an inference workflow in the form:

```powershell
adaptive-gpu run
```

The exact arguments and model configuration depend on the current CLI implementation. Consult `adaptive-gpu --help` for the supported syntax.

### Chat

The runtime has also included a chat workflow in the form:

```powershell
adaptive-gpu chat
```

Use the model-selection and runtime options exposed by the current CLI.

### Configure the model path

The development workflow has used the following environment variable:

```powershell
$env:HETEROACCEL_MODEL_PATH = "C:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```

This setting is relevant only to workflows that read this environment variable.

---

## Testing and Validation

Testing is essential because runtime abstractions, planning decisions, and backend detection do not automatically guarantee correct inference.

Validation should cover the following areas.

### Unit Tests

- Resource and memory-management behavior.
- Scheduling and execution policies.
- Model metadata inspection.
- Capability-analysis decisions.
- Execution-plan construction.
- Residency and eviction behavior.
- Tensor and buffer operations.
- HydroXL numerical operations.
- Error handling and cancellation.

### Integration Tests

- Engine registration and lifecycle.
- Runtime-to-engine communication.
- Planner-to-residency integration.
- Execution-graph integration.
- Model loading and supported inference workflows.
- Backend selection and fallback behavior.

### Numerical Correctness

For supported operations and models, compare outputs with a trusted reference implementation where possible. Include edge cases, supported quantization formats, boundary values, and tolerances appropriate to the operation.

### Performance Validation

Record reproducible measurements, including:

- Time to first token (TTFT).
- Tokens generated per second.
- Model loading time.
- Peak memory usage.
- CPU and GPU utilization where measurable.
- Data-transfer and streaming overhead.
- Performance under memory pressure.

Performance claims should include the hardware, model, quantization, backend, prompt and generation settings, and benchmark procedure.

### Test Reporting

Always obtain the current result from the repository rather than reusing a historical test count.

```powershell
ctest --test-dir build -C Release --output-on-failure
```

Document skipped tests separately from passing tests and explain any missing model files, optional dependencies, or unavailable hardware that prevent execution.

---

## Design Principles

### Modularity

Runtime orchestration, execution primitives, and inference engines should have distinct responsibilities.

### Capability-Driven Execution

Execution decisions should be based on detected capabilities and verified support, not assumptions about a device or backend.

### Explicit Failure Handling

Unsupported operations, resource exhaustion, invalid inputs, and backend failures should produce clear errors or documented fallback behavior.

### Measurable Performance

Optimization decisions should be informed by reproducible measurements rather than unverified speedup claims.

### Incremental Compatibility

Support for new model architectures, quantization formats, and hardware backends should be added and tested incrementally.

### Honest Capability Reporting

The project should clearly distinguish implemented functionality, tested functionality, experimental features, and planned capabilities.

---

## Current Limitations

HeteroAccel is an actively developed runtime. The following limitations should be considered when evaluating the project:

- Model compatibility depends on the implemented inspection and execution paths.
- Model analysis and execution planning are not guarantees of successful inference.
- Streaming reduces the need to keep all model data resident, but it does not remove compute or memory-bandwidth constraints.
- Vulkan availability does not establish full GPU inference support.
- CUDA execution requires compatible NVIDIA hardware and a functioning CUDA backend.
- HydroXL's supported architectures, quantization formats, operators, and inference workflows must be verified against the current implementation.
- External engine integrations require adapters and compatibility testing.
- Dynamic plugin loading, a stable public SDK, and production packaging are separate deliverables and should not be assumed complete merely because an engine registry exists.

The project should be described according to verified implementation and test results rather than its long-term goals.

---

## Roadmap

The roadmap below identifies planned work beyond Phase 17. These are future development objectives, not claims of completed functionality.

### Phase 18 — Dynamic Engine Plugin System and SDK

Planned work:

- Define a versioned engine-plugin interface or ABI.
- Specify engine capability declarations.
- Implement plugin discovery, loading, initialization, and shutdown.
- Validate plugin and runtime compatibility.
- Handle plugin failures and lifecycle errors safely.
- Provide a minimal example plugin and integration tests.
- Document the SDK and compatibility expectations.

A registry of engine implementations is not, by itself, a complete dynamic plugin-loading system.

### Phase 19 — External Engine Integrations

Planned work:

- Evaluate external inference engines for compatibility with HeteroAccel.
- Implement adapters where appropriate.
- Keep optional dependencies separate from the runtime core.
- Define capability mappings and execution-context conversion.
- Validate model support, resource behavior, correctness, and performance.
- Document limitations and integration requirements for each engine.

Potential candidates include Colibrì and AirLLM, subject to compatibility and feasibility assessment. Neither should be represented as integrated until a working adapter has been implemented and tested.

### Phase 20 — HydroXL Maturation and Optimization

Planned work:

- Expand verified model and operator support.
- Improve numerical correctness and quantization handling.
- Optimize memory allocation and tensor operations.
- Improve execution-graph integration.
- Evaluate supported CPU and accelerator execution paths.
- Establish repeatable performance benchmarks.
- Document supported model architectures and quantization formats.

### Beyond Phase 20

Potential follow-on work includes:

- A stable public application-facing API.
- Packaging, installation, and release automation.
- Reproducible benchmarking across hardware configurations.
- Broader backend coverage.
- Developer documentation and example applications.
- Compatibility guarantees and release testing.

These items should receive explicit scope, implementation milestones, and acceptance criteria before being described as completed phases.

---

## Contributing

Contributions should focus on correctness, clear interfaces, reproducible tests, and well-documented behavior.

Before submitting changes:

1. Build the project using the documented configuration.
2. Run the relevant unit and integration tests.
3. Add regression tests for bug fixes.
4. Document new commands, configuration options, and supported capabilities.
5. Avoid claiming support for untested models, operators, or backends.
6. Include benchmark methodology when reporting performance improvements.

For larger changes, first document the intended architecture and acceptance criteria, then implement and validate the change incrementally.

---

## Project Links

- **GitHub Repository:** https://github.com/Deep455-16/HeteroAccel
- **Current Development Scope:** Phases 1–17
- **Native Inference Subsystem:** HydroXL
- **Next Planned Milestone:** Phase 18 — Dynamic Engine Plugin System and SDK

---

## Closing Note

HeteroAccel aims to provide a modular, hardware-aware foundation for local AI inference. Through Phase 17, its development has progressed from hardware detection and runtime scheduling to execution planning, model residency, heterogeneous execution infrastructure, native execution primitives, and the introduction of HydroXL.

The next priority is to strengthen correctness, verify supported workloads, establish reproducible benchmarks, and expand the runtime through clearly defined interfaces. The project's maturity should ultimately be demonstrated through working implementations, repeatable tests, and measured results on real hardware.
