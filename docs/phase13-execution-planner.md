# Phase 13: Execution Planner

Phase 13 decides **how** a workload should run. It does not run inference, and it does not stream tensors.

Phase 12 answers whether a model is feasible on the detected hardware. Phase 13 consumes that answer and produces a `ModelExecutionPlan`.

The Phase 5 scheduler already uses the name `ExecutionPlan` for a cost-model result on one submitted job. Phase 13 does not overload that type.

```text
Model / GGUF
    → GGUFInspector / ModelRequirements
    → CapabilityAnalyzer / CapabilityReport
    → ExecutionPlanner
    → ModelExecutionPlan
    → IExecutionEngine
```

## What the planner reads

The planner does not open the GGUF, probe Vulkan, or detect CUDA. Its inputs are:

- `ModelRequirements` from the inspector
- `CapabilityReport` from the analyzer
- registered `IExecutionEngine` instances and their `EngineCapabilitySet`
- `PlannerConfig` policy switches

Engine selection is capability-driven. The planner asks which registered engine advertises the backend required by the chosen placement. It does not special-case a model name, a GPU, or a RAM size. Registry iteration order does not matter: ties break by engine key so the same inputs always produce the same plan.

`llama.cpp` is the engine shipped today. A future engine is selected the same way once it is registered and the capability report marks it supported. The planner does not embed Colibrì, a native HeteroAccel engine, or a plugin SDK.

## How memory changes the plan

Residency reuses the Phase 9 `ExecutionStrategy` names.

| Situation | Plan | Executable now |
|---|---|---|
| Model fits in reported CUDA or Vulkan memory, and an engine advertises that backend | `FULL_RESIDENT` on that accelerator | yes |
| Model exceeds accelerator memory, but at least 15% of it fits | `PARTIAL_RESIDENT` (`HYBRID`) with a GPU layer budget | yes, via existing layer offload |
| Partial residency does not fit, and the capability report says streaming may eventually work | `STREAMING` | no |
| No accelerator placement, and the model fits in reported RAM | `CPU_FALLBACK` | yes |
| Nothing fits, or no capable engine is registered | unsupported plan with reason codes | no |

CUDA is selected only when the capability report lists a usable CUDA device. An engine may advertise `BACKEND_CUDA` without CUDA being chosen.

Unknown or zero model memory does not become a resident accelerator plan. The planner returns an explicit unknown-memory reason instead.

## What Phase 13 does not do

`STREAMING` in a plan means "this is the strategy a later phase should implement." The explanation states that streaming execution needs Phase 14. Phase 13 does not move tensors from SSD to GPU, place individual tensors, or run a native kernel.

Partial residency is different: it is a layer budget the existing engine can already apply through `n_gpu_layers`. The planner records that budget. It does not load the model.

## CLI

```text
adaptive-gpu analyze <model.gguf>   # Phase 12 feasibility
adaptive-gpu plan <model.gguf>      # Phase 13 placement decision
```

`HeteroRuntime::planExecution` inspects the model, analyzes capabilities, and returns the plan. It does not generate tokens.

## Reason codes

Plans carry `PlanReason` values plus a human-readable explanation. Callers can see why Vulkan was chosen, why an engine was rejected, why memory forced partial residency or streaming, and why the result is CPU fallback or unsupported.
