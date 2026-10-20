# HeteroAccel — Architecture Overview

## Project Purpose

HeteroAccel is a heterogeneous CPU/GPU inference runtime for quantized LLM models. It provides adaptive backend selection, memory management, scheduling, and a native inference engine (HydroXL) that runs Qwen2 transformer models directly from GGUF files without requiring llama.cpp at runtime.

## Subsystem Map

```
adaptive-gpu-runtime/
└── src/
    ├── core/          Shared types, JSON serialization
    ├── hardware/      Phase 1: CPU, GPU, CUDA, Vulkan device detection
    ├── gpu/           Phase 2: VulkanBackend — compute buffers, dispatch, shaders
    ├── llm/           Phase 3: LlamaCppEngine — wraps llama.cpp for CPU/Vulkan inference
    ├── mem/           Phase 4: MemoryManager — CPU/GPU pools, LRU eviction, priorities
    ├── backend/       Phase 4: BackendManager, DeviceSelector — backend enumeration
    ├── scheduler/     Phase 5/8/9: Adaptive heterogeneous scheduler, cost model,
    │                             feedback loop, auto-tuner, strategy selection
    ├── model/         Phase 6/14: LayerManager, residency tracking, GGUFRegionExtractor,
    │                              streaming model I/O
    ├── inference/     Phase 7: HeteroRuntime — orchestrates engine + memory + scheduler
    ├── profiler/      Phase 8: PerformanceHistory, AutoTuner persistence
    ├── engine/        Phase 11: IExecutionEngine interface, ExecutionEngineRegistry,
    │                            LlamaCppExecutionEngine adapter
    ├── analysis/      Phase 12: GGUFInspector (llama.cpp-based), CapabilityAnalyzer
    ├── planner/       Phase 13: ExecutionPlanner — chooses engine/backend for a request
    ├── execution/     Phase 15: ExecutionGraph — DAG of TensorComputeNodes
    ├── primitives/    Phase 16: Tensor, HostBuffer, CpuOps (native linear algebra)
    ├── hydroxl/       Phase 17-19: HydroXL native inference engine (see below)
    ├── plugin/        Phase 18-19: Dynamic engine plugin loader
    └── cli/           Command-line interface (main.cpp)
```

## HydroXL Inference Engine (`src/hydroxl/`)

HydroXL is a self-contained, native Qwen2 transformer inference engine. It reads GGUF files directly using its own parser and performs all computation on the CPU without any llama.cpp dependency at inference time.

### Files

| File | Purpose |
|------|---------|
| `HydroXLTypes.h` | GGMLType enum (correct GGML type IDs), GGUFTensorDesc, Qwen2HParams, block-size constants |
| `GGUFLoader.h/.cpp` | GGUF v2/v3 binary parser — metadata, tensor descriptors, alignment, data reads |
| `HydroXLDequant.h/.cpp` | Dequantization: Q4_K, Q3_K, Q5_0, Q5_1, Q6_K, Q8_0, F16 → float32 |
| `HydroXLOps.h/.cpp` | Transformer operators: RMSNorm, matmul, embed_lookup, RoPE, SiLU, SwiGLU, softmax, GQA attention |
| `HydroXLKVCache.h/.cpp` | Incremental key-value cache for autoregressive generation |
| `HydroXLSampler.h/.cpp` | Temperature, top-k, top-p, repetition penalty with LCG RNG |
| `HydroXLTokenizer.h/.cpp` | GPT-2 byte-level BPE tokenizer (Qwen2 compatible) |
| `HydroXLModel.h/.cpp` | GGUF weight loader — reads and dequantizes all Qwen2 layer tensors |
| `HydroXLForwardPass.h/.cpp` | Full Qwen2 forward pass: embed → N layers → norm → lm_head → logits |
| `HydroXLEngine.h/.cpp` | `IExecutionEngine` adapter; registers as `"hydroxl"` in ExecutionEngineRegistry |

### Inference Path

```
HydroXLEngine::execute(ctx)
  └── HydroXLTokenizer::encode(prompt)
  └── for each token position:
        HydroXLForwardPass::forward(tokens, pos, kv_cache)
          └── embed_lookup → token embedding
          └── for each layer:
                RMSNorm(attn_norm)
                QKV matmul + RoPE
                GQA attention (KV cache read/write)
                residual_add
                RMSNorm(ffn_norm)
                SwiGLU FFN (gate × up → SiLU → down)
                residual_add
          └── final RMSNorm
          └── lm_head matmul → logits
  └── HydroXLSampler::sample(logits, temperature, top_k, top_p)
  └── HydroXLTokenizer::decode(output_ids)
```

### Supported Models (Phase 19)

- Architecture: `qwen2` (Qwen2, Qwen2.5)
- Quantization: Q4_K_M, Q4_K_S, Q3_K_M, Q6_K (mixed), Q8_0, F16, F32
- Tokenizer: `tokenizer.ggml.model == "gpt2"` (byte-level BPE)

## Plugin System (`src/plugin/`)

The dynamic plugin system (`PluginLoader.h/.cpp`) allows native DLL plugins to provide additional inference engines. Plugins implement the C-linkage ABI in `PluginAbis.h`. Discovery, ABI verification, lifecycle management, and concurrent access are handled by the host loader.

Plugins are **not** automatically registered as `IExecutionEngine` instances in the registry. They are invoked through `plugin_execute()`. Factory integration (`pluginCreateEngine → IExecutionEngine`) is defined in the ABI but not yet wired to the execution engine registry.

## Key Interfaces

### `IExecutionEngine` (`src/engine/IExecutionEngine.h`)
```cpp
virtual EngineIdentity     identity() const = 0;
virtual EngineCapabilitySet capabilities() const = 0;
virtual bool               initialize() = 0;
virtual void               shutdown() = 0;
virtual bool               loadModel(const ModelDescriptor&) = 0;
virtual void               unloadModel() = 0;
virtual ExecutionResult    execute(const ExecutionContext&) = 0;
virtual ExecutionResult    executeStreaming(const ExecutionContext&, TokenCallback) = 0;
virtual EngineError        lastEngineError() const = 0;
```

### `GGMLType` enum (real GGML type IDs — verified against `ggml.h`)
```
F32=0  F16=1  Q4_0=2  Q4_1=3  Q5_0=6  Q5_1=7  Q8_0=8  Q8_1=9
Q2_K=10  Q3_K=11  Q4_K=12  Q5_K=13  Q6_K=14  Q8_K=15
```

## Build System

- **Generator:** Visual Studio / MSVC
- **C++ standard:** C++17
- **Dependencies:** Vulkan SDK, llama.cpp (FetchContent), GGML (bundled with llama.cpp)
- **Primary build:** `cmake --build build --config Release`
- **Tests:** `ctest --test-dir build -C Release`
- **Real-model tests:** Set `HETEROACCEL_MODEL_PATH` env var to a `.gguf` file path

## Research Archive

`HeteroAccel-Research/` contains the original research prototype (separate source tree, not compiled as part of the main build). Phase 2 verification logs exist in both `phase2_verification_logs/` (root) and `HeteroAccel-Research/phase2_verification_logs/` — they are identical copies.
