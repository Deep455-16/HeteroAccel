# Phase 3 Setup Guide

To run Phase 3 locally on Windows with an Intel Iris Xe integrated GPU, follow these steps exactly.

## 1. Environment Check
Ensure you have installed:
- Visual Studio 2022 or 2026 (with Desktop C++ workload)
- CMake
- Vulkan SDK

## 2. Obtain a Model
Download a small, instruction-tuned GGUF model.
Example: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` from HuggingFace.
Save it to a permanent location, e.g., `C:\models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`.

**Do not save the model in the git repository.**

## 3. Set the Environment Variable
For tests to locate your model without hardcoding, set:
```powershell
$env:HETEROACCEL_MODEL_PATH = "C:\models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```

## 4. Build
```powershell
# 1. Fetch llama.cpp and configure
cmake -S . -B build -G "Visual Studio 18 2026" -A x64

# 2. Build Release (this will take time as llama.cpp is large)
cmake --build build --config Release
```

## 5. Verify Tests
```powershell
ctest --test-dir build -C Release --output-on-failure
```
You should see Phase 1, Phase 2, and all Phase 3 tests pass.

## 6. Run the CLI
```powershell
# CPU-only inference
.\build\Release\adaptive-gpu.exe llm --cpu --prompt "What is an integrated GPU?"

# Vulkan GPU offloaded inference
.\build\Release\adaptive-gpu.exe llm --gpu-layers 99 --prompt "What is an integrated GPU?"

# Full benchmark comparison
.\build\Release\adaptive-gpu.exe benchmark llm
```
