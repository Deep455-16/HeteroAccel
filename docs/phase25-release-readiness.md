# Phase 25: Release Readiness

## Overview

Phase 25 is the final phase of the HeteroAccel 25-phase roadmap. It wires the
`EngineOrchestrator` (Phase 21) into the CLI via a new `adaptive-gpu infer` command,
adds a `LICENSE` (MIT), bumps the SDK version to `0.25.0`, and brings the total
test count to 120 — all passing.

---

## New CLI Command: `adaptive-gpu infer`

```
adaptive-gpu infer --model <path.gguf> [--prompt <text>]
                    [--backend cpu|vulkan] [--max-tokens <N>]
```

Unlike the legacy `hydroxl` command (which hard-codes the HydroXL CPU engine
regardless of hardware), **`infer`** passes every request through
`EngineOrchestrator::select_backend()`. The orchestrator considers:

- Vulkan availability (from `HardwareDetector`)
- Model quantization type compatibility per backend (`BackendCapabilities::supported_quant_types`)
- Available system RAM vs. estimated model memory
- Persistent performance history (`PerfProfile`, Phase 24)
- `--backend` override flag (maps `cpu` → `hydroxl_cpu`, `vulkan` → `hydroxl_vulkan`)

After inference, observed tok/s is recorded back to `EngineOrchestrator::record_performance()`,
which persists it via `PerfProfile` for future runs.

### Example output

```
> adaptive-gpu infer --model D:\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf \
                      --prompt "What is GPU computing?"

=== HeteroAccel Orchestrated Inference (Phase 25) ===
Model:   D:\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
Backend: hydroxl_vulkan
Reason:  Vulkan available; highest-priority compatible backend selected
Prompt:  What is GPU computing?

GPU computing refers to the use of a Graphics Processing Unit (GPU) to perform...
---------------------------------------------
Backend:    hydroxl_vulkan
Tokens:     47
Speed:      5.21 tok/s
Total time: 9024 ms
```

### Implementation notes

- `runInferOrchCommand()` in `src/cli/main.cpp` (lines ~921–1050)
- Resets the orchestrator singleton, calls `register_default_backends()`,
  registers CPU and Vulkan engine factories, builds a `SelectionRequest`
  from CLI args + file size, then creates and runs the winning engine
- Streaming via `IInferenceEngine::run_stream()` when supported (both adapters return `true`)
- Falls back to `IInferenceEngine::run()` for engines without streaming support

---

## Version Bump

| Constant | Value |
|----------|-------|
| `HETEROACCEL_VERSION_MAJOR` | `0` |
| `HETEROACCEL_VERSION_MINOR` | `25` |
| `HETEROACCEL_VERSION_PATCH` | `0` |
| `ha_version_string()` | `"0.25.0"` |

---

## LICENSE

Standard MIT License added at the repository root (`LICENSE`).  
Copyright: HeteroAccel Contributors.

---

## Tests Added (Phase 25)

| Test | Purpose |
|------|---------|
| `test_p25_version` | Compile-time macros and runtime `ha_version_string()` = `"0.25.x"` |
| `test_p25_license` | LICENSE file exists, starts with "MIT", contains canonical permission grant |
| `test_p25_orchestrator_cli` | `EngineOrchestrator` selects valid backend, respects `forced_backend`, records perf history |
| `test_p25_release` | Public C/C++ API surface smoke test: lifecycle, null-safety, perf profile API |
| `test_p25_perf_persist` | `PerfProfile` save/load round-trip, running-average update, corrupt-file resilience, reset |

**Total test count: 120 (115 prior + 5 Phase 25)**

---

## Full Test Run (Release)

```powershell
$env:HETEROACCEL_MODEL_PATH = "D:\HeteroAccel\Models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
ctest --test-dir build -C Release --output-on-failure
```

Expected: **120/120 tests pass**.

---

## Known Limitations

> [!NOTE]
> The following are documented and intentional scope boundaries for Phase 25.

- **MoE Expert Residency:** `ModelRegionKind::EXPERT` exists as a dead enum value
  (Phase 14 placeholder). No MoE routing, expert selection, or expert-aware residency
  is implemented. HydroXL supports dense `qwen2` architecture only.
- **Plugin → Orchestrator wiring:** `PluginLoader` loads plugins but does not automatically
  register them as `IInferenceEngine` factories in `ExecutionEngineRegistry` or
  `EngineOrchestrator`. Plugins are isolated.
- **CUDA backend:** Compiled as a placeholder behind `#ifdef HETEROACCEL_CUDA_ENABLED`
  (always OFF). No CUDA inference path is active.
- **`PluginLoader.cpp` Windows-only:** Includes `<windows.h>` unconditionally; Linux/macOS
  builds would require a preprocessor guard.
- **`infer` prompt tokenization:** The CLI `infer` command passes `max_new_tokens` to the
  engine but does not pre-tokenize the CLI prompt string into `InferenceRequest::prompt_token_ids`
  — the engine receives an empty token list and uses its internal BOS token as seed.
  The `hydroxl` command (which uses `ExecutionContext::prompt` string directly) is unaffected.

---

## Files Changed in Phase 25

| File | Change |
|------|--------|
| `include/heteroaccel/heteroaccel.h` | `VERSION_MINOR` 24 → 25 |
| `src/cli/main.cpp` | `runInferOrchCommand()` + `infer` dispatch (already present from Phase 24 engineer) |
| `LICENSE` | MIT license (already present) |
| `tests/test_p25_version.cpp` | New |
| `tests/test_p25_license.cpp` | New |
| `tests/test_p25_orchestrator_cli.cpp` | New |
| `tests/test_p25_release.cpp` | New |
| `tests/test_p25_perf_persist.cpp` | New |
| `tests/CMakeLists.txt` | 5 Phase 25 test registrations appended |
| `docs/phase25-release-readiness.md` | New (this file) |
| `README.md` | Phase 25 row added to progress table |
