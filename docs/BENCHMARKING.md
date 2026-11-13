# Benchmarking & Performance (Phase 29)

`heteroaccel-bench` measures **complete end-to-end inference** through the same
`IInferenceEngine` implementations the SDK uses. Nothing is synthetic: every
figure is wall-clock timing of real generation on the host, and metrics that
cannot be measured portably are written as `null` with a note — never invented.

## Reproduce

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target heteroaccel-bench
# Windows: build\Release\heteroaccel-bench.exe   Linux/macOS: build/heteroaccel-bench
heteroaccel-bench --model /path/model.gguf --out report.json
```

| Option | Default | Meaning |
|---|---|---|
| `--backends` | `hydroxl_cpu,llama_cpu,hydroxl_vulkan,llama_vulkan,cuda` | what to run |
| `--prompt` / `--gen` | 32 / 32 | prompt tokens / generated tokens |
| `--runs` / `--warmup` | 3 / 1 | medians are taken over `runs` |
| `--ctx` | `16,64,256` | context-length scaling points |
| `--concurrency` | 4 | client threads sharing one engine |
| `--baseline F --tolerance 0.25` | – | exit code 3 if slower than baseline by more than the tolerance |

Each backend runs in its **own child process**, so peak-RSS is not contaminated
by earlier backends (`--in-process` disables this). All backends receive the
same deterministic prompt token ids, greedy decoding (`temperature = 0`), and
the same model file, so HydroXL vs llama.cpp comparisons use identical inputs.

### Backend ids

| id | engine | device |
|---|---|---|
| `hydroxl_cpu` | HydroXL native | host CPU |
| `llama_cpu` | llama.cpp | host CPU |
| `hydroxl_vulkan` | HydroXL native | physical Vulkan GPU |
| `llama_vulkan` | llama.cpp | physical Vulkan GPU |
| `cuda` | llama.cpp GGML_CUDA | NVIDIA GPU (needs `-DHETEROACCEL_ENABLE_CUDA=ON`) |

Backends that cannot run are reported honestly: `skipped` (no *physical* Vulkan
GPU — software rasterizers such as llvmpipe are never benchmarked as GPUs) or
`pending_hardware` (CUDA not compiled in / no NVIDIA device).

### Metric definitions

* **load_ms** – `initialize()` wall time.
* **ttft_ms** – wall time of a 1-token generation for the prompt (prefill + first sample).
* **decode_tps** – `(n-1) / (t_n − t_1)`; **e2e_tps** – `n / t_n`.
* **peak_rss_mb** – process peak working set. **peak_vram_mb** – only where the engine exposes device accounting (llama.cpp Vulkan/CUDA); `null` otherwise.
* **cpu_utilization_pct** – process CPU time / wall / logical cores during the timed runs. **gpu_utilization_pct** – *not measured* (no portable vendor-neutral API).
* **host_triad_bandwidth_gbs** – measured STREAM-triad bandwidth of the host.
* **effective_weight_bandwidth_gbs_lower_bound** – `model_file_bytes × decode_tps`; a lower bound on memory traffic, not a hardware counter.
* **context_scaling** – TTFT and decode rate per prompt length. Decode rates here come from subtracting two separate runs and are **noisy** (outliers such as 35.9 tok/s in the HydroXL-CPU 64-token point are timing artefacts; `null` means the subtraction was non-positive). Use TTFT for scaling conclusions.
* **concurrent** – N threads issuing requests to one shared engine; shows serialisation, not parallel speed-up.

## Regression gate

`test_p29_perf_regression` checks harness correctness (comparator, serialisation,
JSON, host counters) and runs real engines when `HETEROACCEL_MODEL_PATH` is set.
To gate against a stored baseline set:

```
HETEROACCEL_PERF_BASELINE=report.json   # a report from heteroaccel-bench
HETEROACCEL_PERF_TOLERANCE=0.30         # allowed fractional slowdown (default 0.30)
```

`*.decode_tps` / `*.e2e_tps` must not fall, and `*.ttft_ms` / `*.load_ms` must not
rise, by more than the tolerance. Baselines are hardware-specific; do not reuse
one across machines.

## Measured results

Hardware: Intel Core i5-1235U (10 cores / 12 threads), 16 GB RAM, Intel Iris Xe
(physical Vulkan, shared memory), Windows, MSVC Release, **no NVIDIA GPU**.
Model: Qwen2.5-0.5B-Instruct Q4_K_M (491 MB). Reports:
[before](benchmarks/p29_before_i5-1235U_IrisXe.json),
[after](benchmarks/p29_after_i5-1235U_IrisXe.json). Single machine, 3 runs,
laptop CPU — expect ±10 % run-to-run variation.

| Backend | load ms | TTFT ms | decode tok/s | peak RSS MB | peak VRAM MB |
|---|---:|---:|---:|---:|---:|
| llama.cpp CPU | 444 | 330 | 61.0 | 532 | – |
| **HydroXL CPU** | 1420 | 4881 | 5.7 | 2567 | – |
| llama.cpp Vulkan (Iris Xe) | 843 | 123 | 48.9 | 1259 | 1043 |
| **HydroXL Vulkan** (Iris Xe) | 3005 | 5805 | 5.8 | 4433 | n/a |
| CUDA | – | – | – | – | **pending: no NVIDIA hardware** |

HydroXL / llama.cpp decode ratio: **0.09× on CPU, 0.12× on Vulkan.**

### Correctness against the reference
Greedy generation with identical prompts matches llama.cpp **exactly**
(12/12 tokens on "The capital of France is", 32/32 on the synthetic prompt) on
both CPU and Vulkan. Teacher-forced logits on the real model:
cosine ≥ 0.9991, identical top-1 (`test_p29_real_model_equiv`). Residual logit
differences (max |Δ| ≈ 0.5) come from llama.cpp multiplying Q4 weights with
int8-quantised activations while HydroXL uses f32.

## What was found and changed

1. **Single-threaded matvec** – HydroXL CPU used ~1 of 12 threads (8 % CPU). Added a row-parallel
   pool (`HETEROACCEL_THREADS` overrides; capped at 8; callers never block each other).
   Rows keep the scalar accumulation order, so results are bit-identical.
   Measured: 4.3 → 5.7–6.0 tok/s, TTFT 7.3 s → 4.9–5.5 s, CPU use 8 % → 38 %.
2. **Greedy sampling applied a repetition penalty** (default 1.1 over *prompt* tokens),
   so greedy output diverged from every reference engine for identical settings
   (`is` was penalised → `was`). Greedy is now a pure argmax. This was a semantic
   bug found by the equivalence work, not a speed change.

## Where the remaining gap is (not yet fixed)

* **HydroXL CPU is memory-bandwidth bound.** It dequantizes every weight to f32 at load
  (2.5 GB resident), so each token streams ≈2 GB; llama.cpp streams ≈0.4 GB of Q4/Q8
  blocks directly. With ~12–20 GB/s of host bandwidth the f32 layout caps decode in the
  single-digit tok/s range, which is what is measured. More threads cannot fix this.
  The real fix is a quantized (block-wise) matvec, which is a larger change left open.
* **Prefill is token-by-token** (no batching), which is why TTFT scales linearly with
  prompt length (≈ 45 s for 256 tokens).
* **HydroXL Vulkan** gains almost nothing over CPU: activations round-trip to the host per
  op and only some ops run on the GPU; it keeps both an f32 copy of the weights in host
  memory (4.4 GB peak RSS) and a device copy. GPU-resident activations and fused
  kernels are required to approach llama.cpp's 49 tok/s. Kernel-launch overhead and
  host/device transfer cost were identified as the likely dominant terms but were
  **not profiled with a GPU profiler**, so no per-kernel figures are claimed.
* GPU utilisation and per-kernel timings are not collected.
