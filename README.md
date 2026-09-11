# Adaptive GPU Runtime — Phase 1-2

Phase 1: vendor-neutral hardware/Vulkan/CUDA detection.
Phase 2: a real, vendor-neutral Vulkan compute backend (vector addition),
CPU-reference-validated, with an honest CPU-vs-GPU benchmark.

## Build — Windows (target platform)

Requirements:
- Visual Studio 2022 or 2026 (MSVC), "Desktop development with C++" workload
- CMake >= 3.20 (bundled with VS, or standalone)
- Vulkan SDK installed from https://vulkan.lunarg.com (sets `VULKAN_SDK`
  env var, which `find_package(Vulkan)` picks up automatically; it also
  provides `glslangValidator`, used to compile the Phase 2 shader)

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure

.\build\Release\adaptive-gpu.exe hardware
.\build\Release\adaptive-gpu.exe hardware --json
.\build\Release\adaptive-gpu.exe benchmark vulkan
```

(No `-G` specified — CMake auto-detects your installed Visual Studio.
A ready-to-run end-to-end verification script that performs all of the
above and logs everything is at `verify_phase2.ps1`.)

CUDA needs no SDK install to be *detected* — it dynamically loads
`nvcuda.dll` only if present. CUDA Toolkit is only needed later (Phase
3+) for actual compute/inference.

The compute shader (`shaders/vector_add.comp`) is compiled to SPIR-V by
`glslangValidator` as part of the CMake build — nothing is committed as
a binary — and copied next to `adaptive-gpu.exe` automatically.

## Build — Linux (dev/CI convenience only — NOT the Phase 1/2 target)

```bash
sudo apt-get install -y cmake build-essential libvulkan-dev glslang-tools
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/adaptive-gpu hardware --json
./build/adaptive-gpu benchmark vulkan
```

On a machine with no GPU (e.g. a headless CI box), the two GPU-dependent
tests (`test_vulkan_vector_add_correctness`, `test_vulkan_buffer_transfer`)
correctly report **SKIPPED** (CTest exit code 77) rather than a fake pass.

The Linux code paths (CPU via `/proc/cpuinfo`, GPU via
`/sys/class/drm`) exist only so this repo is buildable/testable outside
Windows during development. They are intentionally simpler than the
Windows paths and are not held to the same Definition-of-Done bar.

## What "detected" means here

Every value in the output comes from a real system call — nothing is
hardcoded or fabricated. If a backend (Vulkan, CUDA, or a GPU) isn't
present, the tool says so explicitly with a reason, rather than
inventing a plausible-looking number.

## Status of verification against the target hardware

**Phase 1** was independently verified on the real target machine
(Windows, MSVC/Visual Studio, Intel i5-1235U + Iris Xe): build succeeds,
all 6 Phase 1 tests pass, Iris Xe is detected, Vulkan 1.4 loader / 1.3
device support is detected, and a compute-capable queue is reported.

**Phase 2** was implemented and built/tested in a Linux CI/sandbox
container with **no GPU device present**, so:

- Vector-add correctness (CPU reference vs GPU), buffer upload/download
  round-trip, real `vkCreateInstance`→dispatch→fence-wait execution:
  **NOT VERIFIED here** — `test_vulkan_vector_add_correctness` and
  `test_vulkan_buffer_transfer` correctly report SKIPPED, because
  `vkCreateInstance` fails with `VK_ERROR_INCOMPATIBLE_DRIVER` on this
  container (no ICD/GPU) — the same honest failure path Phase 1's
  detector already reported.
- Everything that *can* run without a GPU was run for real: the shader
  compiles from source via `glslangValidator`, the full project builds
  clean with `-Wall -Wextra` and zero warnings, the CPU reference
  implementation is unit-tested, the backend's graceful-failure path
  (`initialize()` returns false with a specific reason) is exercised for
  real, and the benchmark's CPU timings and "N/A" GPU-unavailable
  formatting are verified with a test double.
- DXGI/WinAPI code and the `<codecvt>`→`WideCharToMultiByte` fix remain
  Windows-only and unverified outside MSVC, same caveat as Phase 1.

Running `ctest` and `adaptive-gpu benchmark vulkan` on the real Iris Xe
machine is the remaining step to close out Phase 2 — that will exercise
the two currently-skipped tests for real and produce genuine CPU-vs-GPU
numbers instead of a Linux-container "N/A".
