# Phase 16 — Native HeteroAccel Execution Primitives

## Overview

Phase 16 establishes the foundational low-level execution and tensor memory primitives for HeteroAccel. These reusable components are designed for direct use by heterogeneous execution graph nodes (Phase 15) and serve as the execution baseline for the future HydroXL native inference engine (Phase 17).

---

## Architecture & Components

```
                    ┌────────────────────────┐
                    │    ExecutionGraph      │ (Phase 15)
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   TensorComputeNode    │ (Phase 16)
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   IBackendExecutor     │ (Phase 16 Backend Interface)
                    └───────────┬────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         ▼                                             ▼
┌────────────────────────┐                   ┌────────────────────────┐
│   CpuBackendExecutor   │                   │ VulkanBackend / Future │
│  (Real CPU execution)  │                   │  (Vulkan buffer/copy)  │
└───────────┬────────────┘                   └────────────────────────┘
            │
            ▼
┌────────────────────────┐
│         CpuOps         │ (copy, fill, add, multiply, scale, sum, maxVal)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│     Tensor / View      │ (bounds checking, overflow guards, shape validation)
│       HostBuffer       │ (64-byte aligned RAII host allocation)
└────────────────────────┘
```

---

## 1. Tensor and Buffer Abstractions

### `HostBuffer` (`src/primitives/HostBuffer.h`)
- **Alignment:** 64-byte aligned CPU host memory allocations (`_aligned_malloc` on MSVC, `std::aligned_alloc` on POSIX) for AVX2/AVX-512 and SIMD compatibility.
- **Safety & Bounds:** `data(offset, size)` validates `offset < capacity_` and `offset + size <= capacity_`. Out-of-bounds access returns `nullptr`.
- **Zero-size Handling:** Zero-size buffers are valid (`isValid() == true`, `capacity() == 0`, `data() == nullptr`).
- **RAII Lifecycle:** Non-copyable, strictly movable. Moved-from buffers have zero capacity and null pointer.

### `TensorShape` & `DataType` (`src/primitives/TensorTypes.h`)
- **Supported Data Types:** `FLOAT32`, `FLOAT16`, `INT32`, `INT16`, `INT8`, `UINT8`, `BOOL`.
- **Element Size & Validation:** Verified element sizes with overflow detection.
- **Integer Overflow Protection:** Multiplications across dimensions in `elementCount()` and `byteSize()` guard against `uint64_t` overflow and return `UINT64_MAX` upon overflow.

### `Tensor` & `TensorView` (`src/primitives/Tensor.h`)
- **Ownership:** `Tensor` owns a `HostBuffer`. Non-copyable, movable. Moved-from tensors have their validity flag reset to `false`.
- **Safety:** Guards against allocations exceeding safe system limits (32 GiB ceiling).
- **Element Access:** Bounds-checked access via `at<T>(idx)` and typed base pointers via `dataAs<T>()`.
- **Non-owning Views:** `TensorView` provides non-owning spans with validated lifetime and bounds.

---

## 2. CPU Execution Primitives (`src/primitives/cpu/CpuOps.h`)

Implemented pure scalar CPU operations with rigorous validation:
- `copy`: Byte-wise copy between tensors of identical shapes.
- `fill`: Sets all elements of a `FLOAT32` tensor to a specified scalar value.
- `add`: Elementwise addition (`dst = a + b`) with support for memory aliasing (`dst == a` or `dst == b`).
- `multiply`: Elementwise multiplication (`dst = a * b`).
- `scale`: Elementwise scalar multiplication (`dst = src * scalar`).
- `sum`: Accumulates all elements into a double-precision scalar.
- `maxVal`: Finds the maximum element in a tensor.

All operations return an explicit `OpResult` (`success` and `error` description).

---

## 3. Backend Execution Interface (`src/primitives/IBackendExecutor.h`)

- **Contract:** Uniform interface for buffer allocation, tensor operation dispatch, capability query, and error propagation.
- **`CpuBackendExecutor`:** Production-grade implementation wrapping `CpuOps`. Fully tested and always available.
- **Hardware Truthfulness:** CUDA is not advertised on unsupported hardware. Vulkan compute remains mapped to existing Phase 2/4 infrastructure without false claims.

---

## 4. Phase 15 Execution Graph Integration (`src/execution/TensorComputeNode.h`)

- **Node Type:** `TensorComputeNode` implements `IExecutionNode` (`ExecutionNodeType::COMPUTE`, `ExecutionResource::CPU`).
- **Factory Constructors:** `makeCopy`, `makeFill`, `makeAdd`, `makeMultiply`, `makeScale`.
- **Scheduler Interoperability:** Participates in DAG dependency scheduling, topological dispatch, and respects `ExecutionGraphContext::isCancelled()`.

---

## 5. Limitations & Future Work

- **Quantized Kernels:** Native dequantization and integer dot-product execution belong to future phases (Phase 17).
- **SIMD / Vector Acceleration:** Current operations use scalar reference implementations. AVX2/AVX-512 kernels can be added as drop-in optimizations in `CpuOps.cpp`.
- **Vulkan Operator Library:** Phase 16 preserves existing Vulkan vector-add and buffer transfers; a full SPIR-V operator library will be introduced alongside HydroXL local model execution.
