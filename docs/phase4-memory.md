# Phase 4: Unified Memory Management

## Overview

Phase 4 adds the `agr_mem` library — a hardware-portable memory management layer that sits between application code and the raw Vulkan/CPU allocators.

## Architecture

```
                 MemoryManager
                /             \
        CPUAllocator    VulkanAllocator
              |                |
          std::malloc    VulkanBackend::createBuffer
              |
          MemoryPool (4 MB slab cache, CPU + GPU)
                 \         /
              TransferManager  (CPU<->GPU via VulkanBackend::upload/download)
                     |
              ResidencyManager (tracks per-block location state)
              PressureMonitor  (utilisation-based thresholds)
              EvictionPolicy   (LRU + priority scoring)
```

## Key Types

| Type | Purpose |
|---|---|
| `MemoryBlock` | Opaque allocation handle (id, size, location, priority) |
| `MemoryLocation` | `CPU`, `GPU`, `DISK` |
| `MemoryPriority` | `CRITICAL`, `HIGH`, `NORMAL`, `LOW` |
| `Residency` | `CPU`, `GPU`, `CPU_AND_GPU`, `DISK` |
| `PressureLevel` | `NORMAL`, `WARNING`, `HIGH`, `CRITICAL` |
| `MemoryStats` | Aggregated runtime statistics |

## Files Added

```
src/mem/
  MemoryTypes.h        - Shared enums and structs
  IMemoryAllocator.h   - Abstract allocator interface
  CPUAllocator.{h,cpp} - malloc-based CPU allocator
  VulkanAllocator.{h,cpp} - GPU allocator via VulkanBackend::createBuffer
  TransferManager.{h,cpp} - CPU<->GPU data movement + telemetry
  MemoryPool.{h,cpp}   - Free-list pool (reuse of fixed-size blocks)
  ResidencyManager.{h,cpp} - Tracks block location state
  PressureMonitor.{h,cpp}  - Memory pressure thresholds
  EvictionPolicy.{h,cpp}   - LRU + priority-based eviction scoring
  MemoryManager.{h,cpp}    - Unified facade: allocate/release/move
```

## Usage

```cpp
agr::VulkanBackend backend;
backend.initialize(); // best-effort

agr::MemoryManager mm(backend);

// Allocate
auto cpuBlock = mm.allocate(64 * 1024 * 1024, agr::MemoryLocation::CPU, agr::MemoryPriority::NORMAL);
auto gpuBlock = mm.allocate(64 * 1024 * 1024, agr::MemoryLocation::GPU);

// Move CPU -> GPU
mm.move(cpuBlock, agr::MemoryLocation::GPU);

// Statistics
agr::MemoryStats stats = mm.statistics();
std::cout << "Upload BW: " << stats.upload_bandwidth_gbps << " GB/s\n";

// Release
mm.release(cpuBlock);
mm.release(gpuBlock);
```

## CLI

```
adaptive-gpu memory     - Show memory manager stats + live transfer telemetry
```

## Tests Added (9)

| Test | Covers |
|---|---|
| `test_mem_cpu_alloc` | CPUAllocator allocate/free/tracking |
| `test_mem_vulkan_alloc` | VulkanAllocator via VulkanBackend |
| `test_mem_cpu_to_gpu_transfer` | Upload (CPU->GPU) + telemetry |
| `test_mem_gpu_to_cpu_transfer` | Download (GPU->CPU) + data integrity |
| `test_mem_statistics` | MemoryManager statistics |
| `test_mem_pool_reuse` | MemoryPool free-list reuse counter |
| `test_mem_lru_eviction` | EvictionPolicy selects oldest block |
| `test_mem_priority_eviction` | EvictionPolicy respects priority |
| `test_mem_invalid_ops` | Double-free, zero-size safety |

## Hardware Notes

- Designed for Intel Iris Xe (unified shared memory).
- `VulkanAllocator` queries heap sizes via `vkGetPhysicalDeviceMemoryProperties`.
- `VulkanAllocator` never creates a second `VkInstance`; it reuses the shared `VulkanBackend`.
- `PressureMonitor` defaults: WARNING at 70%, HIGH at 85%, CRITICAL at 95%.
