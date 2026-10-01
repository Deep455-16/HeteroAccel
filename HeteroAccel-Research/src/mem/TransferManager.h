// src/mem/TransferManager.h
#pragma once
#include "mem/MemoryTypes.h"
#include "gpu/VulkanBackend.h"
#include "gpu/GPUBackend.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace agr {

/// Handle returned from a transfer request (currently always synchronous).
struct TransferHandle {
    uint64_t id = 0;    ///< 0 == invalid
    bool   valid() const { return id != 0; }
};

/// Manages CPU<->GPU memory transfers and records telemetry.
class TransferManager {
public:
    explicit TransferManager(VulkanBackend& backend);

    /// Upload `size` bytes from CPU ptr into the GPU buffer identified by
    /// `gpuBlock.backend_handle` (Buffer::id inside VulkanBackend).
    /// Returns a completed TransferHandle on success, invalid on failure.
    TransferHandle upload(const void* cpuPtr, const MemoryBlock& gpuBlock, size_t size);

    /// Download `size` bytes from the GPU buffer into `cpuPtr`.
    TransferHandle download(const MemoryBlock& gpuBlock, void* cpuPtr, size_t size);

    /// For Phase 5+: check if an async transfer is done (always true now).
    bool isComplete(TransferHandle handle) const;

    /// Return all recorded transfers.
    const std::vector<TransferRecord>& records() const;

    /// Aggregated upload bytes and average bandwidth (GB/s).
    uint64_t totalBytesUploaded()   const;
    uint64_t totalBytesDownloaded() const;
    double   avgUploadBandwidthGbps()   const;
    double   avgDownloadBandwidthGbps() const;
    size_t   transferCount()         const;

private:
    TransferRecord makeRecord(MemoryLocation src, MemoryLocation dst, size_t bytes);
    void           finishRecord(TransferRecord& rec, bool success, double durationMs);

    VulkanBackend& backend_;

    mutable std::mutex mutex_;
    std::vector<TransferRecord> records_;
    uint64_t nextId_ = 1;

    uint64_t bytesUploaded_   = 0;
    uint64_t bytesDownloaded_ = 0;
    double   totalUploadMs_   = 0.0;
    double   totalDownloadMs_ = 0.0;
};

} // namespace agr
