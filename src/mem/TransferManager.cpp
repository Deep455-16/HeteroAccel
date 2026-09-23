// src/mem/TransferManager.cpp
#include "mem/TransferManager.h"
#include "gpu/GPUBackend.h"

#include <chrono>
#include <cstring>
#include <iostream>

namespace agr {

using Clock = std::chrono::steady_clock;

TransferManager::TransferManager(VulkanBackend& backend)
    : backend_(backend) {}

TransferRecord TransferManager::makeRecord(MemoryLocation src, MemoryLocation dst, size_t bytes) {
    TransferRecord rec;
    rec.id          = nextId_++;
    rec.source      = src;
    rec.destination = dst;
    rec.bytes       = bytes;
    rec.start_time  = Clock::now();
    return rec;
}

void TransferManager::finishRecord(TransferRecord& rec, bool success, double durationMs) {
    rec.end_time    = Clock::now();
    rec.duration_ms = durationMs;
    rec.success     = success;
    if (durationMs > 0.0 && rec.bytes > 0) {
        // bandwidth in GB/s: (bytes / 1e9) / (ms / 1000)
        rec.bandwidth_gbps = (static_cast<double>(rec.bytes) / 1.0e9) / (durationMs / 1000.0);
    }
}

TransferHandle TransferManager::upload(const void* cpuPtr, const MemoryBlock& gpuBlock, size_t size) {
    if (!cpuPtr || !gpuBlock.isValid() || size == 0) return {};
    if (!backend_.isAvailable()) return {};

    std::lock_guard<std::mutex> lk(mutex_);
    TransferRecord rec = makeRecord(MemoryLocation::CPU, MemoryLocation::GPU, size);

    Buffer buf;
    buf.id         = gpuBlock.backend_handle;
    buf.size_bytes = size;

    double ms = 0.0;
    bool ok = backend_.upload(buf,
                               reinterpret_cast<const float*>(cpuPtr),
                               size / sizeof(float),
                               &ms);

    finishRecord(rec, ok, ms);

    if (ok) {
        bytesUploaded_ += size;
        totalUploadMs_ += ms;
    }
    records_.push_back(rec);

    if (!ok) return {};
    return { rec.id };
}

TransferHandle TransferManager::download(const MemoryBlock& gpuBlock, void* cpuPtr, size_t size) {
    if (!cpuPtr || !gpuBlock.isValid() || size == 0) return {};
    if (!backend_.isAvailable()) return {};

    std::lock_guard<std::mutex> lk(mutex_);
    TransferRecord rec = makeRecord(MemoryLocation::GPU, MemoryLocation::CPU, size);

    Buffer buf;
    buf.id         = gpuBlock.backend_handle;
    buf.size_bytes = size;

    double ms = 0.0;
    bool ok = backend_.download(buf,
                                 reinterpret_cast<float*>(cpuPtr),
                                 size / sizeof(float),
                                 &ms);

    finishRecord(rec, ok, ms);
    if (ok) {
        bytesDownloaded_ += size;
        totalDownloadMs_ += ms;
    }
    records_.push_back(rec);

    if (!ok) return {};
    return { rec.id };
}

bool TransferManager::isComplete(TransferHandle handle) const {
    if (!handle.valid()) return false;
    return true; // synchronous implementation — always done
}

const std::vector<TransferRecord>& TransferManager::records() const { return records_; }
uint64_t TransferManager::totalBytesUploaded()   const { return bytesUploaded_; }
uint64_t TransferManager::totalBytesDownloaded() const { return bytesDownloaded_; }
size_t   TransferManager::transferCount()         const { return records_.size(); }

double TransferManager::avgUploadBandwidthGbps() const {
    if (totalUploadMs_ <= 0.0 || bytesUploaded_ == 0) return 0.0;
    return (static_cast<double>(bytesUploaded_) / 1.0e9) / (totalUploadMs_ / 1000.0);
}

double TransferManager::avgDownloadBandwidthGbps() const {
    if (totalDownloadMs_ <= 0.0 || bytesDownloaded_ == 0) return 0.0;
    return (static_cast<double>(bytesDownloaded_) / 1.0e9) / (totalDownloadMs_ / 1000.0);
}

} // namespace agr
