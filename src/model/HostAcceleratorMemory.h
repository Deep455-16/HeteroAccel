#pragma once

#include "model/IAcceleratorMemory.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agr {

/// Host-RAM stand-in used when no device allocator is attached, and for tests.
class HostAcceleratorMemory : public IAcceleratorMemory {
public:
    explicit HostAcceleratorMemory(uint64_t capacity_bytes);

    std::string name() const override { return "HostAcceleratorMemory"; }
    bool isAvailable() const override { return true; }
    uint64_t capacity() const override;
    uint64_t available() const override;
    std::string lastError() const override;

    void* allocate(size_t bytes) override;
    void release(void* ptr) override;
    bool upload(const void* source, void* destination, size_t bytes) override;
    bool download(void* destination, const void* source, size_t bytes) override;

    void setFailNextUpload(bool fail) { fail_next_upload_ = fail; }

private:
    uint64_t capacity_;
    uint64_t used_ = 0;
    bool fail_next_upload_ = false;
    mutable std::mutex mutex_;
    std::string last_error_;
    std::unordered_map<void*, size_t> allocs_;
};

} // namespace agr
