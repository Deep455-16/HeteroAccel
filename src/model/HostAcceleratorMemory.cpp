#include "model/HostAcceleratorMemory.h"

#include <cstring>
#include <new>

namespace agr {

HostAcceleratorMemory::HostAcceleratorMemory(uint64_t capacity_bytes)
    : capacity_(capacity_bytes) {}

uint64_t HostAcceleratorMemory::capacity() const {
    return capacity_;
}

uint64_t HostAcceleratorMemory::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return capacity_ > used_ ? capacity_ - used_ : 0;
}

std::string HostAcceleratorMemory::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_error_;
}

void* HostAcceleratorMemory::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (bytes == 0) {
        last_error_ = "Zero-size accelerator allocation";
        return nullptr;
    }
    if (used_ + bytes > capacity_) {
        last_error_ = "Accelerator budget exceeded";
        return nullptr;
    }
    auto* p = new (std::nothrow) uint8_t[bytes];
    if (!p) {
        last_error_ = "Host accelerator allocation failed";
        return nullptr;
    }
    allocs_[p] = bytes;
    used_ += bytes;
    last_error_.clear();
    return p;
}

void HostAcceleratorMemory::release(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocs_.find(ptr);
    if (it == allocs_.end()) return;
    used_ -= it->second;
    delete[] static_cast<uint8_t*>(ptr);
    allocs_.erase(it);
}

bool HostAcceleratorMemory::upload(const void* source, void* destination, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fail_next_upload_) {
        fail_next_upload_ = false;
        last_error_ = "Simulated accelerator upload failure";
        return false;
    }
    auto it = allocs_.find(destination);
    if (!source || it == allocs_.end() || bytes > it->second) {
        last_error_ = "Invalid accelerator upload";
        return false;
    }
    std::memcpy(destination, source, bytes);
    last_error_.clear();
    return true;
}

bool HostAcceleratorMemory::download(void* destination, const void* source, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocs_.find(const_cast<void*>(source));
    if (!destination || it == allocs_.end() || bytes > it->second) {
        last_error_ = "Invalid accelerator download";
        return false;
    }
    std::memcpy(destination, source, bytes);
    return true;
}

} // namespace agr
