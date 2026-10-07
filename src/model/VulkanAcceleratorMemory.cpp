#include "model/VulkanAcceleratorMemory.h"

namespace agr {

VulkanAcceleratorMemory::VulkanAcceleratorMemory(VulkanBackend* backend, uint64_t budget_bytes)
    : backend_(backend), budget_(budget_bytes) {}

VulkanAcceleratorMemory::~VulkanAcceleratorMemory() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (backend_) {
        for (auto& kv : buffers_) {
            backend_->destroyBuffer(kv.second);
        }
    }
    buffers_.clear();
    used_ = 0;
}

bool VulkanAcceleratorMemory::isAvailable() const {
    return backend_ && backend_->isAvailable();
}

uint64_t VulkanAcceleratorMemory::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return budget_ > used_ ? budget_ - used_ : 0;
}

std::string VulkanAcceleratorMemory::lastError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_error_.empty()) return last_error_;
    if (backend_) return backend_->lastError();
    return "No Vulkan backend";
}

void* VulkanAcceleratorMemory::allocate(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isAvailable()) {
        last_error_ = "Vulkan accelerator is unavailable";
        return nullptr;
    }
    if (bytes == 0) {
        last_error_ = "Zero-size Vulkan allocation";
        return nullptr;
    }
    if (used_ + bytes > budget_) {
        last_error_ = "Vulkan residency budget exceeded";
        return nullptr;
    }
    Buffer buf = backend_->createBuffer(bytes);
    if (buf.id == 0) {
        last_error_ = backend_->lastError();
        if (last_error_.empty()) last_error_ = "Vulkan createBuffer failed";
        return nullptr;
    }
    uint64_t key = next_key_++;
    buffers_[key] = buf;
    used_ += bytes;
    last_error_.clear();
    return reinterpret_cast<void*>(static_cast<uintptr_t>(key));
}

void VulkanAcceleratorMemory::release(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ptr || !backend_) return;
    uint64_t key = reinterpret_cast<uintptr_t>(ptr);
    auto it = buffers_.find(key);
    if (it == buffers_.end()) return;
    if (used_ >= it->second.size_bytes) used_ -= it->second.size_bytes;
    else used_ = 0;
    backend_->destroyBuffer(it->second);
    buffers_.erase(it);
}

bool VulkanAcceleratorMemory::upload(const void* source, void* destination, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!isAvailable()) {
        last_error_ = "Vulkan accelerator is unavailable";
        return false;
    }
    uint64_t key = reinterpret_cast<uintptr_t>(destination);
    auto it = buffers_.find(key);
    if (!source || it == buffers_.end()) {
        last_error_ = "Invalid Vulkan upload destination";
        return false;
    }
    if (!backend_->uploadBytes(it->second, source, bytes, nullptr)) {
        last_error_ = backend_->lastError();
        if (last_error_.empty()) last_error_ = "Vulkan uploadBytes failed";
        return false;
    }
    last_error_.clear();
    return true;
}

} // namespace agr
