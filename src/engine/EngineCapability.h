// src/engine/EngineCapability.h
// Phase 11: Backend-independent capability enumeration.
//
// Each IExecutionEngine implementation reports what it can do.
// HeteroAccel code queries capabilities before routing requests.
// Capabilities are extensible: engines that do not support a
// capability simply exclude it from their reported set.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace agr {

/// Identifies a functional capability of an execution engine.
/// Designed to be extensible without breaking existing engines:
/// engines report only the capabilities they actually support.
enum class EngineCapability : uint32_t {
    // -----------------------------------------------------------------------
    // Workload types
    // -----------------------------------------------------------------------
    TEXT_GENERATION  = 0x0001, ///< Autoregressive text token generation
    EMBEDDINGS       = 0x0002, ///< Dense vector embedding extraction
    VISION           = 0x0004, ///< Image understanding (VLM)
    AUDIO            = 0x0008, ///< Audio transcription / generation
    MULTIMODAL       = 0x0010, ///< Multiple modalities in one request

    // -----------------------------------------------------------------------
    // Compute backends
    // -----------------------------------------------------------------------
    BACKEND_CPU      = 0x0100, ///< CPU-only compute path available
    BACKEND_VULKAN   = 0x0200, ///< Vulkan GPU path available
    BACKEND_CUDA     = 0x0400, ///< CUDA GPU path available
    BACKEND_NPU      = 0x0800, ///< NPU / dedicated AI accelerator

    // -----------------------------------------------------------------------
    // Runtime features
    // -----------------------------------------------------------------------
    STREAMING        = 0x1000, ///< Per-token streaming callbacks
    BATCHING         = 0x2000, ///< Simultaneous multi-request batching
    CANCELLATION     = 0x4000, ///< Supports mid-inference cancellation
    DYNAMIC_LOADING  = 0x8000, ///< Can load/unload models at runtime
};

/// A set of capabilities reported by an execution engine.
/// Use EngineCapabilitySet::has() to test individual capabilities.
class EngineCapabilitySet {
public:
    EngineCapabilitySet() = default;

    explicit EngineCapabilitySet(std::initializer_list<EngineCapability> caps) {
        for (auto c : caps) add(c);
    }

    void add(EngineCapability cap) {
        mask_ |= static_cast<uint32_t>(cap);
    }

    bool has(EngineCapability cap) const {
        return (mask_ & static_cast<uint32_t>(cap)) != 0;
    }

    bool empty() const { return mask_ == 0; }

    /// Raw bitmask — for serialisation / diagnostic use.
    uint32_t mask() const { return mask_; }

    /// Human-readable summary of all reported capabilities.
    std::string toString() const;

private:
    uint32_t mask_ = 0;
};

} // namespace agr
