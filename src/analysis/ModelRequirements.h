#pragma once
#include <string>
#include <optional>
#include <cstdint>
#include "engine/EngineCapability.h"
#include "engine/ModelDescriptor.h"

namespace agr {

enum class QuantizationType {
    F32, F16, BF16, Q8, Q6, Q5, Q4, Q3, Q2, UNKNOWN
};

inline std::string toString(QuantizationType q) {
    switch (q) {
        case QuantizationType::F32: return "F32";
        case QuantizationType::F16: return "F16";
        case QuantizationType::BF16: return "BF16";
        case QuantizationType::Q8: return "Q8";
        case QuantizationType::Q6: return "Q6";
        case QuantizationType::Q5: return "Q5";
        case QuantizationType::Q4: return "Q4";
        case QuantizationType::Q3: return "Q3";
        case QuantizationType::Q2: return "Q2";
        default: return "UNKNOWN";
    }
}

// ModelModality is already defined in ModelDescriptor.h

struct ModelRequirements {
    std::string source_path;
    ModelFormat format = ModelFormat::UNKNOWN;

    std::optional<std::string> architecture;
    std::optional<uint64_t> parameter_count;
    std::optional<uint64_t> file_size_bytes;
    std::optional<uint64_t> estimated_memory_bytes;
    std::optional<uint32_t> layer_count;
    std::optional<uint32_t> context_length;

    QuantizationType quantization = QuantizationType::UNKNOWN;
    ModelModality modality = ModelModality::UNKNOWN;
    EngineCapabilitySet required_capabilities;
};

} // namespace agr
