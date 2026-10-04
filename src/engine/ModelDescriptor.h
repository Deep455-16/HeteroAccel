// src/engine/ModelDescriptor.h
// Phase 11: Backend-independent model descriptor.
//
// Describes a model or model resource without assuming any specific
// inference engine or file format.
//
// Phase 12 (Model Inspection & Capability Analysis) will build a
// richer capability-analysis system on top of this descriptor.
// Phase 11 only needs enough structure to load and identify a model.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace agr {

/// File format / serialisation format of the model weights.
/// Unknown formats are represented by UNKNOWN so that engines can
/// report an honest "I cannot handle this format" rather than crash.
enum class ModelFormat {
    UNKNOWN,        ///< Not determined or not applicable
    GGUF,           ///< llama.cpp native quantised format
    SAFETENSORS,    ///< HuggingFace safetensors
    ONNX,           ///< ONNX runtime format
    PYTORCH,        ///< PyTorch .pt / .bin shards
    NATIVE_AGR,     ///< Future: native HeteroAccel format
    OTHER,          ///< Any other engine-specific format
};

/// Broad modality class. A multimodal model may serve several.
enum class ModelModality {
    UNKNOWN = 0,
    TEXT,
    VISION,
    AUDIO,
    MULTIMODAL,
};

/// Backend-independent descriptor of a model resource.
/// All fields except `model_id` and `source_path` are optional because
/// different engines and phases may populate them progressively.
struct ModelDescriptor {
    // ------------------------------------------------------------------
    // Required fields
    // ------------------------------------------------------------------
    std::string model_id;    ///< Stable identifier (e.g. model path or UUID)
    std::string source_path; ///< Path or URI to the model file(s)

    // ------------------------------------------------------------------
    // Optional metadata (populated by Phase 12 capability analysis)
    // ------------------------------------------------------------------
    ModelFormat  format   = ModelFormat::UNKNOWN;
    ModelModality modality = ModelModality::UNKNOWN;

    std::string architecture;     ///< e.g. "llama", "qwen2", "mistral"
    std::string quantization;     ///< e.g. "Q4_K_M", "FP16"
    std::string description;      ///< Human-readable label

    std::optional<int64_t>  param_count; ///< Parameter count (if known)
    std::optional<int32_t>  n_layers;    ///< Transformer layer count
    std::optional<int32_t>  n_ctx_max;   ///< Maximum context length
    std::optional<size_t>   size_bytes;  ///< Approximate weight size on disk

    // ------------------------------------------------------------------
    // Arbitrary engine-specific metadata (key-value pairs)
    // ------------------------------------------------------------------
    std::unordered_map<std::string, std::string> metadata;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------
    bool isValid() const { return !model_id.empty() && !source_path.empty(); }

    /// Produce a human-readable summary (single line).
    std::string summary() const;
};

/// Stringify ModelFormat for logging/diagnostics.
inline const char* toString(ModelFormat f) {
    switch (f) {
        case ModelFormat::GGUF:          return "GGUF";
        case ModelFormat::SAFETENSORS:   return "SafeTensors";
        case ModelFormat::ONNX:          return "ONNX";
        case ModelFormat::PYTORCH:       return "PyTorch";
        case ModelFormat::NATIVE_AGR:    return "NativeAGR";
        case ModelFormat::OTHER:         return "Other";
        default:                         return "Unknown";
    }
}

/// Stringify ModelModality.
inline const char* toString(ModelModality m) {
    switch (m) {
        case ModelModality::TEXT:        return "Text";
        case ModelModality::VISION:      return "Vision";
        case ModelModality::AUDIO:       return "Audio";
        case ModelModality::MULTIMODAL:  return "Multimodal";
        default:                         return "Unknown";
    }
}

} // namespace agr
