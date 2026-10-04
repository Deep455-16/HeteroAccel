// src/engine/EngineError.h
// Phase 11: Backend-independent error/status type.
//
// Engines report errors through EngineError rather than exposing
// llama.cpp-specific error codes or throwing exceptions.
// HeteroAccel code may inspect the kind for routing/fallback decisions.
#pragma once

#include <string>

namespace agr {

/// Categorised engine error kind.
/// Deliberately independent of any backend-specific error enumerations.
enum class EngineErrorKind {
    NONE,                ///< No error
    ENGINE_UNAVAILABLE,  ///< Engine backend not installed / initialised
    MODEL_NOT_FOUND,     ///< Model file missing or URI unreachable
    MODEL_LOAD_FAILED,   ///< File found but could not be parsed / loaded
    MODEL_NOT_LOADED,    ///< Operation requires a loaded model
    UNSUPPORTED_MODEL,   ///< Engine cannot handle this model format/arch
    UNSUPPORTED_OPERATION, ///< Capability not implemented by this engine
    UNSUPPORTED_DEVICE,  ///< Requested device not available
    RESOURCE_UNAVAILABLE,///< Insufficient RAM/VRAM to proceed
    EXECUTION_FAILED,    ///< Inference failed at runtime
    CANCELLED,           ///< Execution was cancelled by the caller
    INVALID_REQUEST,     ///< Malformed or logically invalid request
    INTERNAL,            ///< Unexpected internal engine error
};

/// Lightweight error descriptor returned by IExecutionEngine methods.
struct EngineError {
    EngineErrorKind kind    = EngineErrorKind::NONE;
    std::string     message;

    bool ok() const { return kind == EngineErrorKind::NONE; }

    static EngineError success() { return {}; }

    static EngineError make(EngineErrorKind k, std::string msg) {
        return { k, std::move(msg) };
    }
};

/// Stringify error kind for logging.
inline const char* toString(EngineErrorKind k) {
    switch (k) {
        case EngineErrorKind::NONE:                  return "None";
        case EngineErrorKind::ENGINE_UNAVAILABLE:    return "EngineUnavailable";
        case EngineErrorKind::MODEL_NOT_FOUND:       return "ModelNotFound";
        case EngineErrorKind::MODEL_LOAD_FAILED:     return "ModelLoadFailed";
        case EngineErrorKind::MODEL_NOT_LOADED:      return "ModelNotLoaded";
        case EngineErrorKind::UNSUPPORTED_MODEL:     return "UnsupportedModel";
        case EngineErrorKind::UNSUPPORTED_OPERATION: return "UnsupportedOperation";
        case EngineErrorKind::UNSUPPORTED_DEVICE:    return "UnsupportedDevice";
        case EngineErrorKind::RESOURCE_UNAVAILABLE:  return "ResourceUnavailable";
        case EngineErrorKind::EXECUTION_FAILED:      return "ExecutionFailed";
        case EngineErrorKind::CANCELLED:             return "Cancelled";
        case EngineErrorKind::INVALID_REQUEST:       return "InvalidRequest";
        case EngineErrorKind::INTERNAL:              return "Internal";
        default:                                     return "Unknown";
    }
}

} // namespace agr
