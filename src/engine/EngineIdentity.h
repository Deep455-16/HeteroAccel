// src/engine/EngineIdentity.h
// Phase 11: Identity metadata self-reported by an execution engine.
//
// Every IExecutionEngine must expose an immutable EngineIdentity.
// This allows HeteroAccel to log, route, and display engines without
// knowing their concrete type.
#pragma once

#include <string>

namespace agr {

/// Self-reported identity of an execution engine instance.
struct EngineIdentity {
    std::string name;     ///< Human-readable engine name, e.g. "llama.cpp"
    std::string version;  ///< Semantic or git-describe version, e.g. "b9999"
    std::string type;     ///< Category string, e.g. "LLM", "VLM", "Embed"
    std::string author;   ///< Attribution / project name

    /// Factory / registry key used to create this engine.
    /// Must match the key used with ExecutionEngineRegistry::registerFactory.
    std::string registry_key;
};

} // namespace agr
