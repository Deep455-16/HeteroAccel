// src/engine/ExecutionEngineRegistry.h
// Phase 11: Factory registry for IExecutionEngine implementations.
//
// Provides a central place to register and create execution engines by
// string key. Avoids hard-coded if/else chains in HeteroRuntime and
// enables future plugins to register engines at startup.
//
// Phase 11 ships with one built-in registration:
//   "llama.cpp" → LlamaCppExecutionEngine
//
// Future registrations (not Phase 11):
//   "colibri"  → ColibriBackend
//   "native"   → NativeHeteroAccelBackend
//
// Thread safety:
//   - registerFactory() is not thread-safe; call from one thread at
//     startup before any concurrent usage.
//   - create() and query methods are thread-safe after registration.
#pragma once

#include "engine/IExecutionEngine.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace agr {

/// Factory function type: creates a new engine instance.
using EngineFactory = std::function<std::shared_ptr<IExecutionEngine>()>;

/// Metadata stored per registered engine factory.
struct EngineRegistration {
    std::string    key;         ///< Registry key, e.g. "llama.cpp"
    std::string    description; ///< Human-readable description
    EngineFactory  factory;     ///< Callable that produces a new instance
};

/// Singleton registry mapping engine keys to factory functions.
///
/// Usage:
///   // At startup:
///   ExecutionEngineRegistry::instance().registerFactory(
///       "llama.cpp", "llama.cpp LLM backend", []{ return std::make_shared<LlamaCppExecutionEngine>(); });
///
///   // At runtime:
///   auto engine = ExecutionEngineRegistry::instance().create("llama.cpp");
class ExecutionEngineRegistry {
public:
    /// Singleton accessor.
    static ExecutionEngineRegistry& instance();

    /// Register a factory under a given key.
    /// Overwrites any previous registration for the same key.
    /// Not thread-safe — call from one thread at startup.
    void registerFactory(const std::string& key,
                         const std::string& description,
                         EngineFactory factory);

    /// Create a new engine instance for the given key.
    /// Returns nullptr if the key is not registered.
    std::shared_ptr<IExecutionEngine> create(const std::string& key) const;

    /// Return true if a factory is registered under key.
    bool isRegistered(const std::string& key) const;

    /// Return keys of all registered engines.
    std::vector<std::string> registeredKeys() const;

    /// Return the full registration record for a key, or nullptr.
    const EngineRegistration* registration(const std::string& key) const;

    /// Remove all registrations (mainly for test isolation).
    void clear();

private:
    ExecutionEngineRegistry();

    mutable std::mutex mutex_;
    std::unordered_map<std::string, EngineRegistration> factories_;
};

} // namespace agr
