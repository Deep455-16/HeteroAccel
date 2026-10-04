// src/engine/IExecutionEngine.h
// Phase 11: Universal Execution Engine Interface.
//
// This is the central abstraction introduced in Phase 11.
// All inference backends (llama.cpp, Colibrì, future engines) must
// implement this interface. HeteroAccel's runtime layer talks only to
// IExecutionEngine; llama.cpp internals never leak upward.
//
// ARCHITECTURAL POSITION:
//
//   HeteroRuntime
//        │
//        ▼
//   IExecutionEngine   ← this file
//        │
//   ┌────┼────────────────┐
//   ▼    ▼                ▼
//  LlamaCppBackend  (future: Colibrì / NativeAGR / ...)
//        │
//   HeteroAccel Resource Layer (CPU / GPU / NPU)
//
// CONCURRENCY CONTRACT (Phase 11):
//   - A single IExecutionEngine instance is NOT guaranteed thread-safe
//     for concurrent execute() calls. Each engine documents its own
//     thread safety in its header.
//   - Independent IExecutionEngine instances may execute concurrently
//     provided their shared resources (e.g. llama_backend) are managed
//     safely (see LlamaCppEngine's reference-counted backend lifecycle).
//   - Callers that require concurrent inference must use independent
//     engine instances.
//
// CANCELLATION:
//   - ExecutionContext::cancel_flag is a caller-owned atomic<bool>.
//   - Engines must poll it during generation and stop within a bounded
//     number of additional tokens.
//
// NOT IN PHASE 11:
//   - Automatic device planning (Phase 13)
//   - Tensor-level streaming / residency (Phase 14)
//   - Heterogeneous execution graph (Phase 15)
//   - Native HeteroAccel inference (Phase 17)
//   - Application SDK (Phase 18)
#pragma once

#include "engine/EngineCapability.h"
#include "engine/EngineError.h"
#include "engine/EngineIdentity.h"
#include "engine/ExecutionContext.h"
#include "engine/ExecutionResult.h"
#include "engine/ModelDescriptor.h"
#include "inference/InferenceTypes.h"  // for TokenCallback

#include <functional>
#include <memory>
#include <string>

namespace agr {

/// Universal Execution Engine Interface.
///
/// Implementations must be non-copyable (engines own significant
/// resources). Move semantics are implementation-defined.
class IExecutionEngine {
public:
    virtual ~IExecutionEngine() = default;

    // ===================================================================
    // Identity & capabilities
    // ===================================================================

    /// Return immutable identity of this engine.
    virtual EngineIdentity identity() const = 0;

    /// Return the set of capabilities supported by this engine.
    /// HeteroAccel uses this to route requests to capable engines.
    virtual EngineCapabilitySet capabilities() const = 0;

    // ===================================================================
    // Engine lifecycle
    // ===================================================================

    /// Initialise the engine (allocate internal resources, probe devices).
    /// Must be called before any model or execution operations.
    /// Idempotent: returns true immediately if already initialised.
    ///
    /// On failure, populates internal error state retrievable via lastError().
    virtual bool initialize() = 0;

    /// Shut down the engine and release all resources.
    /// After shutdown(), initialize() may be called again.
    /// Safe to call on an uninitialised engine (no-op).
    virtual void shutdown() = 0;

    /// Whether the engine has been successfully initialised.
    virtual bool isInitialized() const = 0;

    // ===================================================================
    // Model lifecycle
    // ===================================================================

    /// Load a model described by `descriptor`.
    ///
    /// The engine inspects descriptor.format and descriptor.source_path
    /// to locate and load the model. Fields that the engine does not
    /// need (e.g. param_count) are ignored.
    ///
    /// Returns true on success.
    /// On failure populates internal error state.
    virtual bool loadModel(const ModelDescriptor& descriptor) = 0;

    /// Unload the currently loaded model and free associated memory.
    /// Safe to call if no model is loaded (no-op).
    virtual void unloadModel() = 0;

    /// Whether a model is currently loaded and ready for execution.
    virtual bool isModelLoaded() const = 0;

    /// Descriptor of the currently loaded model.
    /// Valid only when isModelLoaded() == true.
    virtual ModelDescriptor currentModel() const = 0;

    // ===================================================================
    // Execution
    // ===================================================================

    /// Execute inference synchronously.
    ///
    /// Blocks until generation is complete or cancelled.
    /// ctx.cancel_flag may be polled during generation; set it to true
    /// to abort. The returned result will have success == false and
    /// error indicating cancellation.
    ///
    /// CONCURRENCY: calls on the same instance are serialised.
    virtual ExecutionResult execute(const ExecutionContext& ctx) = 0;

    /// Execute inference with per-token streaming callback.
    ///
    /// cb is called for each generated text piece.
    /// The complete result is also returned when generation finishes.
    ///
    /// Engines that do not support STREAMING capability fall back to
    /// calling cb once with the complete output.
    virtual ExecutionResult executeStreaming(
        const ExecutionContext& ctx,
        TokenCallback cb) = 0;

    // ===================================================================
    // Error handling
    // ===================================================================

    /// Return the last error encountered by any operation.
    /// Cleared on each successful call to initialize() / loadModel().
    virtual EngineError lastEngineError() const = 0;

    /// Convenience: return last error message as string.
    std::string lastError() const {
        auto e = lastEngineError();
        return e.ok() ? "" : e.message;
    }
};

/// Shared ownership alias used throughout HeteroAccel.
using ExecutionEnginePtr = std::shared_ptr<IExecutionEngine>;

} // namespace agr
