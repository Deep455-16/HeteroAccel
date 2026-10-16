// src/plugin/PluginSdk.h
// Phase 18: Plugin SDK — Documentation and API reference for third-party engine authors.
// This header documents the stable, versioned C-ABI that independently compiled
// inference engines must implement to be discoverable, loadable, and executable
// within the HeteroAccel runtime.
//
// ABI VERSION: 1.0 (defined in PluginAbis.h)
// Do NOT modify this file when adding new features — add new exports instead.
// Existing plugins must continue to work with the current ABI version.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include "plugin/PluginAbis.h"

// ============================================================================
// ABI Version Check
// ============================================================================

// The host expects plugins to define AGR_PLUGIN_ABI_VERSION matching this value.
// If a plugin defines a different version, it will be rejected during discovery.
// Plugins must NOT change this value across minor releases without a major ABI bump.

#ifndef AGR_PLUGIN_ABI_VERSION
#define AGR_PLUGIN_ABI_VERSION 1
#error "AGR_PLUGIN_ABI_VERSION must be defined by the plugin build system"
#endif

#if AGR_PLUGIN_ABI_VERSION != AGR_HOST_EXPECTED_ABI_VERSION
#error "Plugin ABI version mismatch: plugin defines " #AGR_PLUGIN_ABI_VERSION \
       " but host expects " AGR_HOST_EXPECTED_ABI_VERSION
#endif

// ============================================================================
// Build Instructions for Plugin Authors
// ============================================================================

// 1. Add the following preprocessor define to your CMakeLists.txt or project settings:
//    add_definitions(-DAGR_PLUGIN_BUILDING_DLL)
//    add_definitions(-DAGR_PLUGIN_ABI_VERSION=1)
//
// 2. Link against the HeteroAccel import library (provided with the runtime).
//    The import library exports the function-pointer types so your plugin can
//    use them without redefining the calling convention.
//
// 3. Place your compiled .dll file in the plugin directory configured at runtime.
//    Default: ./plugins relative to the HeteroAccel executable.
//    Alternative: Set the AGR_PLUGIN_DIR environment variable to a custom path.
//
// 4. Ensure your compiler targets the same platform (Windows x64 / MSVC).
//    The ABI uses __stdcall calling convention; other platforms are not supported.

// ============================================================================
// Required Exports Checklist
// ============================================================================

// Every plugin DLL MUST export the following functions. The host uses
// GetProcAddress to resolve them at load time. Missing required exports
// cause the plugin to be rejected during discovery.

// Mandatory exports (must be present for the plugin to be considered loadable):

//   1. int __stdcall pluginGetIdentity(
//          agrPluginModuleHandle hModule,
//          agrPluginIdentity* outIdentity);
//
//      Writes the plugin's self-reported identity into outIdentity.
//      The host provides a pre-allocated agrPluginIdentity struct.
//      The plugin must not write beyond the struct's bounds.
//      Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
//
//   2. int __stdcall pluginInitialize(
//          agrPluginModuleHandle hModule,
//          const agrPluginIdentity* identity,
//          void** outEngineContext);
//
//      Performs one-time initialization: probe devices, allocate buffers,
//      validate model format support.
//      identity is the value returned by pluginGetIdentity.
//      outEngineContext receives an opaque context pointer (may be nullptr
//      if the plugin manages all state internally).
//      Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
//      Idempotent: may be called again after pluginShutdown().

//   3. int __stdcall pluginExecute(
//          agrPluginModuleHandle hModule,
//          const void* ctx,
//          void* outResult);
//
//      Perform inference with the currently loaded model.
//      ctx is an opaque ExecutionContext pointer provided by the host.
//      outResult must be filled with an ExecutionResult struct (defined
//      below). The host owns the memory for outResult; the plugin may write
//      into it but must not allocate new memory for the result struct itself.
//      Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
//      Must honour the cancel_flag embedded in ctx if set.

//   4. int __stdcall pluginShutdown(
//          agrPluginModuleHandle hModule);
//
//      Release all resources and prepare for unloading.
//      After this, pluginInitialize() may be called again.
//      Safe to call multiple times; subsequent calls are no-ops.
//      Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.

// Optional exports (if present, the host will use them; absence is not an error):

//   5. int __stdcall pluginGetState(
//          agrPluginModuleHandle hModule,
//          agrPluginState* outState);
//
//      Returns the current lifecycle state of the plugin instance.
//      outState receives one of the agrPluginState enum values.
//      If this export is absent, the host determines state from internal tracking.

//   6. int __stdcall pluginGetError(
//          agrPluginModuleHandle hModule,
//          agrPluginError* outError);
//
//      Returns the last error encountered by the plugin.
//      outError receives an agrPluginError struct with code and message.
//      If absent, the host reports no plugin-specific error information.

//   7. int __stdcall pluginGetCapabilities(
//          agrPluginModuleHandle hModule,
//          agrPluginCapabilityManifest* outManifest);
//
//      Returns the capability manifest (bitmask of supported capabilities).
//      outManifest->capabilities is a bitmask matching EngineCapability bits.
//      outManifest->capability_count is the number of bits set in the mask.
//      If absent, the plugin is assumed to support no capabilities (will not
//      be selected for any workload).

//   8. void __stdcall pluginCreateEngine(
//          agrPluginModuleHandle hModule,
//          const agrPluginIdentity* identity,
//          void** outEngineInstance);
//
//      Optional factory function. The host may use this to create an engine
//      instance without going through the full LoadLibrary + GetProcAddress
//      dance. If absent, the host always uses the dynamic-loading path.
//      The host owns the returned engine_instance.

// ============================================================================
// Data Structure Summaries
// ============================================================================

// agrPluginIdentity — self-reported plugin identity (filled by pluginGetIdentity):
//   - name[64]: Human-readable engine name, e.g. "kimi-llm"
//   - version[32]: Semantic version, e.g. "1.0.0"
//   - type[32]: Category string, e.g. "LLM"
//   - author[64]: Project or organization name
//   - registry_key[64]: Must match the key used with
//        ExecutionEngineRegistry::registerPlugin() (or the host's plugin
//        registration mechanism). Uniqueness is enforced; duplicate keys
//        cause registration to fail.

// agrPluginCapabilityManifest — reported capabilities:
//
//   - capabilities [uint32_t]: Bitmask. Each bit corresponds to an
//        EngineCapability enum value (see EngineCapability.h).
//        Bit 0  = TEXT_GENERATION (0x0001)
//        Bit 1  = EMBEDDINGS      (0x0002)
//        Bit 2  = VISION          (0x0004)
//        Bit 3  = AUDIO           (0x0008)
//        Bit 4  = MULTIMODAL      (0x0010)
//        Bit 8  = BACKEND_CPU     (0x0100)
//        Bit 9  = BACKEND_VULKAN  (0x0200)
//        Bit 10 = BACKEND_CUDA    (0x0400)
//        Bit 11 = BACKEND_NPU    (0x0800)
//        Bit 12 = STREAMING       (0x1000)
//        Bit 13 = BATCHING        (0x2000)
//        Bit 14 = CANCELLATION    (0x4000)
//        Bit 15 = DYNAMIC_LOADING (0x8000)
//        Bits 16-31 are reserved for future use; plugins must not set them.
//        Only bits that the plugin genuinely supports should be set.
//        Capabilities that are not set are treated as "not supported" by the host.
//
//   - capability_count [size_t]: Number of set bits in the capabilities mask.
//        The host uses this for quick bit-count operations.

// agrPluginError — error information returned by pluginGetError:
//
//   - code [agrPluginErrorCode]: One of:
//        AGR_PLUGIN_ERROR_NONE (0)
//        AGR_PLUGIN_ERROR_INIT_FAILED (1)
//        AGR_PLUGIN_ERROR_MODEL_NOT_FOUND (2)
//        AGR_PLUGIN_ERROR_MODEL_LOAD_FAILED (3)
//        AGR_PLUGIN_ERROR_UNSUPPORTED_MODEL (4)
//        AGR_PLUGIN_ERROR_OUT_OF_MEMORY (5)
//        AGR_PLUGIN_ERROR_INVALID_REQUEST (6)
//        AGR_PLUGIN_ERROR_INTERNAL (7)
//   - message [256 chars]: Null-terminated human-readable description.
//        The plugin must not write beyond 255 characters (plus null terminator).

// agrPluginState — lifecycle state reported by pluginGetState (or derived):
//
//   - DISCOVERED   (0): Plugin library found on disk, not yet validated.
//   - VALIDATED    (1): Metadata validated successfully.
//   - LOADED       (2): Module handle loaded via LoadLibrary.
//   - INITIALIZED  (3): engine.initialize() has been called successfully.
//   - REGISTERED   (4): Engine registered with the ExecutionEngineRegistry.
//   - READY        (5): Ready to accept execute() calls.
//   - SHUTTING_DOWN (6): Shutdown in progress.
//   - UNLOADED     (7): Module unloaded from memory.

// agrPluginDiscoveryResult — result of scanning a plugin directory:
//
//   - path [512 chars]: Full filesystem path to the .dll file.
//   - filename [256 chars]: Filename portion only.
//   - identity [agrPluginIdentity]: Reported identity, populated if
//        pluginGetIdentity() succeeded during discovery.
//   - load_error [agrPluginError]: Error from attempting to load/query the plugin.
//   - loadable [bool]: True if the plugin passed basic validation
//        (has required exports, identity fetchable, etc.).

// ============================================================================
// Compatibility Guidelines for ABI Evolution
// ============================================================================

// When the HeteroAccel team introduces a new ABI version (e.g., 2.0):
//
// 1. Do NOT remove or re-number existing exports.
// 2. May add new optional exports.
// 3. May change the semantics of existing exports only with a major version bump.
// 4. Plugins built for old ABI versions must continue to work with the new host
//    (backward compatibility).
// 5. New plugins may target only the current ABI version.
// 6. The host will reject plugins whose AGR_PLUGIN_ABI_VERSION does not match
//    the expected version at discovery time.
// 7. A compatibility shim may be provided for one release series only.

// ============================================================================
// Example: Minimal Plugin Main (for authors)
// ============================================================================

/*
// plugin_main.cpp — Minimal plugin entry point (what your DllMain should look like)
// This is a skeleton; real plugins will have substantial internal logic.

#include "plugin/PluginAbis.h"

// The one thing DllEntryPoint must do: pass control to the runtime's
// module-reference-counting mechanism (if any). Most plugin frameworks
// do NOT require DllMain, but if you have global state, protect it.

BOOL API DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // One-time initialization (avoid heavy work here)
        break;
    case DLL_PROCESS_DETACH:
        // Cleanup
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}

// The exported functions are the plugin's contract with the host.
// They are invoked by the host via GetProcAddress after LoadLibrary.

// pluginGetIdentity:
int __stdcall pluginGetIdentity(agrPluginModuleHandle hModule, agrPluginIdentity* outIdentity) {
    // Write identity information
    strncpy_s(outIdentity->name, outIdentity->name_size, "my-engine", _TRUNCATE);
    strncpy_s(outIdentity->version, outIdentity->version_size, "1.0.0", _TRUNCATE);
    strncpy_s(outIdentity->type, outIdentity->type_size, "LLM", _TRUNCATE);
    strncpy_s(outIdentity->author, outIdentity->author_size, "My Organization", _TRUNCATE);
    strncpy_s(outIdentity->registry_key, outIdentity->registry_key_size, "my-engine", _TRUNCATE);
    return AGR_PLUGIN_ERROR_NONE;
}

// pluginInitialize:
int __stdcall pluginInitialize(agrPluginModuleHandle hModule, const agrPluginIdentity* identity, void** outEngineContext) {
    // Allocate internal structures, probe devices, etc.
    *outEngineContext = nullptr; // or a valid opaque context
    return AGR_PLUGIN_ERROR_NONE;
}

// pluginExecute:
int __stdcall pluginExecute(agrPluginModuleHandle hModule, const void* ctx, void* outResult) {
    // Perform inference, fill outResult with ExecutionResult
    // outResult must be written in-place; do not malloc new memory for it.
    // Return AGR_PLUGIN_ERROR_NONE on success.
    return AGR_PLUGIN_ERROR_NONE;
}

// pluginShutdown:
int __stdcall pluginShutdown(agrPluginModuleHandle hModule) {
    // Free all allocated resources.
    return AGR_PLUGIN_ERROR_NONE;
}
*/

// ============================================================================
// End of PluginSdk.h
// ============================================================================