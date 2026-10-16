// src/plugin/PluginAbis.h
// Phase 18: Stable plugin ABI definitions
// Versioned, C-compatible interface for dynamically loadable inference engines.
//
// ABI VERSION: 1.0
// This header defines the exported types and function signatures that
// all plugins must implement. It is intentionally C-compatible so that
// independently compiled plugins can interop with the HeteroAccel runtime.
//
// Ownership model: The host (HeteroAccel) owns plugin module handles.
// Plugins must not retain pointers into unloaded modules.
// All resources must be destroyed by the module's allocator.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

// ============================================================================
// ABI Version
// ============================================================================

// Current ABI version. Plugins and host must agree on this version.
#define AGR_PLUGIN_ABI_VERSION 1

// ============================================================================
// Plugin Identity
// ============================================================================

// Opaque handle to a loaded plugin module.
typedef struct agrPluginModuleTag* agrPluginModuleHandle;

// Plugin identity information reported by the plugin.
typedef struct {
    char name[64];          ///< Engine name (null-terminated, max 63 chars)
    char version[32];       ///< Plugin version (null-terminated, max 31 chars)
    char type[32];          ///< Category: "LLM", "VLM", "Embed", etc.
    char author[64];        ///< Author/project name
    char registry_key[64]; ///< Must match key used with registry::registerPlugin()
} agrPluginIdentity;

// ============================================================================
// Capability Manifest
// ============================================================================

// A capability bit, corresponding to EngineCapability enum values.
// Mapped 1:1 for interop; plugins report capabilities using the same bit positions.
typedef uint32_t agrPluginCapabilityMask;

// Maximum number of capabilities a plugin can report.
#define AGR_MAX_PLUGIN_CAPABILITIES 32

// Capability manifest reported by each plugin.
typedef struct {
    agrPluginCapabilityMask capabilities;  ///< Bitmask of supported capabilities
    size_t  capability_count;              ///< Number of set bits in mask
} agrPluginCapabilityManifest;

// ============================================================================
// Plugin Error Reporting
// ============================================================================

// Error code returned by plugin functions.
// Mapped to EngineErrorKind on the host side.
typedef enum {
    AGR_PLUGIN_ERROR_NONE           = 0,
    AGR_PLUGIN_ERROR_INIT_FAILED,      ///< Plugin initialization failed
    AGR_PLUGIN_ERROR_MODEL_NOT_FOUND,  ///< Model file not found
    AGR_PLUGIN_ERROR_MODEL_LOAD_FAILED,///< Model could not be loaded
    AGR_PLUGIN_ERROR_UNSUPPORTED_MODEL,///< Model format not supported
    AGR_PLUGIN_ERROR_OUT_OF_MEMORY,    ///< Out of resources
    AGR_PLUGIN_ERROR_INVALID_REQUEST,  ///< Invalid request parameters
    AGR_PLUGIN_ERROR_INTERNAL,         ///< Internal engine error
    AGR_PLUGIN_ERROR_LAST              ///< Sentinel (not a valid error code)
} agrPluginErrorCode;

// Human-readable error message for plugin errors.
typedef struct {
    agrPluginErrorCode code;   ///< Error code
    char message[256];         ///< Null-terminated error description
} agrPluginError;

// ============================================================================
// Plugin Lifecycle States
// ============================================================================

// Lifecycle state of a plugin instance, reported to the host.
typedef enum {
    AGR_PLUGIN_STATE_DISCOVERED,   ///< Plugin library found on disk
    AGR_PLUGIN_STATE_VALIDATED,    ///< Plugin metadata validated successfully
    AGR_PLUGIN_STATE_LOADED,        ///< Module handle loaded via LoadLibrary
    AGR_PLUGIN_STATE_INITIALIZED,  ///< Engine initialized (initialize() called)
    AGR_PLUGIN_STATE_REGISTERED,   ///< Engine registered with registry
    AGR_PLUGIN_STATE_READY,         ///< Ready to accept execute() calls
    AGR_PLUGIN_STATE_SHUTTING_DOWN, ///< Shutdown in progress
    AGR_PLUGIN_STATE_UNLOADED        ///< Module unloaded from memory
} agrPluginState;

// ============================================================================
// Exported Function Entry Points
// Every plugin DLL must export these functions. The host uses GetProcAddress
// to resolve them. Failure to export required functions is a fatal load error.
// Linkage: __stdcall on Windows (matches GetProcAddress conventions).

// ---------------------------------------------------------------------------
// pluginGetIdentity()
// Called once after the module is loaded (before pluginInitialize()).
// The host provides a buffer where the plugin must write its identity.
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// The plugin must not retain pointers into the host-provided buffer.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginGetIdentityFn)(
    agrPluginModuleHandle hModule,
    agrPluginIdentity* outIdentity);

// ---------------------------------------------------------------------------
// pluginInitialize()
// Called after pluginGetIdentity(). Performs one-time initialization:
//   - Probes devices (CPU/Vulkan)
//   - Allocates internal buffers
//   - Validates model format support
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// Idempotent: may be called again after pluginShutdown() without harm.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginInitializeFn)(
    agrPluginModuleHandle hModule,
    const agrPluginIdentity* identity,
    void** outEngineContext);

// ---------------------------------------------------------------------------
// pluginExecute()
// Perform inference with the currently loaded model.
// ctx is an ExecutionContext-like structure passed by the host.
// outResult receives the ExecutionResult.
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// Must honour the cancel_flag from ctx if set.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginExecuteFn)(
    agrPluginModuleHandle hModule,
    const void* ctx,            ///< Opaque ExecutionContext pointer
    void* outResult);           ///< Pointer to ExecutionResult struct

// ---------------------------------------------------------------------------
// pluginShutdown()
// Release all resources and prepare for unloading.
// After this, pluginInitialize() may be called again.
// Safe to call multiple times; subsequent calls are no-ops.
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginShutdownFn)(
    agrPluginModuleHandle hModule);

// ---------------------------------------------------------------------------
// pluginGetState()
// Returns the current lifecycle state of the plugin instance.
// Returns: agrPluginState value.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginGetStateFn)(
    agrPluginModuleHandle hModule);

// ---------------------------------------------------------------------------
// pluginGetError()
// Returns the last error encountered by the plugin.
// outError receives an agrPluginError struct.
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginGetErrorFn)(
    agrPluginModuleHandle hModule,
    agrPluginError* outError);

// ---------------------------------------------------------------------------
// pluginGetCapabilities()
// Returns the capability manifest reported by the plugin.
// outManifest receives the capability bitmask and count.
// Returns: AGR_PLUGIN_ERROR_NONE on success, or an error code.
// ---------------------------------------------------------------------------
typedef int (__stdcall *agrPluginGetCapabilitiesFn)(
    agrPluginModuleHandle hModule,
    agrPluginCapabilityManifest* outManifest);

// ============================================================================
// Plugin Factory Function (optional)
// Used by the host to create engine instances without LoadLibrary
// if the plugin is already known to be available.
// Returns: nullptr if factory cannot create an engine.
// The host owns the returned engine instance.
// ---------------------------------------------------------------------------
typedef void* (__stdcall *agrPluginCreateEngineFn)(
    agrPluginModuleHandle hModule,
    const agrPluginIdentity* identity);

// ============================================================================
// Plugin Discovery Result
// ============================================================================

// Result of scanning a directory for eligible plugin libraries.
typedef struct {
    char path[512];          ///< Full path to the plugin .dll file
    char filename[256];      ///< Filename portion only
    agrPluginIdentity identity; ///< Identity if pluginGetIdentity() succeeded
    agrPluginError load_error; ///< Error from attempting to load/query the plugin
    bool    loadable;        ///< True if the plugin passed basic validation
} agrPluginDiscoveryResult;

// ============================================================================
// Host-Plugin Interface Version Check
// ============================================================================

// The host queries the plugin's ABI version during discovery.
// Plugins must define AGR_PLUGIN_ABI_VERSION in their build configuration.
// If the plugin's version does not match the host's expected version,
// the plugin is rejected with a clear diagnostic.
#define AGR_HOST_EXPECTED_ABI_VERSION 1

// ============================================================================
// Export Control Macros (for plugin authors)
// When building a plugin DLL, define AGR_PLUGIN_BUILDING_DLL before
// including this header to get the __declspec(dllexport) behavior.
// When using/consuming the DLL, do NOT define that macro.
// ---------------------------------------------------------------------------
#ifdef AGR_PLUGIN_BUILDING_DLL
#define AGR_PLUGIN_EXPORT __declspec(dllexport)
#else
#define AGR_PLUGIN_EXPORT __declspec(dllimport)
#endif // AGR_PLUGIN_BUILDING_DLL

// ============================================================================
// Legacy compatibility macro (Phase 11-era engines)
// Engines that were built before this ABI was introduced can still be
// registered through the existing ExecutionEngineRegistry mechanism.
// This is not a plugin ABI entry point; it's a compatibility shim.
#define AGR_COMPATIBILITY_MODE_LLAMA_CPP 0

// ============================================================================
// Static assertions / compile-time checks
// ---------------------------------------------------------------------------
// Ensure size compatibility between plugin types and host types.
// These static_asserts help catch ABI mismatches at compile time
// when building the host (not when building plugins, which are C-only).

static_assert(sizeof(agrPluginIdentity) == 256,
    "agrPluginIdentity must be exactly 256 bytes (64+32+32+64+64)");

static_assert(sizeof(agrPluginError) == 260,
    "agrPluginError must be exactly 260 bytes (4+256)");

// The message field inside agrPluginError is 256 bytes + 4 bytes kind +
// alignment padding to 292 total on 4-byte boundaries.

// ============================================================================
// End of PluginAbis.h
// ============================================================================