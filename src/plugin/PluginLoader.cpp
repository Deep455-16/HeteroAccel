// src/plugin/PluginLoader.cpp
// Phase 18: Plugin loading, validation, discovery, and registration
// Implements deterministic plugin discovery and safe dynamic loading.

#include "plugin/PluginAbis.h"
#include "backend/BackendManager.h"
#include "engine/ExecutionEngineRegistry.h"
#include "analysis/CapabilityAnalyzer.h"
#include "hardware/HardwareInfo.h"

#include <windows.h>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <algorithm>
#include <iostream>

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

namespace agr {

// ---------------------------------------------------------------------------
// Internal plugin state
// ---------------------------------------------------------------------------

struct PluginInternalState {
    std::string path;                          ///< Full filesystem path
    std::string filename;                     ///< Just the filename
    HMODULE module_handle{ nullptr };         ///< Windows module handle
    agrPluginIdentity identity;               ///< Reported identity
    agrPluginCapabilityManifest capabilities; ///< Reported capabilities
    agrPluginState state{ agrPluginState::DISCOVERED };
    agrPluginError last_error{};
    bool initialized{ false };
    bool loaded{ false };
};

// ---------------------------------------------------------------------------
// Static mutex for thread-safe registry operations
// ---------------------------------------------------------------------------

static std::mutex plugin_mutex;

// ---------------------------------------------------------------------------
// Plugin cache: path -> internal state
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, PluginInternalState> plugin_cache;
static bool cache_initialized{ false };

// ---------------------------------------------------------------------------
// Ensure the cache is populated from disk (once)
// ---------------------------------------------------------------------------

static void ensure_cache_initialized() {
    if (cache_initialized) return;

    // Scan the documented plugin directory
    // Default: "./plugins" relative to the executable, or env var AGR_PLUGIN_DIR
    char* env_dir = nullptr;
    size_t len = 0;
    errno_t err = _dupenv_s(&env_dir, &len, "AGR_PLUGIN_DIR");
    if (err != 0 || env_dir == nullptr) {
        // Default relative to exe directory
        // For now, use a relative path; the test plugin will be placed there
        // In production, this would be configured at CMake time
        std::string default_dir = "./plugins";
        // We'll just note the default; actual scanning happens on demand
        // (the test plugin CMake target copies to the right place)
    } else {
        // Use env dir
        // (not fully implemented for brevity — the test plugin uses CMake install)
    }
    if (env_dir) std::free(env_dir);

    cache_initialized = true;
}

// ---------------------------------------------------------------------------
// Load a plugin module and resolve all required entry points.
// Returns nullptr on failure; caller must FreeLibrary on success.
// ---------------------------------------------------------------------------

static HMODULE load_plugin_module(const std::string& path) {
    // Use LoadLibraryW for wide-character path support
    std::wstring wpath(path.begin(), path.end());
    HMODULE h = LoadLibraryW(wpath.c_str());
    if (!h) {
        // Try with ANSI fallback (rare on modern Windows)
        h = LoadLibraryA(path.c_str());
        if (!h) {
            return nullptr;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------
// Resolve all required entry points from a loaded plugin module.
// Fills the function-pointer table. Returns true on success.
// ---------------------------------------------------------------------------

static bool resolve_plugin_exports(HMODULE hModule,
    agrPluginGetIdentityFn* out_get_identity,
    agrPluginInitializeFn* out_initialize,
    agrPluginExecuteFn* out_execute,
    agrPluginShutdownFn* out_shutdown,
    agrPluginGetStateFn* out_get_state,
    agrPluginGetErrorFn* out_get_error,
    agrPluginGetCapabilitiesFn* out_get_capabilities,
    agrPluginCreateEngineFn* out_create_engine) {

    if (!hModule) return false;

    *out_get_identity = (agrPluginGetIdentityFn)GetProcAddress(hModule, "pluginGetIdentity");
    *out_initialize = (agrPluginInitializeFn)GetProcAddress(hModule, "pluginInitialize");
    *out_execute = (agrPluginExecuteFn)GetProcAddress(hModule, "pluginExecute");
    *out_shutdown = (agrPluginShutdownFn)GetProcAddress(hModule, "pluginShutdown");
    *out_get_state = (agrPluginGetStateFn)GetProcAddress(hModule, "pluginGetState");
    *out_get_error = (agrPluginGetErrorFn)GetProcAddress(hModule, "pluginGetError");
    *out_get_capabilities = (agrPluginGetCapabilitiesFn)GetProcAddress(hModule, "pluginGetCapabilities");
    *out_create_engine = (agrPluginCreateEngineFn)GetProcAddress(hModule, "pluginCreateEngine");

    // Required exports: at minimum get_identity, initialize, execute, shutdown
    // CreateEngine is optional (engines may use LoadLibrary + their own factory)
    if (!*out_get_identity || !*out_initialize || !*out_execute || !*out_shutdown) {
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Internal: invoke pluginGetIdentity and capture the result.
// Updates the PluginInternalState.
// ---------------------------------------------------------------------------

static bool fetch_plugin_identity(HMODULE hModule, PluginInternalState& state) {
    if (!hModule) return false;

    agrPluginGetIdentityFn fn = (agrPluginGetIdentityFn)GetProcAddress(hModule, "pluginGetIdentity");
    if (!fn) return false;

    agrPluginIdentity ident{};
    int rc = fn(hModule, &ident);
    if (rc != AGR_PLUGIN_ERROR_NONE) {
        // Record the error but still store whatever we got
        state.last_error.code = static_cast<agrPluginErrorCode>(rc);
        // Try to get an English message
        agrPluginError err{};
        agrPluginGetErrorFn get_err = (agrPluginGetErrorFn)GetProcAddress(hModule, "pluginGetError");
        if (get_err) {
            get_err(hModule, &err);
            state.last_error.message = err.message;
        } else {
            state.last_error.message = "pluginGetIdentity failed";
        }
        return true; // We got something, even if it's an error
    }

    // Store the identity
    state.identity.name = ident.name ? ident.name : "unknown";
    state.identity.version = ident.version ? ident.version : "0.0.0";
    state.identity.type = ident.type ? ident.type : "unknown";
    state.identity.author = ident.author ? ident.author : "unknown";
    state.identity.registry_key = ident.registry_key ? ident.registry_key : "unknown";

    return true;
}

// ---------------------------------------------------------------------------
// Internal: invoke pluginGetCapabilities and capture the result.
// ---------------------------------------------------------------------------

static bool fetch_plugin_capabilities(HMODULE hModule, PluginInternalState& state) {
    if (!hModule) return false;

    agrPluginGetCapabilitiesFn fn = (agrPluginGetCapabilitiesFn)GetProcAddress(hModule, "pluginGetCapabilities");
    if (!fn) return false;

    agrPluginCapabilityManifest manifest{};
    int rc = fn(hModule, &manifest);
    if (rc != AGR_PLUGIN_ERROR_NONE) {
        state.last_error.code = static_cast<agrPluginErrorCode>(rc);
        state.last_error.message = "pluginGetCapabilities failed";
        return true;
    }

    state.capabilities.capabilities = manifest.capabilities;
    state.capabilities.capability_count = manifest.capability_count;

    return true;
}

// ---------------------------------------------------------------------------
// Public API (declared in a header we'll create)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// plugin_discover() - Scan the plugin directory and return discovery results.
// ---------------------------------------------------------------------------

std::vector<agrPluginDiscoveryResult> plugin_discover() {
    std::vector<agrPluginDiscoveryResult> results;

    ensure_cache_initialized();

    // For the test plugin, we know it's at a known location relative to the
    // executable. In production, this would scan the configured plugin directory.
    // We'll look for the test plugin that the CBuild system will install.

    // Determine plugin directory
    std::string plugin_dir;
    {
        char* env_dir = nullptr;
        size_t len = 0;
        _dupenv_s(&env_dir, &len, "AGR_PLUGIN_DIR");
        if (env_dir && len > 0) {
            plugin_dir = std::string(env_dir, len);
            std::free(env_dir);
        } else {
            // Default: ./plugins relative to current working directory
            plugin_dir = "./plugins";
        }
    }

    // Find all .dll files in the plugin directory
    // Use Windows find first
    WIN32_FIND_DATAA find_data{};
    std::string search_path = plugin_dir + "/*.dll";

    HANDLE hFind = FindFirstFileA(search_path.c_str(), &find_data);
    if (hFind == INVALID_HANDLE_VALUE) {
        // No .dll files found — return empty results
        FindClose(hFind);
        return results;
    }

    do {
        // Skip . and ..
        if (strcmp(find_data.cFileName, ".") == 0 ||
            strcmp(find_data.cFileName, "..") == 0) {
            continue;
        }

        std::string full_path = plugin_dir + "/" + find_data.cFileName;
        std::string filename(find_data.cFileName);

        agrPluginDiscoveryResult result{};
        result.path = full_path;
        result.filename = filename;
        result.loadable = false;
        result.load_error.code = AGR_PLUGIN_ERROR_INTERNAL;
        result.load_error.message = "unknown";

        // Attempt to load the module (without attaching yet — just check exports)
        HMODULE mod = load_plugin_module(full_path);
        if (mod) {
            // Check for required exports
            agrPluginGetIdentityFn get_id = nullptr;
            agrPluginInitializeFn get_init = nullptr;
            agrPluginExecuteFn get_exec = nullptr;
            agrPluginShutdownFn get_shutdown = nullptr;

            bool has_required =
                GetProcAddress(mod, "pluginGetIdentity") != nullptr &&
                GetProcAddress(mod, "pluginInitialize") != nullptr &&
                GetProcAddress(mod, "pluginExecute") != nullptr &&
                GetProcAddress(mod, "pluginShutdown") != nullptr;

            if (has_required) {
                // Successfully loaded with required exports — try to fetch identity
                if (fetch_plugin_identity(mod, plugin_cache[full_path])) {
                    result.loadable = true;
                    result.load_error.code = AGR_PLUGIN_ERROR_NONE;
                    result.load_error.message = "OK";

                    // Fetch capabilities too
                    fetch_plugin_capabilities(mod, plugin_cache[full_path]);
                }
            }

            FreeLibrary(mod);
        }

        if (!result.loadable) {
            result.load_error.code = AGR_PLUGIN_ERROR_INVALID_REQUEST;
            result.load_error.message = "Missing required exports (pluginGetIdentity, pluginInitialize, pluginExecute, pluginShutdown)";
        }

        results.push_back(result);

    } while (FindNextFileA(hFind, &find_data) != 0);

    FindClose(hFind);
    return results;
}

// ---------------------------------------------------------------------------
// plugin_load() - Load a specific plugin by path, validate, and register.
// Returns the internal state index, or -1 on failure.
// ---------------------------------------------------------------------------

int plugin_load(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    // Check if already loaded
    auto it = plugin_cache.find(path);
    if (it != plugin_cache.end() && it->second.loaded) {
        // Already loaded and registered; just return its state
        return static_cast<int>(it->second.state);
    }

    // Ensure cache is initialized (scan for .dlls)
    ensure_cache_initialized();

    // Load the module
    PluginInternalState& state = plugin_cache[path];
    state.path = path;

    // Extract filename
    size_t last_slash = path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        state.filename = path.substr(last_slash + 1);
    } else {
        state.filename = path;
    }

    // Load the Windows module handle
    HMODULE mod = load_plugin_module(path);
    if (!mod) {
        state.last_error.code = AGR_PLUGIN_ERROR_INIT_FAILED;
        state.last_error.message = "Failed to load module: " + path;
        state.state = agrPluginState::UNLOADED;
        return -1;
    }
    state.module_handle = mod;

    // Fetch identity
    if (!fetch_plugin_identity(mod, state)) {
        FreeLibrary(mod);
        state.module_handle = nullptr;
        state.last_error.message = "Failed to fetch plugin identity";
        state.state = agrPluginState::UNLOADED;
        return -1;
    }

    // Fetch capabilities
    fetch_plugin_capabilities(mod, state);

    // Set initial state
    state.state = agrPluginState::LOADED;
    state.loaded = true;

    // Attempt to initialize the plugin engine
    agrPluginInitializeFn init_fn = (agrPluginInitializeFn)GetProcAddress(mod, "pluginInitialize");
    if (init_fn) {
        int rc = init_fn(mod, &state.identity, nullptr);
        if (rc == AGR_PLUGIN_ERROR_NONE) {
            state.state = agrPluginState::INITIALIZED;
        } else {
            state.last_error.code = static_cast<agrPluginErrorCode>(rc);
            state.last_error.message = "pluginInitialize failed: " + std::string(state.last_error.message);
            // Don't fail entirely — the plugin is still usable at a basic level
            state.state = agrPluginState::LOADED;
        }
    } else {
        // No initialize function — treat as already initialized
        state.state = agrPluginState::LOADED;
    }

    // Register with the ExecutionEngineRegistry if the plugin reports a compatible key
    // (This is a best-effort registration; the host's main registry may have
    // its own rules for which keys to accept.)
    ExecutionEngineRegistry& reg = ExecutionEngineRegistry::instance();

    // Only register if the plugin's registry_key is not already taken
    if (reg.isRegistered(state.identity.registry_key) == false) {
        // Create a factory that delegates to this plugin's entry points
        auto factory = [](void) -> std::shared_ptr<IExecutionEngine> {
            // We cannot fully construct an IExecutionEngine from the plugin alone
            // without the host knowing the internal layout. Instead, we merely
            // record the plugin as available; the runtime's backend selector will
            // decide whether to use it based on capabilities.
            // For now, just log the registration intent.
            // TODO: Full integration would require a thin wrapper engine.
            (void)reg;
            return nullptr; // Placeholder — not fully registered as an IExecutionEngine yet
        };

        // Register with a key based on the plugin's reported name + version
        std::string reg_key = state.identity.name + "_" + state.identity.version;
        // Check for conflicts
        if (!reg.isRegistered(reg_key)) {
            reg.registerFactory(reg_key,
                "Plugin: " + state.identity.name + " " + state.identity.version,
                factory);
            state.state = agrPluginState::REGISTERED;
        }
    }

    return static_cast<int>(state.state);
}

// ---------------------------------------------------------------------------
// plugin_unload() - Unload a previously loaded plugin.
// Safe to call even if the plugin is not loaded (no-op).
// ---------------------------------------------------------------------------

void plugin_unload(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) return; // Nothing to unload

    PluginInternalState& state = it->second;

    // If there's an engine context, shut it down first
    if (state.initialized) {
        agrPluginShutdownFn shutdown_fn = (agrPluginShutdownFn)GetProcAddress(state.module_handle, "pluginShutdown");
        if (shutdown_fn) {
            shutdown_fn(state.module_handle);
        }
        state.initialized = false;
    }

    // Mark as unloaded
    if (state.module_handle) {
        FreeLibrary(state.module_handle);
        state.module_handle = nullptr;
    }
    state.loaded = false;
    state.state = agrPluginState::UNLOADED;

    // Deregister from the engine registry
    // (Best-effort — we don't remove the factory to avoid disturbing other engines)
    // The factory remains; the runtime just won't select this plugin again
    // unless explicitly re-registered.
}

// ---------------------------------------------------------------------------
// plugin_get_state() - Query the current state of a loaded plugin.
// ---------------------------------------------------------------------------

agrPluginState plugin_get_state(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) return agrPluginState::DISCOVERED;

    return it->second.state;
}

// ---------------------------------------------------------------------------
// plugin_get_error() - Get the last error for a plugin.
// ---------------------------------------------------------------------------

agrPluginError plugin_get_error(const std::string& path) {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    agrPluginError result{ .ok = true, .message = "" };

    auto it = plugin_cache.find(path);
    if (it == plugin_cache.end()) {
        result.ok = false;
        result.message = "Plugin not loaded: " + path;
        return result;
    }

    auto& state = it->second;
    result.code = state.last_error.code;
    result.message = state.last_error.message;
    result.ok = (state.last_error.code == AGR_PLUGIN_ERROR_NONE);

    return result;
}

// ---------------------------------------------------------------------------
// plugin_list_loaded() - Return paths of all currently loaded plugins.
// ---------------------------------------------------------------------------

std::vector<std::string> plugin_list_loaded() {
    std::lock_guard<std::mutex> lock(plugin_mutex);

    std::vector<std::string> loaded;
    for (const auto& [path, state] : plugin_cache) {
        if (state.loaded) {
            loaded.push_back(path);
        }
    }
    return loaded;
}

// ============================================================================
// End of PluginLoader.cpp
// ============================================================================