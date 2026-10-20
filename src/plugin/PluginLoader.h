// src/plugin/PluginLoader.h
// Phase 19: public API of the dynamic plugin loader.
//
// SECURITY MODEL: plugins are native DLLs loaded into the host process.
// LoadLibrary executes plugin code (DllMain) with full host-process
// privileges. This is NOT a sandbox. Only load plugins from trusted
// locations; OS-level isolation is required for untrusted code.

#pragma once

#include "plugin/PluginAbis.h"

#include <string>
#include <vector>

namespace agr {

/// Scan the plugin directory (AGR_PLUGIN_DIR, or "./plugins") for plugin
/// DLLs. Each candidate is probed: loaded, ABI-checked, and queried for its
/// identity. Returns one entry per .dll found, with loadable=true for
/// plugins that passed validation.
std::vector<agrPluginDiscoveryResult> plugin_discover();

/// Load, ABI-check, validate and initialize a plugin by path.
/// Returns the plugin's agrPluginState value on success (>= LOADED), or -1
/// on failure (see plugin_get_error). Plugins are NOT registered as
/// executable engines by this call; execution requires plugin_execute(),
/// which fails for plugins that were not fully initialized.
int plugin_load(const std::string& path);

/// Execute one inference call through a loaded plugin. Resolves the
/// plugin's pluginExecute export and invokes it with the given context.
/// Returns AGR_PLUGIN_ERROR_NONE on success or an error code; -1 if the
/// plugin is not loaded.
int plugin_execute(const std::string& path, const void* ctx, void* outResult);

/// Shut a plugin down (pluginShutdown) and unload its module. Safe no-op if
/// the plugin is not loaded.
void plugin_unload(const std::string& path);

/// Current lifecycle state of a plugin (DISCOVERED if unknown).
agrPluginState plugin_get_state(const std::string& path);

/// Last error recorded for a plugin.
agrPluginError plugin_get_error(const std::string& path);

/// Identity of a loaded plugin. Returns false if not loaded.
bool plugin_get_identity(const std::string& path, agrPluginIdentity* out);

/// Paths of all currently loaded plugins.
std::vector<std::string> plugin_list_loaded();

} // namespace agr
