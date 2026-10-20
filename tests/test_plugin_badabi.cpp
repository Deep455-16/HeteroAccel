// tests/test_plugin_badabi.cpp
// Negative-control plugin: implements the required entry points but reports
// an ABI version the host does not expect. The loader must reject it during
// discovery and load, and must never call its other entry points.

#include "plugin/PluginAbis.h"
#include <cstring>

static bool g_other_entry_called = false;

extern "C" __declspec(dllexport) uint32_t __stdcall pluginGetAbiVersion(void) {
    return AGR_PLUGIN_ABI_VERSION + 99; // deliberately incompatible
}

extern "C" __declspec(dllexport) int __stdcall pluginGetIdentity(
    agrPluginModuleHandle, agrPluginIdentity* out) {
    g_other_entry_called = true;
    if (!out) return AGR_PLUGIN_ERROR_INVALID_REQUEST;
    const char* name = "badabi-plugin";
    size_t n = strlen(name);
    memcpy(out->name, name, n);
    out->name[n] = '\0';
    out->version[0] = '0'; out->version[1] = '\0';
    out->type[0] = 'L'; out->type[1] = 'L'; out->type[2] = 'M'; out->type[3] = '\0';
    out->author[0] = 'x'; out->author[1] = '\0';
    out->registry_key[0] = 'x'; out->registry_key[1] = '\0';
    return AGR_PLUGIN_ERROR_NONE;
}

extern "C" __declspec(dllexport) int __stdcall pluginInitialize(
    agrPluginModuleHandle, const agrPluginIdentity*, void**) {
    g_other_entry_called = true;
    return AGR_PLUGIN_ERROR_NONE;
}

extern "C" __declspec(dllexport) int __stdcall pluginExecute(
    agrPluginModuleHandle, const void*, void*) {
    g_other_entry_called = true;
    return AGR_PLUGIN_ERROR_NONE;
}

extern "C" __declspec(dllexport) int __stdcall pluginShutdown(agrPluginModuleHandle) {
    g_other_entry_called = true;
    return AGR_PLUGIN_ERROR_NONE;
}
