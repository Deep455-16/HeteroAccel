#include "gpu/ShaderLocator.h"

#include <fstream>

#if defined(_WIN32)
    #include <windows.h>
#else
    #include <unistd.h>
    #include <climits>
#endif

// Defined by CMake to the build tree's compiled-shader output directory.
// This is a development/test convenience fallback only -- production
// distribution should ship the exe alongside its "shaders/" folder (the
// build already copies it there as a post-build step).
#ifndef AGR_SHADER_BUILD_DIR
    #define AGR_SHADER_BUILD_DIR ""
#endif

namespace agr {

std::string ShaderLocator::executableDir() {
#if defined(_WIN32)
    char path[MAX_PATH] = {0};
    DWORD len = GetModuleFileNameA(nullptr, path, MAX_PATH);
    if (len == 0) return "";
    std::string full(path, len);
#else
    char path[PATH_MAX] = {0};
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len <= 0) return "";
    std::string full(path, static_cast<size_t>(len));
#endif
    size_t pos = full.find_last_of("/\\");
    return pos == std::string::npos ? "" : full.substr(0, pos);
}

std::vector<std::string> ShaderLocator::candidateDirs() {
    std::vector<std::string> dirs;
    std::string exeDir = executableDir();
    if (!exeDir.empty()) {
        dirs.push_back(exeDir + "/shaders");
        // MSVC multi-config generators put the .exe in build/Release or
        // build/Debug -- shaders copied to the build root need one more "..".
        dirs.push_back(exeDir + "/../shaders");
    }
    std::string buildDir = AGR_SHADER_BUILD_DIR;
    if (!buildDir.empty()) {
        dirs.push_back(buildDir);
    }
    return dirs;
}

std::string ShaderLocator::find(const std::string& shaderFileName) {
    for (const auto& dir : candidateDirs()) {
        std::string candidate = dir + "/" + shaderFileName;
        std::ifstream f(candidate, std::ios::binary);
        if (f.good()) {
            return candidate;
        }
    }
    return "";
}

} // namespace agr
