#pragma once
#include <string>
#include <vector>

namespace agr {

// Locates a compiled SPIR-V shader on disk. Shaders are compiled from
// source at build time (see /shaders/*.comp + CMakeLists.txt) and never
// committed as binaries. At runtime we look in a small set of plausible
// locations relative to the executable so the same binary works whether
// it's run from the build tree or copied elsewhere alongside its
// "shaders/" folder.
class ShaderLocator {
public:
    // Returns the full path to the first candidate location that exists,
    // or an empty string if none was found.
    static std::string find(const std::string& shaderFileName);

private:
    static std::string executableDir();
    static std::vector<std::string> candidateDirs();
};

} // namespace agr
