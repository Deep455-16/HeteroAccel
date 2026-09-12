cmake_minimum_required(VERSION 3.20)
project(AdaptiveGPURuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# --- Vulkan (Phase 1 detection + Phase 2 real compute) --------------------
# The Vulkan SDK/loader is expected to be installed on the system
# (per project ground rules, we never bundle SDKs into the repo).
find_package(Vulkan)

# --- Phase 2: compile the compute shader from source at build time -------
# Never a committed binary -- glslangValidator (or glslc) regenerates the
# SPIR-V module from shaders/vector_add.comp on every build.
find_program(GLSLANG_VALIDATOR NAMES glslangValidator)
if(NOT GLSLANG_VALIDATOR)
    find_program(GLSLC NAMES glslc)
endif()

set(AGR_SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shaders/vector_add.comp)
set(AGR_SHADER_SPV ${CMAKE_BINARY_DIR}/shaders/vector_add.comp.spv)
set(AGR_HAVE_SHADER_COMPILER FALSE)

if(GLSLANG_VALIDATOR)
    add_custom_command(
        OUTPUT ${AGR_SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLANG_VALIDATOR} -V ${AGR_SHADER_SRC} -o ${AGR_SHADER_SPV}
        DEPENDS ${AGR_SHADER_SRC}
        COMMENT "Compiling vector_add.comp -> SPIR-V (glslangValidator)"
        VERBATIM
    )
    set(AGR_HAVE_SHADER_COMPILER TRUE)
elseif(GLSLC)
    add_custom_command(
        OUTPUT ${AGR_SHADER_SPV}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${GLSLC} ${AGR_SHADER_SRC} -o ${AGR_SHADER_SPV}
        DEPENDS ${AGR_SHADER_SRC}
        COMMENT "Compiling vector_add.comp -> SPIR-V (glslc)"
        VERBATIM
    )
    set(AGR_HAVE_SHADER_COMPILER TRUE)
endif()

if(AGR_HAVE_SHADER_COMPILER)
    add_custom_target(agr_shaders ALL DEPENDS ${AGR_SHADER_SPV})
else()
    add_custom_target(agr_shaders)
    message(WARNING "Neither glslangValidator nor glslc was found. Phase 2's compute shader cannot be compiled -- install the Vulkan SDK (it includes both) and reconfigure.")
endif()

# =============================================================================
# Phase 3: llama.cpp external dependency (fetched at configure time)
# =============================================================================
# llama.cpp is fetched via CMake FetchContent at a pinned tag.
# GIT_SHALLOW TRUE keeps the download small (~150 MB instead of ~500 MB).
# We disable all extras we don't need to keep build times and disk usage down.
# GGML_VULKAN=ON enables the Vulkan compute backend inside llama.cpp/ggml.
#
# NOTE: First configure will download llama.cpp from GitHub.
#       Ensure internet access and ~2 GB free disk space.
# =============================================================================
include(FetchContent)

set(GGML_VULKAN          ON  CACHE BOOL "Enable Vulkan backend in ggml/llama.cpp" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "Disable CUDA (not present on target machine)" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "Force static libraries" FORCE)

FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggerganov/llama.cpp.git
    GIT_TAG        b9999
    GIT_SHALLOW    TRUE
    GIT_PROGRESS   TRUE
)

message(STATUS "Fetching llama.cpp b9999 (first run may take several minutes) ...")
FetchContent_MakeAvailable(llama_cpp)
message(STATUS "llama.cpp ready.")

# Suppress W4 warnings in llama.cpp code (we can't fix upstream warnings)
if(MSVC AND TARGET llama)
    target_compile_options(llama PRIVATE /W0)
endif()
if(MSVC AND TARGET ggml)
    target_compile_options(ggml PRIVATE /W0)
endif()

# =============================================================================
# agr_core — Phase 1 + Phase 2 (unchanged from c3fc2f2)
# =============================================================================
add_library(agr_core STATIC
    src/hardware/CPUDetector.cpp
    src/hardware/MemoryDetector.cpp
    src/hardware/GPUDetector.cpp
    src/hardware/VulkanDetector.cpp
    src/hardware/CUDADetector.cpp
    src/hardware/HardwareDetector.cpp
    src/json/HardwareJson.cpp
    src/gpu/CpuReference.cpp
    src/gpu/ShaderLocator.cpp
    src/gpu/VulkanBackend.cpp
    src/benchmark/VectorAddBenchmark.cpp
)

target_include_directories(agr_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Dev/test convenience: a compiled-in fallback path to the build tree's
# shaders directory, so unit tests (which run from wherever CTest puts
# them) can still find the shader without needing their own copy step.
# Production distribution instead ships the exe alongside a "shaders/"
# folder (see the post-build copy step below).
target_compile_definitions(agr_core PUBLIC AGR_SHADER_BUILD_DIR="${CMAKE_BINARY_DIR}/shaders")

if(Vulkan_FOUND)
    target_link_libraries(agr_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(agr_core PUBLIC AGR_HAVE_VULKAN=1)
    message(STATUS "Vulkan found: building with real Vulkan detection + Phase 2 compute enabled.")
else()
    target_compile_definitions(agr_core PUBLIC AGR_HAVE_VULKAN=0)
    message(WARNING "Vulkan SDK/loader not found. Building with Vulkan detection and Phase 2 compute stubbed to 'unavailable'. Install the Vulkan SDK (loader + headers) and reconfigure to enable real detection/compute.")
endif()

if(WIN32)
    target_link_libraries(agr_core PUBLIC dxgi)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(agr_core PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)
endif()

add_dependencies(agr_core agr_shaders)

# =============================================================================
# agr_llm — Phase 3: llama.cpp adapter
# =============================================================================
add_library(agr_llm STATIC
    src/llm/LlamaCppEngine.cpp
)
target_include_directories(agr_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# llama target exposes its own include dirs (llama.h, ggml.h) transitively
target_link_libraries(agr_llm PUBLIC agr_core llama)

# =============================================================================
# adaptive-gpu — main CLI executable (Phase 1, 2, 3)
# =============================================================================
add_executable(adaptive-gpu src/cli/main.cpp)
target_link_libraries(adaptive-gpu PRIVATE agr_core agr_llm)
add_dependencies(adaptive-gpu agr_shaders)

# Ship the compiled shader(s) alongside the executable for distribution
# (works for both single-config and multi-config/Visual-Studio builds).
add_custom_command(TARGET adaptive-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:adaptive-gpu>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGR_SHADER_SPV} $<TARGET_FILE_DIR:adaptive-gpu>/shaders/vector_add.comp.spv
)

enable_testing()
add_subdirectory(tests)
