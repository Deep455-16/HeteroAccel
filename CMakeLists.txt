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
# agr_core â€” Phase 1 + Phase 2 (unchanged from c3fc2f2)
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
# agr_llm â€” Phase 3: llama.cpp adapter
# =============================================================================
add_library(agr_llm STATIC
    src/llm/LlamaCppEngine.cpp
)
target_include_directories(agr_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
# llama target exposes its own include dirs (llama.h, ggml.h) transitively
target_link_libraries(agr_llm PUBLIC agr_core llama)

# =============================================================================
# agr_mem â€” Phase 4: Unified Memory Management
# =============================================================================
add_library(agr_mem STATIC
    src/mem/CPUAllocator.cpp
    src/mem/VulkanAllocator.cpp
    src/mem/MemoryPool.cpp
    src/mem/TransferManager.cpp
    src/mem/ResidencyManager.cpp
    src/mem/PressureMonitor.cpp
    src/mem/EvictionPolicy.cpp
    src/mem/MemoryManager.cpp
)
target_include_directories(agr_mem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_mem PUBLIC agr_core)

# =============================================================================
# agr_backend â€” Phase 4: Automatic Backend Selection
# CPU, Vulkan, and CUDA are discovered uniformly; the user never chooses.
# =============================================================================
add_library(agr_backend STATIC
    src/backend/BackendManager.cpp
    src/backend/DeviceSelector.cpp
    src/backend/CUDAAllocator.cpp
)
target_include_directories(agr_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_backend PUBLIC agr_core agr_mem)

# =============================================================================
# agr_profiler â€” Phase 8: Unified Profiling System
# =============================================================================
add_library(agr_profiler STATIC
    src/profiler/Profiler.cpp
)
target_include_directories(agr_profiler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_profiler PUBLIC agr_core agr_scheduler)

# =============================================================================
# agr_scheduler â€” Phase 5 + 8: Adaptive Scheduler & Auto-Tuning
# =============================================================================
add_library(agr_scheduler STATIC
    src/scheduler/PerformanceHistory.cpp
    src/scheduler/CostModel.cpp
    src/scheduler/CPUWorker.cpp
    src/scheduler/VulkanWorker.cpp
    src/scheduler/CUDAWorker.cpp
    src/scheduler/AdaptiveScheduler.cpp
    src/scheduler/AutoTuner.cpp
    # Phase 9
    src/scheduler/WorkloadRegistry.cpp
    src/scheduler/ExecutionPolicyEngine.cpp
)
target_include_directories(agr_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_scheduler PUBLIC agr_core agr_mem agr_backend)

# =============================================================================
# agr_model â€” Phase 6: Model / Layer Manager, Streaming & Prefetch
# =============================================================================
add_library(agr_model STATIC
    src/model/FileStorageBackend.cpp
    src/model/ResourceResidencyManager.cpp
    src/model/StreamingEngine.cpp
    src/model/LayerManager.cpp
    src/model/PrefetchEngine.cpp
    src/model/ModelManager.cpp
)
target_include_directories(agr_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_model PUBLIC agr_scheduler agr_mem agr_backend)

# =============================================================================
# agr_residency â€” Phase 14: Real Model Residency & Streaming
# =============================================================================
add_library(agr_residency STATIC
    src/model/FileModelDataSource.cpp
    src/model/ModelResidencyManager.cpp
    src/model/HostAcceleratorMemory.cpp
    src/model/VulkanAcceleratorMemory.cpp
    src/model/GGUFRegionExtractor.cpp
    src/model/PlanResidencyAdapter.cpp
)
target_include_directories(agr_residency PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_residency PUBLIC agr_model agr_backend agr_core llama)

# =============================================================================
# agr_primitives â€” Phase 16: Native HeteroAccel Execution Primitives
# Tensor types, HostBuffer, CPU ops, IBackendExecutor, CpuBackendExecutor.
# =============================================================================
add_library(agr_primitives STATIC
    src/primitives/cpu/CpuOps.cpp
    src/execution/TensorComputeNode.cpp
)
target_include_directories(agr_primitives PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_primitives PUBLIC agr_core)

# =============================================================================
# agr_hydroxl â€” Phase 17: HydroXL Native Inference Engine
# Independent native engine: GGUF loader, Q4_K dequant, BPE tokenizer,
# Qwen2 forward pass, KV cache, sampler, and IExecutionEngine adapter.
# Does NOT link against llama at runtime; uses agr_engine for interfaces only.
# =============================================================================
add_library(agr_hydroxl STATIC
    src/hydroxl/GGUFLoader.cpp
    src/hydroxl/HydroXLDequant.cpp
    src/hydroxl/HydroXLOps.cpp
    src/hydroxl/HydroXLKVCache.cpp
    src/hydroxl/HydroXLSampler.cpp
    src/hydroxl/HydroXLTokenizer.cpp
    src/hydroxl/HydroXLModel.cpp
    src/hydroxl/HydroXLForwardPass.cpp
    src/hydroxl/HydroXLEngine.cpp
)
target_include_directories(agr_hydroxl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_hydroxl PUBLIC agr_engine agr_core)

# =============================================================================
# agr_execution â€” Phase 15: Heterogeneous Execution Graph
# =============================================================================
add_library(agr_execution STATIC
    src/execution/ExecutionGraph.cpp
    src/execution/ExecutionGraphScheduler.cpp
    src/execution/ExecutionGraphBuilder.cpp
)
target_include_directories(agr_execution PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_execution PUBLIC agr_residency agr_engine agr_analysis agr_primitives agr_core)


# =============================================================================
# agr_inference â€” Phase 7+11: HeteroRuntime API
# Note: LlamaCppBackend.cpp is now compiled in agr_engine to avoid circular deps.
# =============================================================================
add_library(agr_inference STATIC
    src/inference/HeteroRuntime.cpp
)
target_include_directories(agr_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_inference PUBLIC agr_model agr_residency agr_execution agr_llm agr_analysis agr_engine agr_scheduler agr_mem agr_backend agr_core llama agr_profiler)

# =============================================================================
# agr_engine â€” Phase 11: Universal Execution Engine Interface
# LlamaCppBackend.cpp is compiled here so that agr_engine owns the full
# llama.cppâ†’IExecutionEngine chain without depending on agr_inference.
# =============================================================================
add_library(agr_engine STATIC
    src/engine/EngineCapability.cpp
    src/engine/ModelDescriptor.cpp
    src/engine/ExecutionEngineRegistry.cpp
    src/engine/LlamaCppExecutionEngine.cpp
    src/inference/LlamaCppBackend.cpp
)
target_include_directories(agr_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_engine PUBLIC agr_llm agr_core)
# Capability bits must match the llama.cpp/ggml build. GGML_CUDA is forced
# OFF in this project unless a CUDA build is explicitly configured.
if(GGML_CUDA)
    target_compile_definitions(agr_engine PUBLIC AGR_GGML_CUDA=1)
else()
    target_compile_definitions(agr_engine PUBLIC AGR_GGML_CUDA=0)
endif()

# =============================================================================
# agr_analysis â€” Phase 12: Model Inspection and Capability Analysis
# =============================================================================
add_library(agr_analysis STATIC
    src/analysis/CapabilityReport.cpp
    src/analysis/GGUFInspector.cpp
    src/analysis/CapabilityAnalyzer.cpp
    src/planner/ExecutionPlanner.cpp
)
target_include_directories(agr_analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(agr_analysis PUBLIC agr_engine agr_backend agr_core llama)

# =============================================================================
# adaptive-gpu â€” main CLI executable (Phase 1..13)
# =============================================================================
add_executable(adaptive-gpu src/cli/main.cpp)
target_link_libraries(adaptive-gpu PRIVATE agr_core agr_llm agr_analysis agr_residency agr_execution agr_engine agr_mem agr_backend agr_scheduler agr_model agr_inference agr_profiler agr_primitives agr_hydroxl)
add_dependencies(adaptive-gpu agr_shaders)

# Ship the compiled shader(s) alongside the executable for distribution
# (works for both single-config and multi-config/Visual-Studio builds).
add_custom_command(TARGET adaptive-gpu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:adaptive-gpu>/shaders
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${AGR_SHADER_SPV} $<TARGET_FILE_DIR:adaptive-gpu>/shaders/vector_add.comp.spv
)

# =============================================================================
# Phase 18: Dynamic Engine Plugin System
# =============================================================================
# Plugin ABI header and test plugin targets.
# The plugin SDK (PluginSdk.h/PluginAbis.h) is header-only; the actual
# plugin DLLs are built as separate shared-library targets and installed
# into the documented plugin directory beside the executable.

# --- Plugin ABI header (header-only, included by host and plugins) ---
add_library(agr_plugin_abi INTERFACE src/plugin/PluginAbis.h)
target_include_directories(agr_plugin_abi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin)

# --- Test plugin: independently compiled plugin demonstrating the ABI ---
# This plugin is built as a CDLL (Windows .dll) that implements the full
# plugin ABI. It is completely independent of the host executable.
# The CMake target 'hydroxl_test_plugin' builds the plugin DLL.
#
# Build requirements:
#   - Define AGR_PLUGIN_BUILDING_DLL before including PluginAbis.h
#   - Target the same platform as the host (Windows x64 / MSVC)
#   - Link against no internal HeteroAccel libraries (standalone DLL)
add_library(hydroxl_test_plugin SHARED test_plugin.cpp)
set_target_properties(hydroxl_test_plugin PROPERTIES
    WIN32 TRUE
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    VS_DEBUG_RUNTIME_LOOKUP TRUE
    VS_DEBUG_INFO INHERIT
)
target_include_directories(hydroxl_test_plugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Export the plugin DLL functions with the correct calling convention
# (the __stdcall calling convention is already in the ABI header)
if(MSVC)
    target_compile_definitions(hydroxl_test_plugin PRIVATE
        "AGR_PLUGIN_BUILDING_DLL"
    )
    # Use the MSVC C runtime DLL (CRTDLL) - no forced LIBCMT to avoid
    # ABI mismatch between host and plugin
endif()

# Post-build step: copy the plugin DLL to the executable's plugin directory.
# The host's PluginLoader scans ./plugins relative to the executable by default.
# We copy here so 'plugin_discover()' can find it without environment variable setup.
# Use generator expression to copy to the directory where the executable resides.
add_custom_command(TARGET hydroxl_test_plugin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:adaptive-gpu>/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:hydroxl_test_plugin>"
        "$<TARGET_FILE_DIR:adaptive-gpu>/plugins/hydroxl-test.dll"
    COMMENT "Copy test plugin to executable plugin directory"
    VERBATIM
)

# Ensure the plugins directory exists relative to the executable at
# every build configuration (single-config and multi-config).
# We use a lightweight guard so the directory is created once.
if(NOT CMAKE_CONFIGURATION_TYPES)
    # Single-config generators (Makefiles, Ninja)
    add_custom_command(
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            $<TARGET_FILE_DIR:adaptive-gpu>/plugins
        VERBATIM
    )
else()
    # Multi-config generators (Visual Studio): create per-config
    foreach(cfg IN ITEMS ${CMAKE_CONFIGURATION_TYPES})
        add_custom_command(
            TARGET adaptive-gpu POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory
                $<TARGET_FILE_DIR:adaptive-gpu>/plugins
            VERBATIM
        )
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Plugin discovery integration
# ---------------------------------------------------------------------------
# The host's PluginLoader (PluginLoader.cpp) scans the directory specified by
# AGR_PLUGIN_DIR env var, defaulting to "./plugins" relative to the executable.
# The installed plugin (hydroxl-test.dll) placed above will be found automatically.

# Registry of installed plugins (for CMake introspection only; the runtime
# discovers plugins dynamically at runtime).
set(INSTALLED_PLUGINS
    "hydroxl-test:hydroxl-test:1.0.0:LLM:HeteroAccel Project"
)

# ---------------------------------------------------------------------------
# Integration with ExecutionEngineRegistry
# ---------------------------------------------------------------------------
# At host startup, the PluginLoader will call ExecutionEngineRegistry::registerFactory
# for each discovered plugin that reports a compatible registry_key.
# The built-in "llama.cpp" registration is preserved; plugin registrations
# are added dynamically.

# The host plugin initialization sequence (simplified):
#   1. PluginLoader::plugin_discover() â€” scan ./plugins or AGR_PLUGIN_DIR
#   2. For each loadable plugin, fetch identity + capabilities
#   3. ExecutionEngineRegistry::registerFactory(key, description, factory)
#   4. Engine can be selected via EngineCapabilitySet filtering

# The existing ExecutionEngineRegistry registration for "llama.cpp" remains
# untouched; new plugins are added alongside it.

# ---------------------------------------------------------------------------
# Plugin loader (host side, Phase 19)
# ---------------------------------------------------------------------------
# Compiled from src/plugin/PluginLoader.cpp into its own static library.
# A loaded plugin is NOT an IExecutionEngine: the loader validates ABI
# version, exports and identity, and the host invokes plugin entry points
# through the plugin_execute() API. Native plugins run in-process without
# sandboxing (see PluginLoader.h).
add_library(agr_plugin STATIC src/plugin/PluginLoader.cpp)
target_include_directories(agr_plugin PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

# --- Negative-control plugin: reports a mismatched ABI version ---
add_library(test_plugin_badabi SHARED tests/test_plugin_badabi.cpp)
set_target_properties(test_plugin_badabi PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
target_include_directories(test_plugin_badabi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(MSVC)
    target_compile_definitions(test_plugin_badabi PRIVATE "AGR_PLUGIN_BUILDING_DLL")
endif()
# Place it next to the good plugin so the plugin tests can discover both.
add_custom_command(TARGET test_plugin_badabi POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_plugin_badabi>"
        "$<TARGET_FILE_DIR:hydroxl_test_plugin>/test_plugin_badabi.dll"
    COMMENT "Copy bad-ABI plugin next to the good test plugin"
    VERBATIM
)

# ---------------------------------------------------------------------------
# Test suite registration
# ---------------------------------------------------------------------------
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
