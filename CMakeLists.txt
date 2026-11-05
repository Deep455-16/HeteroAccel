
cmake_minimum_required(VERSION 3.20)
project(AdaptiveGPURuntime VERSION 0.24.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Static libs are linked into the heteroaccel shared library; they need -fPIC on Linux.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra)
endif()

# --- Vulkan (Phase 1 detection + Phase 2 real compute) --------------------
# The Vulkan SDK/loader is expected to be installed on the system.
# SDKs are never bundled into the repository.
find_package(Vulkan)

# --- Phase 2: compile compute shaders from source at build time -----------
find_program(GLSLANG_VALIDATOR
    NAMES glslangValidator
    HINTS "D:/VulkanSDK/1.4.357.0/Bin"
)

if(NOT GLSLANG_VALIDATOR)
    find_program(GLSLC
        NAMES glslc
        HINTS "D:/VulkanSDK/1.4.357.0/Bin"
    )
endif()

set(AGR_ALL_SHADERS
    vector_add.comp
    hydroxl_matvec.comp
    hydroxl_rmsnorm.comp
    hydroxl_elemwise.comp
    hydroxl_softmax.comp
)

set(AGR_ALL_SHADER_SPVS "")
set(AGR_HAVE_SHADER_COMPILER FALSE)

foreach(shader IN LISTS AGR_ALL_SHADERS)
    set(shader_src "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}")
    set(shader_spv "${CMAKE_BINARY_DIR}/shaders/${shader}.spv")

    list(APPEND AGR_ALL_SHADER_SPVS "${shader_spv}")

    if(GLSLANG_VALIDATOR)
        add_custom_command(
            OUTPUT "${shader_spv}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_BINARY_DIR}/shaders"
            COMMAND "${GLSLANG_VALIDATOR}" -V
                "${shader_src}" -o "${shader_spv}"
            DEPENDS "${shader_src}"
            COMMENT "Compiling ${shader} -> SPIR-V (glslangValidator)"
            VERBATIM
        )
        set(AGR_HAVE_SHADER_COMPILER TRUE)

    elseif(GLSLC)
        add_custom_command(
            OUTPUT "${shader_spv}"
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "${CMAKE_BINARY_DIR}/shaders"
            COMMAND "${GLSLC}" "${shader_src}" -o "${shader_spv}"
            DEPENDS "${shader_src}"
            COMMENT "Compiling ${shader} -> SPIR-V (glslc)"
            VERBATIM
        )
        set(AGR_HAVE_SHADER_COMPILER TRUE)
    endif()
endforeach()

if(AGR_HAVE_SHADER_COMPILER)
    add_custom_target(agr_shaders ALL
        DEPENDS ${AGR_ALL_SHADER_SPVS}
    )
else()
    add_custom_target(agr_shaders)
    message(WARNING
        "Neither glslangValidator nor glslc was found. "
        "Compute shaders cannot be compiled. Install the Vulkan SDK "
        "and reconfigure."
    )
endif()

# =============================================================================
# Phase 3: llama.cpp external dependency
# =============================================================================

include(FetchContent)

set(GGML_VULKAN          ON  CACHE BOOL "Enable Vulkan backend in ggml/llama.cpp" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(GGML_CUDA            OFF CACHE BOOL "Disable CUDA" FORCE)
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

# Suppress upstream warnings in llama.cpp.
if(MSVC AND TARGET llama)
    target_compile_options(llama PRIVATE /W0)
endif()

if(MSVC AND TARGET ggml)
    target_compile_options(ggml PRIVATE /W0)
endif()

# =============================================================================
# agr_core - Phase 1 + Phase 2
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

target_include_directories(agr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(agr_core PUBLIC
    AGR_SHADER_BUILD_DIR="${CMAKE_BINARY_DIR}/shaders"
)

if(Vulkan_FOUND)
    target_link_libraries(agr_core PUBLIC Vulkan::Vulkan)
    target_compile_definitions(agr_core PUBLIC AGR_HAVE_VULKAN=1)

    message(STATUS
        "Vulkan found: building with real Vulkan detection + Phase 2 compute enabled."
    )
else()
    target_compile_definitions(agr_core PUBLIC AGR_HAVE_VULKAN=0)

    message(WARNING
        "Vulkan SDK/loader not found. Building with Vulkan detection "
        "and Phase 2 compute stubbed to unavailable."
    )
endif()

if(WIN32)
    target_link_libraries(agr_core PUBLIC dxgi)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(agr_core PUBLIC
        ${CMAKE_DL_LIBS}
        Threads::Threads
    )
endif()

add_dependencies(agr_core agr_shaders)

# =============================================================================
# agr_llm - Phase 3: llama.cpp adapter
# =============================================================================

add_library(agr_llm STATIC
    src/llm/LlamaCppEngine.cpp
)

target_include_directories(agr_llm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_llm PUBLIC
    agr_core
    llama
)

# =============================================================================
# agr_mem - Phase 4: Unified Memory Management
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

target_include_directories(agr_mem PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_mem PUBLIC agr_core)

# =============================================================================
# agr_backend - Phase 4: Automatic Backend Selection
# =============================================================================

add_library(agr_backend STATIC
    src/backend/BackendManager.cpp
    src/backend/DeviceSelector.cpp
    src/backend/CUDAAllocator.cpp
)

target_include_directories(agr_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_backend PUBLIC
    agr_core
    agr_mem
)

# =============================================================================
# agr_profiler - Phase 8: Unified Profiling System
# =============================================================================

add_library(agr_profiler STATIC
    src/profiler/Profiler.cpp
)

target_include_directories(agr_profiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_profiler PUBLIC
    agr_core
    agr_scheduler
)

# =============================================================================
# agr_scheduler - Phase 5 + 8: Adaptive Scheduler & Auto-Tuning
# =============================================================================

add_library(agr_scheduler STATIC
    src/scheduler/PerformanceHistory.cpp
    src/scheduler/CostModel.cpp
    src/scheduler/CPUWorker.cpp
    src/scheduler/VulkanWorker.cpp
    src/scheduler/CUDAWorker.cpp
    src/scheduler/AdaptiveScheduler.cpp
    src/scheduler/AutoTuner.cpp
    src/scheduler/WorkloadRegistry.cpp
    src/scheduler/ExecutionPolicyEngine.cpp
)

target_include_directories(agr_scheduler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_scheduler PUBLIC
    agr_core
    agr_mem
    agr_backend
)

# =============================================================================
# agr_model - Phase 6: Model / Layer Manager, Streaming & Prefetch
# =============================================================================

add_library(agr_model STATIC
    src/model/FileStorageBackend.cpp
    src/model/ResourceResidencyManager.cpp
    src/model/StreamingEngine.cpp
    src/model/LayerManager.cpp
    src/model/PrefetchEngine.cpp
    src/model/ModelManager.cpp
)

target_include_directories(agr_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_model PUBLIC
    agr_scheduler
    agr_mem
    agr_backend
)

# =============================================================================
# agr_residency - Phase 14: Real Model Residency & Streaming
# =============================================================================

add_library(agr_residency STATIC
    src/model/FileModelDataSource.cpp
    src/model/ModelResidencyManager.cpp
    src/model/HostAcceleratorMemory.cpp
    src/model/VulkanAcceleratorMemory.cpp
    src/model/GGUFRegionExtractor.cpp
    src/model/PlanResidencyAdapter.cpp
)

target_include_directories(agr_residency PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_residency PUBLIC
    agr_model
    agr_backend
    agr_core
    llama
)

# =============================================================================
# agr_primitives - Phase 16: Native HeteroAccel Execution Primitives
# =============================================================================

add_library(agr_primitives STATIC
    src/primitives/cpu/CpuOps.cpp
    src/execution/TensorComputeNode.cpp
)

target_include_directories(agr_primitives PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_primitives PUBLIC agr_core)

# =============================================================================
# agr_hydroxl - Phase 17: HydroXL Native Inference Engine
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
    src/hydroxl/VulkanTensorOps.cpp
)

target_include_directories(agr_hydroxl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_hydroxl PUBLIC
    agr_engine
    agr_core
)

add_dependencies(agr_hydroxl agr_shaders)

# =============================================================================
# agr_execution - Phase 15: Heterogeneous Execution Graph
# =============================================================================

add_library(agr_execution STATIC
    src/execution/ExecutionGraph.cpp
    src/execution/ExecutionGraphScheduler.cpp
    src/execution/ExecutionGraphBuilder.cpp
)

target_include_directories(agr_execution PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_execution PUBLIC
    agr_residency
    agr_engine
    agr_analysis
    agr_primitives
    agr_core
)

# =============================================================================
# agr_inference - Phase 7 + 11: HeteroRuntime API
# =============================================================================

add_library(agr_inference STATIC
    src/inference/HeteroRuntime.cpp
)

target_include_directories(agr_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_inference PUBLIC
    agr_model
    agr_residency
    agr_execution
    agr_llm
    agr_analysis
    agr_engine
    agr_scheduler
    agr_mem
    agr_backend
    agr_core
    llama
    agr_profiler
)

# =============================================================================
# agr_engine - Phase 11: Universal Execution Engine Interface
# =============================================================================

add_library(agr_engine STATIC
    src/engine/EngineCapability.cpp
    src/engine/ModelDescriptor.cpp
    src/engine/ExecutionEngineRegistry.cpp
    src/engine/LlamaCppExecutionEngine.cpp
    src/inference/LlamaCppBackend.cpp
)

target_include_directories(agr_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_engine PUBLIC
    agr_llm
    agr_core
)

if(GGML_CUDA)
    target_compile_definitions(agr_engine PUBLIC AGR_GGML_CUDA=1)
else()
    target_compile_definitions(agr_engine PUBLIC AGR_GGML_CUDA=0)
endif()

# =============================================================================
# agr_analysis - Phase 12: Model Inspection and Capability Analysis
# =============================================================================

add_library(agr_analysis STATIC
    src/analysis/CapabilityReport.cpp
    src/analysis/GGUFInspector.cpp
    src/analysis/CapabilityAnalyzer.cpp
    src/planner/ExecutionPlanner.cpp
)

target_include_directories(agr_analysis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(agr_analysis PUBLIC
    agr_engine
    agr_backend
    agr_core
    llama
)

# =============================================================================
# adaptive-gpu - Main CLI executable
# =============================================================================

add_executable(adaptive-gpu
    src/cli/main.cpp
)

target_link_libraries(adaptive-gpu PRIVATE
    agr_core
    agr_llm
    agr_analysis
    agr_residency
    agr_execution
    agr_engine
    agr_mem
    agr_backend
    agr_scheduler
    agr_model
    agr_inference
    agr_profiler
    agr_primitives
    agr_hydroxl
    agr_orchestration
)

add_dependencies(adaptive-gpu agr_shaders)

# Copy compiled shaders beside the executable.
add_custom_command(
    TARGET adaptive-gpu
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/shaders/vector_add.comp.spv"
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders/vector_add.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/shaders/hydroxl_matvec.comp.spv"
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders/hydroxl_matvec.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/shaders/hydroxl_rmsnorm.comp.spv"
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders/hydroxl_rmsnorm.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/shaders/hydroxl_elemwise.comp.spv"
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders/hydroxl_elemwise.comp.spv"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_BINARY_DIR}/shaders/hydroxl_softmax.comp.spv"
        "$<TARGET_FILE_DIR:adaptive-gpu>/shaders/hydroxl_softmax.comp.spv"
    COMMENT "Copying compiled shaders beside adaptive-gpu"
    VERBATIM
)

# =============================================================================
# Phase 18: Dynamic Engine Plugin System
# =============================================================================

add_library(agr_plugin_abi INTERFACE
    src/plugin/PluginAbis.h
)

target_include_directories(agr_plugin_abi INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin
)

# --- Good test plugin ------------------------------------------------------

add_library(hydroxl_test_plugin SHARED
    test_plugin.cpp
)

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

if(MSVC)
    target_compile_definitions(hydroxl_test_plugin PRIVATE
        AGR_PLUGIN_BUILDING_DLL
    )
endif()

# Copy the plugin using its actual platform-specific filename.
add_custom_command(
    TARGET hydroxl_test_plugin
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:adaptive-gpu>/plugins"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:hydroxl_test_plugin>"
        "$<TARGET_FILE_DIR:adaptive-gpu>/plugins/$<TARGET_FILE_NAME:hydroxl_test_plugin>"
    COMMENT "Copying test plugin to executable plugin directory"
    VERBATIM
)

# Ensure the plugins directory exists beside the executable.
# TARGET-based POST_BUILD works with both single-config and multi-config
# generators, including Ninja, Makefiles and Visual Studio.

add_custom_command(
    TARGET adaptive-gpu
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:adaptive-gpu>/plugins"
    COMMENT "Creating executable plugin directory"
    VERBATIM
)

# =============================================================================
# Phase 19: Plugin Loader
# =============================================================================

set(INSTALLED_PLUGINS
    "hydroxl-test:hydroxl-test:1.0.0:LLM:HeteroAccel Project"
)

add_library(agr_plugin STATIC
    src/plugin/PluginLoader.cpp
)

target_include_directories(agr_plugin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# dlopen/dlsym requires libdl on Linux; CMAKE_DL_LIBS is empty on Windows/macOS
target_link_libraries(agr_plugin PUBLIC ${CMAKE_DL_LIBS})

# --- Negative-control plugin: mismatched ABI version ---------------------

add_library(test_plugin_badabi SHARED
    tests/test_plugin_badabi.cpp
)

set_target_properties(test_plugin_badabi PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_include_directories(test_plugin_badabi PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(MSVC)
    target_compile_definitions(test_plugin_badabi PRIVATE
        AGR_PLUGIN_BUILDING_DLL
    )
endif()

# Copy the negative-control plugin using its actual platform-specific filename.
add_custom_command(
    TARGET test_plugin_badabi
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:test_plugin_badabi>"
        "$<TARGET_FILE_DIR:hydroxl_test_plugin>/$<TARGET_FILE_NAME:test_plugin_badabi>"
    COMMENT "Copying bad-ABI plugin next to the good test plugin"
    VERBATIM
)

# =============================================================================
# Phase 21: Universal Hardware-Aware Engine Orchestration
# =============================================================================

file(GLOB ORCHESTRATION_SRCS
    src/orchestration/*.cpp
)

add_library(agr_orchestration STATIC
    ${ORCHESTRATION_SRCS}
)

target_include_directories(agr_orchestration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/orchestration
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hydroxl
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin
    ${CMAKE_CURRENT_SOURCE_DIR}/src/engine
)

target_link_libraries(agr_orchestration PUBLIC
    agr_hydroxl
    agr_core
    agr_plugin
)

# =============================================================================
# Phase 23: Public C API & Developer SDK
# =============================================================================

add_library(agr_sdk STATIC
    src/sdk/HeteroAccelSDK.cpp
)

target_include_directories(agr_sdk
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(agr_sdk PUBLIC
    HETEROACCEL_STATIC
)

target_link_libraries(agr_sdk PUBLIC
    agr_orchestration
    agr_hydroxl
    agr_core
)

add_dependencies(agr_sdk agr_shaders)

# Standalone SDK shared library for external consumers.
add_library(heteroaccel SHARED
    src/sdk/HeteroAccelSDK.cpp
)

target_include_directories(heteroaccel
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(heteroaccel PRIVATE
    HETEROACCEL_BUILDING_DLL
)

target_link_libraries(heteroaccel PRIVATE
    agr_orchestration
    agr_hydroxl
    agr_core
    agr_llm
    agr_engine
    agr_mem
    agr_backend
    agr_inference
    agr_profiler
    agr_primitives
    agr_execution
    agr_analysis
    agr_residency
    agr_scheduler
    agr_model
    agr_plugin
)

if(WIN32)
    target_link_libraries(heteroaccel PRIVATE dxgi)
endif()

add_dependencies(heteroaccel agr_shaders)

# =============================================================================
# Phase 24: Install & Packaging Rules
# =============================================================================

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(HeteroAccel::heteroaccel ALIAS heteroaccel)

install(TARGETS heteroaccel
    EXPORT HeteroAccelTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(TARGETS agr_sdk
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/heteroaccel
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT HeteroAccelTargets
    FILE HeteroAccelTargets.cmake
    NAMESPACE HeteroAccel::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeteroAccel
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeteroAccelConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/HeteroAccelConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeteroAccel
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/HeteroAccelConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/HeteroAccelConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/HeteroAccelConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/HeteroAccel
)

export(EXPORT HeteroAccelTargets
    FILE ${CMAKE_CURRENT_BINARY_DIR}/HeteroAccelTargets.cmake
    NAMESPACE HeteroAccel::
)

# Pkg-config configuration and installation.
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/heteroaccel.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/heteroaccel.pc
    @ONLY
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/heteroaccel.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

# =============================================================================
# Examples
# =============================================================================

add_executable(c_example
    examples/c_example.c
)

target_link_libraries(c_example PRIVATE agr_sdk)

set_source_files_properties(
    examples/c_example.c
    PROPERTIES LANGUAGE CXX
)

add_executable(cpp_example
    examples/cpp_example.cpp
)

target_link_libraries(cpp_example PRIVATE agr_sdk)

# =============================================================================
# Test suite registration
# =============================================================================

include(CTest)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
