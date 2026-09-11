function(agr_add_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE agr_core)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_dependencies(${test_name} agr_shaders)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Phase 1
agr_add_test(test_cpu_detector test_cpu_detector.cpp)
agr_add_test(test_memory_detector test_memory_detector.cpp)
agr_add_test(test_gpu_detector test_gpu_detector.cpp)
agr_add_test(test_vulkan_detector test_vulkan_detector.cpp)
agr_add_test(test_cuda_detector test_cuda_detector.cpp)
agr_add_test(test_json_serialization test_json_serialization.cpp)

# Phase 2
agr_add_test(test_cpu_reference_vector_add test_cpu_reference_vector_add.cpp)
agr_add_test(test_vulkan_backend_lifecycle test_vulkan_backend_lifecycle.cpp)
agr_add_test(test_vulkan_vector_add_correctness test_vulkan_vector_add_correctness.cpp)
agr_add_test(test_vulkan_buffer_transfer test_vulkan_buffer_transfer.cpp)
agr_add_test(test_vulkan_invalid_buffer_handling test_vulkan_invalid_buffer_handling.cpp)
agr_add_test(test_benchmark_formatting test_benchmark_formatting.cpp)
# Regression: descriptor pool must be reusable across repeated executeVectorAdd
# calls (catches the vkAllocateDescriptorSets failed bug fixed in Phase 2).
agr_add_test(test_vulkan_descriptor_reuse test_vulkan_descriptor_reuse.cpp)

# GPU-dependent tests report CTest's SKIP code (77) instead of a fake pass
# when no Vulkan compute device is present in the environment running them.
set_tests_properties(test_vulkan_vector_add_correctness test_vulkan_buffer_transfer
                     test_vulkan_descriptor_reuse
    PROPERTIES SKIP_RETURN_CODE 77)
