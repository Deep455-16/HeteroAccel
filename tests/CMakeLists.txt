function(agr_add_test test_name source_file)
    add_executable(${test_name} ${source_file})
    # For Phase 3, we link agr_llm which brings in agr_core and llama
    target_link_libraries(${test_name} PRIVATE agr_llm agr_mem agr_core)
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

# Phase 3
agr_add_test(test_llm_availability test_llm_availability.cpp)
agr_add_test(test_llm_model_path test_llm_model_path.cpp)
agr_add_test(test_llm_cpu_inference test_llm_cpu_inference.cpp)
agr_add_test(test_llm_vulkan_inference test_llm_vulkan_inference.cpp)
agr_add_test(test_llm_mode_distinction test_llm_mode_distinction.cpp)

# GPU-dependent tests report CTest's SKIP code (77) instead of a fake pass
# when no Vulkan compute device is present in the environment running them.
# Phase 3 model-dependent tests also skip (77) if HETEROACCEL_MODEL_PATH is missing.

# Phase 4
agr_add_test(test_mem_cpu_alloc test_mem_cpu_alloc.cpp)
agr_add_test(test_mem_vulkan_alloc test_mem_vulkan_alloc.cpp)
agr_add_test(test_mem_cpu_to_gpu_transfer test_mem_cpu_to_gpu_transfer.cpp)
agr_add_test(test_mem_gpu_to_cpu_transfer test_mem_gpu_to_cpu_transfer.cpp)
agr_add_test(test_mem_statistics test_mem_statistics.cpp)
agr_add_test(test_mem_pool_reuse test_mem_pool_reuse.cpp)
agr_add_test(test_mem_lru_eviction test_mem_lru_eviction.cpp)
agr_add_test(test_mem_priority_eviction test_mem_priority_eviction.cpp)
agr_add_test(test_mem_invalid_ops test_mem_invalid_ops.cpp)
set_tests_properties(
    test_vulkan_vector_add_correctness 
    test_vulkan_buffer_transfer
    test_vulkan_descriptor_reuse
    test_llm_cpu_inference
    test_llm_vulkan_inference
    test_llm_mode_distinction
        test_mem_vulkan_alloc
    test_mem_cpu_to_gpu_transfer
    test_mem_gpu_to_cpu_transfer
    PROPERTIES SKIP_RETURN_CODE 77
)

