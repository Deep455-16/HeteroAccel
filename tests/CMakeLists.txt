function(agr_add_test test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} PRIVATE agr_scheduler agr_backend agr_llm agr_mem agr_core)
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
agr_add_test(test_vulkan_descriptor_reuse test_vulkan_descriptor_reuse.cpp)

# Phase 3
agr_add_test(test_llm_availability test_llm_availability.cpp)
agr_add_test(test_llm_model_path test_llm_model_path.cpp)
agr_add_test(test_llm_cpu_inference test_llm_cpu_inference.cpp)
agr_add_test(test_llm_vulkan_inference test_llm_vulkan_inference.cpp)
agr_add_test(test_llm_mode_distinction test_llm_mode_distinction.cpp)

# Phase 4 - Memory
agr_add_test(test_mem_cpu_alloc test_mem_cpu_alloc.cpp)
agr_add_test(test_mem_vulkan_alloc test_mem_vulkan_alloc.cpp)
agr_add_test(test_mem_cpu_to_gpu_transfer test_mem_cpu_to_gpu_transfer.cpp)
agr_add_test(test_mem_gpu_to_cpu_transfer test_mem_gpu_to_cpu_transfer.cpp)
agr_add_test(test_mem_statistics test_mem_statistics.cpp)
agr_add_test(test_mem_pool_reuse test_mem_pool_reuse.cpp)
agr_add_test(test_mem_lru_eviction test_mem_lru_eviction.cpp)
agr_add_test(test_mem_priority_eviction test_mem_priority_eviction.cpp)
agr_add_test(test_mem_invalid_ops test_mem_invalid_ops.cpp)

# Phase 4 - Automatic Backend Selection
agr_add_test(test_backend_cpu_detected test_backend_cpu_detected.cpp)
agr_add_test(test_backend_vulkan_detected test_backend_vulkan_detected.cpp)
agr_add_test(test_backend_cuda_graceful test_backend_cuda_graceful.cpp)
agr_add_test(test_backend_enumeration test_backend_enumeration.cpp)
agr_add_test(test_backend_auto_select test_backend_auto_select.cpp)
agr_add_test(test_backend_cpu_fallback test_backend_cpu_fallback.cpp)

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
    test_backend_vulkan_detected
    test_backend_auto_select
    PROPERTIES SKIP_RETURN_CODE 77
)

# Phase 5 - Adaptive Heterogeneous Scheduler
agr_add_test(test_sched_cpu_only test_sched_cpu_only.cpp)
agr_add_test(test_sched_cost_model test_sched_cost_model.cpp)
agr_add_test(test_sched_oom_rejection test_sched_oom_rejection.cpp)
agr_add_test(test_sched_residency_bias test_sched_residency_bias.cpp)
agr_add_test(test_sched_feedback_loop test_sched_feedback_loop.cpp)
agr_add_test(test_sched_fallback test_sched_fallback.cpp)
