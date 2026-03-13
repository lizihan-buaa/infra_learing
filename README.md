# cuda_study
1.matmul:
    v0:naive
    v1:global_memory_coalescing
    v2:shared_memory_tiling
    v3:multi_results_per_thread; register_tiling
    v4:vectorize_memory_accesses; padding_no_bank_conflict
    v5:double_buffering
    v6:use_tensor_core