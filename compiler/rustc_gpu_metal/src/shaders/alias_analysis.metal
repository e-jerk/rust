#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_accesses;
    uint num_locals;
};

kernel void alias_analysis(
    const device uint* accesses [[buffer(0)]],
    device atomic_uint* alias_matrix [[buffer(1)]],
    constant PushConstants& pc [[buffer(2)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_accesses) return;
    
    uint access_i = accesses[idx];
    uint local_i = (access_i >> 16) & 0xFFFF;
    
    // Compare with all accesses j >= idx (symmetric matrix)
    for (uint j = idx; j < pc.num_accesses; j++) {
        uint access_j = accesses[j];
        uint local_j = (access_j >> 16) & 0xFFFF;
        
        bool may_alias = false;
        
        if (local_i == local_j && local_i != 0xFFFF) {
            may_alias = true;
        }
        
        if (may_alias) {
            uint matrix_size = (pc.num_accesses + 31) / 32;
            uint word_idx = idx * matrix_size + (j / 32);
            uint bit_idx = j % 32;
            atomic_fetch_or_explicit(&alias_matrix[word_idx], 1u << bit_idx, memory_order_relaxed);
            
            if (idx != j) {
                uint word_idx2 = j * matrix_size + (idx / 32);
                uint bit_idx2 = idx % 32;
                atomic_fetch_or_explicit(&alias_matrix[word_idx2], 1u << bit_idx2, memory_order_relaxed);
            }
        }
    }
}
