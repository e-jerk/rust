#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint statements_per_block;
    uint hash_table_size;
};

kernel void gvn(
    const device uint* expr_hashes [[buffer(0)]],
    device uint* value_numbers [[buffer(1)]],
    device atomic_uint* convergence [[buffer(2)]],
    constant PushConstants& pc [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint base = block_idx * pc.statements_per_block;
    bool changed = false;
    
    // For each statement in the block
    for (uint stmt_idx = 0; stmt_idx < pc.statements_per_block; stmt_idx++) {
        uint hash = expr_hashes[base + stmt_idx];
        if (hash == 0) continue;
        
        uint old_vn = value_numbers[base + stmt_idx];
        uint new_vn = hash;
        
        if (old_vn != new_vn) {
            value_numbers[base + stmt_idx] = new_vn;
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
