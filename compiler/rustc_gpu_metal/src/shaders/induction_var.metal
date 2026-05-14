#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint num_locals;
    uint max_stmts_per_block;
};

kernel void induction_var(
    const device uint* block_info [[buffer(0)]],
    device uint* induction_vars [[buffer(1)]],
    constant PushConstants& pc [[buffer(2)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint info_offset = block_idx * (2 + pc.max_stmts_per_block * 2);
    bool is_loop_header = block_info[info_offset] != 0;
    
    if (!is_loop_header) return;
    
    uint num_stmts = block_info[info_offset + 1];
    
    // Track which locals are modified and how
    uint modified_locals[128];
    uint local_kinds[128];
    for (uint i = 0; i < 128; i++) {
        modified_locals[i] = 0;
        local_kinds[i] = 0;
    }
    
    // Analyze statements in this block
    for (uint s = 0; s < num_stmts && s < pc.max_stmts_per_block; s++) {
        uint stmt_offset = info_offset + 2 + s * 2;
        uint local = block_info[stmt_offset];
        uint kind = block_info[stmt_offset + 1];
        
        if (local < 128) {
            modified_locals[local]++;
            local_kinds[local] = kind;
        }
    }
    
    // Mark induction variables: modified exactly once with add/sub constant
    for (uint local = 0; local < pc.num_locals && local < 128; local++) {
        if (modified_locals[local] == 1 && (local_kinds[local] == 1 || local_kinds[local] == 2)) {
            uint word = local / 32;
            uint bit = local % 32;
            if (word < (pc.num_locals + 31) / 32) {
                atomic_fetch_or_explicit(&induction_vars[block_idx * ((pc.num_locals + 31) / 32) + word], 1u << bit, memory_order_relaxed);
            }
        }
    }
}
