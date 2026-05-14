#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint num_locals;
    uint max_defs_per_block;
};

kernel void ssa_construct(
    const device uint* block_info [[buffer(0)]],
    const device uint* def_sites [[buffer(1)]],
    device atomic_uint* dom_frontier [[buffer(2)]],
    device atomic_uint* phi_nodes [[buffer(3)]],
    constant PushConstants& pc [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint info_offset = block_idx * (2 + pc.max_defs_per_block);
    uint num_preds = block_info[info_offset];
    uint num_defs = block_info[info_offset + 1];
    
    // For each predecessor, check if any of their defs reach this block
    // and need phi nodes
    for (uint p = 0; p < num_preds; p++) {
        uint pred_idx = block_info[info_offset + 2 + p];
        if (pred_idx >= pc.num_blocks) continue;
        
        uint pred_info_offset = pred_idx * (2 + pc.max_defs_per_block);
        uint pred_num_defs = block_info[pred_info_offset + 1];
        
        for (uint d = 0; d < pred_num_defs; d++) {
            uint local = def_sites[pred_idx * pc.max_defs_per_block + d];
            if (local >= pc.num_locals) continue;
            
            // Check if this block already defines this local
            bool already_defined = false;
            for (uint my_d = 0; my_d < num_defs; my_d++) {
                if (def_sites[block_idx * pc.max_defs_per_block + my_d] == local) {
                    already_defined = true;
                    break;
                }
            }
            
            if (!already_defined) {
                // This local needs a phi node in this block
                uint word = local / 32;
                uint bit = local % 32;
                uint phi_word = block_idx * ((pc.num_locals + 31) / 32) + word;
                if (phi_word < pc.num_blocks * ((pc.num_locals + 31) / 32)) {
                    atomic_fetch_or_explicit(&phi_nodes[phi_word], 1u << bit, memory_order_relaxed);
                }
            }
        }
    }
    
    // Mark this block in the dominance frontier of its predecessors
    if (num_preds > 1) {
        for (uint p = 0; p < num_preds; p++) {
            uint pred_idx = block_info[info_offset + 2 + p];
            if (pred_idx >= pc.num_blocks) continue;
            uint df_word = pred_idx * ((pc.num_blocks + 31) / 32) + (block_idx / 32);
            uint df_bit = block_idx % 32;
            atomic_fetch_or_explicit(&dom_frontier[df_word], 1u << df_bit, memory_order_relaxed);
        }
    }
}
