#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint block_bitmap_words;
};

void intersect_preds(uint block_idx, uint out_set[32],
    const device uint* block_info,
    device uint* dominator_sets,
    uint block_bitmap_words,
    uint num_blocks) {
    uint info_offset = block_idx * 5;
    uint num_preds = block_info[info_offset];
    
    if (num_preds == 0) {
        return;
    }
    
    uint max_words = min(block_bitmap_words, 32u);
    
    // Initialize with all ones
    for (uint w = 0; w < max_words; w++) {
        out_set[w] = 0xFFFFFFFF;
    }
    
    // Intersect with each predecessor's dominator set
    for (uint p = 0; p < num_preds && p < 4; p++) {
        uint pred = block_info[info_offset + 1 + p];
        if (pred >= num_blocks) continue;
        
        uint pred_start = pred * block_bitmap_words;
        for (uint w = 0; w < max_words; w++) {
            out_set[w] &= dominator_sets[pred_start + w];
        }
    }
}

kernel void dominance(
    const device uint* block_info [[buffer(0)]],
    device uint* dominator_sets [[buffer(1)]],
    device atomic_uint* convergence [[buffer(2)]],
    constant PushConstants& pc [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    // Skip the entry block (block 0)
    if (block_idx == 0) return;
    
    uint info_offset = block_idx * 5;
    uint num_preds = block_info[info_offset];
    
    // If no predecessors, this block is unreachable
    if (num_preds == 0) {
        return;
    }
    
    uint max_words = min(pc.block_bitmap_words, 32u);
    
    // Compute new dominator set: {block_idx} U (intersection preds' dominator sets)
    uint new_set[32];
    intersect_preds(block_idx, new_set, block_info, dominator_sets, pc.block_bitmap_words, pc.num_blocks);
    
    // Add this block to its own dominator set
    uint word = block_idx / 32;
    uint bit = block_idx % 32;
    if (word < max_words) {
        new_set[word] |= (1u << bit);
    }
    
    // Check if dominator set changed
    uint set_start = block_idx * pc.block_bitmap_words;
    bool changed = false;
    for (uint w = 0; w < max_words; w++) {
        uint old = dominator_sets[set_start + w];
        if (old != new_set[w]) {
            dominator_sets[set_start + w] = new_set[w];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
