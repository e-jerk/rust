#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint matrix_words;
};

kernel void loop_detect(
    const device uint* block_info [[buffer(0)]],
    device uint* reachability [[buffer(1)]],
    device uint* loop_headers [[buffer(2)]],
    constant PushConstants& pc [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint info_offset = block_idx * 5;
    uint num_succs = block_info[info_offset];
    
    // Initialize reachability with direct edges
    uint row_start = block_idx * pc.matrix_words;
    for (uint w = 0; w < pc.matrix_words; w++) {
        reachability[row_start + w] = 0;
    }
    
    // Set direct successor bits
    for (uint s = 0; s < num_succs && s < 4; s++) {
        uint succ = block_info[info_offset + 1 + s];
        if (succ < pc.num_blocks) {
            uint word = succ / 32;
            uint bit = succ % 32;
            if (word < pc.matrix_words) {
                reachability[row_start + word] |= (1u << bit);
            }
        }
    }
    
    // Iteratively compute transitive closure
    uint max_rounds = pc.num_blocks;
    
    for (uint round = 0; round < max_rounds; round++) {
        uint new_row[32];
        for (uint w = 0; w < pc.matrix_words; w++) {
            new_row[w] = reachability[row_start + w];
        }
        
        for (uint j = 0; j < pc.num_blocks; j++) {
            uint j_word = j / 32;
            uint j_bit = j % 32;
            if (j_word < pc.matrix_words) {
                uint bit_val = (reachability[row_start + j_word] >> j_bit) & 1u;
                if (bit_val != 0u) {
                    uint j_row_start = j * pc.matrix_words;
                    for (uint w = 0; w < pc.matrix_words; w++) {
                        new_row[w] |= reachability[j_row_start + w];
                    }
                }
            }
        }
        
        // Write back
        bool changed = false;
        for (uint w = 0; w < pc.matrix_words; w++) {
            uint old = reachability[row_start + w];
            if (old != new_row[w]) {
                reachability[row_start + w] = new_row[w];
                changed = true;
            }
        }
        
        if (!changed) {
            break;
        }
    }
    
    // Detect loop headers
    uint self_word = block_idx / 32;
    uint self_bit = block_idx % 32;
    bool in_cycle = false;
    if (self_word < pc.matrix_words) {
        in_cycle = ((reachability[row_start + self_word] >> self_bit) & 1u) != 0u;
    }
    
    if (in_cycle) {
        uint num_preds = 0;
        for (uint b = 0; b < pc.num_blocks; b++) {
            uint b_info_offset = b * 5;
            uint b_num_succs = block_info[b_info_offset];
            for (uint s = 0; s < b_num_succs && s < 4; s++) {
                uint succ = block_info[b_info_offset + 1 + s];
                if (succ == block_idx) {
                    num_preds++;
                }
            }
        }
        
        if (num_preds > 1) {
            atomic_fetch_or_explicit(&loop_headers[block_idx / 32], 1u << (block_idx % 32), memory_order_relaxed);
        }
    }
}
