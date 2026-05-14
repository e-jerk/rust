#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_functions;
    uint blocks_per_workgroup;
    uint max_bitset_words;
};

#define EFFECT_NOP 0
#define EFFECT_KILL 1
#define EFFECT_GEN 2

kernel void mega_batch_dataflow(
    const device uint* function_meta [[buffer(0)]],
    const device uint* all_configs [[buffer(1)]],
    const device uint* all_effects [[buffer(2)]],
    device uint* all_entry_states [[buffer(3)]],
    device uint* all_exit_states [[buffer(4)]],
    device atomic_uint* convergence [[buffer(5)]],
    constant PushConstants& pc [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint global_idx = thread_position_in_grid.x;
    
    // Determine which function and which block this thread handles
    uint func_idx = global_idx / pc.blocks_per_workgroup;
    uint block_idx = global_idx % pc.blocks_per_workgroup;
    
    if (func_idx >= pc.num_functions) return;
    
    // Read function metadata
    uint meta_offset = func_idx * 4;
    uint num_blocks = function_meta[meta_offset];
    uint num_locals = function_meta[meta_offset + 1];
    uint effects_stride = function_meta[meta_offset + 2];
    uint data_offset = function_meta[meta_offset + 3];
    
    if (block_idx >= num_blocks) return;
    
    uint bitset_words = (num_locals + 31) / 32;
    
    // Read block config
    uint config_offset = data_offset + block_idx * 4;
    uint stmt_count = all_configs[config_offset];
    
    // Read entry state for this block
    uint state_start = data_offset + block_idx * bitset_words;
    uint state[32]; // Max 1024 locals
    
    uint max_words = min(bitset_words, pc.max_bitset_words);
    for (uint w = 0; w < max_words; w++) {
        state[w] = all_entry_states[state_start + w];
    }
    
    // Apply effects forward through statements
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint effect = all_effects[data_offset + block_idx * effects_stride + stmt_idx];
        uint kind = effect >> 24;
        uint local = effect & 0xFFFFFF;
        
        if (local >= num_locals) continue;
        
        uint word = local / 32;
        uint bit = local % 32;
        uint mask = 1u << bit;
        
        if (kind == EFFECT_GEN) {
            state[word] |= mask;
        } else if (kind == EFFECT_KILL) {
            state[word] &= ~mask;
        }
    }
    
    // Write exit state
    bool changed = false;
    for (uint w = 0; w < max_words; w++) {
        uint old = all_exit_states[state_start + w];
        if (old != state[w]) {
            all_exit_states[state_start + w] = state[w];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[func_idx], 1u, memory_order_relaxed);
    }
}
