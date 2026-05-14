#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint bitset_words;
    uint facts_stride;
};

kernel void reaching_defs(
    const device uint* configs [[buffer(0)]],
    const device uint* def_facts [[buffer(1)]],
    device uint* entry_states [[buffer(2)]],
    device uint* exit_states [[buffer(3)]],
    device atomic_uint* convergence [[buffer(4)]],
    constant PushConstants& pc [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint config_base = block_idx * 4;
    uint stmt_count = configs[config_base];
    
    // Read entry state for this block
    uint state_start = block_idx * pc.bitset_words;
    uint state[32]; // Max 1024 definitions (32 * 32)
    
    uint max_words = min(pc.bitset_words, 32u);
    for (uint w = 0; w < max_words; w++) {
        state[w] = entry_states[state_start + w];
    }
    
    // Track which locals have been defined in this block (for killing)
    uint defined_locals[128]; // local_idx -> def_id (0xFFFFFFFF = not defined yet)
    for (uint i = 0; i < 128; i++) {
        defined_locals[i] = 0xFFFFFFFF;
    }
    
    // Apply definition facts forward through statements
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint fact = def_facts[block_idx * pc.facts_stride + stmt_idx];
        if (fact != 0xFFFFFFFF) {
            uint local = (fact >> 16) & 0xFFFF;
            uint def_id = fact & 0xFFFF;
            
            if (local < 128) {
                // Kill previous definition of this local
                uint old_def = defined_locals[local];
                if (old_def != 0xFFFFFFFF && old_def < 1024) {
                    uint word = old_def / 32;
                    uint bit = old_def % 32;
                    if (word < max_words) {
                        state[word] &= ~(1u << bit);
                    }
                }
                
                // Add new definition
                uint word = def_id / 32;
                uint bit = def_id % 32;
                if (word < max_words) {
                    state[word] |= (1u << bit);
                }
                defined_locals[local] = def_id;
            }
        }
    }
    
    // Write exit state
    bool changed = false;
    for (uint w = 0; w < max_words; w++) {
        uint old = exit_states[state_start + w];
        if (old != state[w]) {
            exit_states[state_start + w] = state[w];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
