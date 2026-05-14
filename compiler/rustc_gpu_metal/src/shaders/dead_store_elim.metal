#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint bitset_words;
    uint effects_stride;
};

#define EFFECT_NOP 0
#define EFFECT_KILL 1
#define EFFECT_GEN 2

kernel void dead_store_elim(
    const device uint* configs [[buffer(0)]],
    const device uint* effects [[buffer(1)]],
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
    
    // Read exit state for this block
    uint state_start = block_idx * pc.bitset_words;
    uint state[32]; // Max 1024 locals (32 * 32)
    
    for (uint w = 0; w < pc.bitset_words; w++) {
        state[w] = exit_states[state_start + w];
    }
    
    // Apply effects backward (from last statement to first)
    for (int stmt_idx = int(stmt_count) - 1; stmt_idx >= 0; stmt_idx--) {
        uint effect = effects[block_idx * pc.effects_stride + uint(stmt_idx)];
        uint kind = effect >> 24;
        uint local = effect & 0xFFFFFF;
        
        if (local >= pc.bitset_words * 32) continue;
        
        uint word = local / 32;
        uint bit = local % 32;
        uint mask = 1u << bit;
        
        if (kind == EFFECT_GEN) {
            state[word] |= mask;
        } else if (kind == EFFECT_KILL) {
            state[word] &= ~mask;
        }
    }
    
    // Write new entry state
    bool changed = false;
    for (uint w = 0; w < pc.bitset_words; w++) {
        uint old = entry_states[state_start + w];
        if (old != state[w]) {
            entry_states[state_start + w] = state[w];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
