#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint num_locals;
    uint effects_stride;
};

#define EFFECT_INIT 1
#define EFFECT_MOVE 2
#define EFFECT_BORROW_MUT 3
#define EFFECT_BORROW_SHARED 4
#define EFFECT_DROP 5

kernel void borrow_check(
    const device uint* configs [[buffer(0)]],
    const device uint* effects [[buffer(1)]],
    device uint* live_states [[buffer(2)]],
    device uint* moved_states [[buffer(3)]],
    device uint* init_states [[buffer(4)]],
    device atomic_uint* convergence [[buffer(5)]],
    constant PushConstants& pc [[buffer(6)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint config_base = block_idx * 4;
    uint stmt_count = configs[config_base];
    
    uint words = (pc.num_locals + 31) / 32;
    uint state_start = block_idx * words;
    
    // Read current states
    uint live[32];
    uint moved[32];
    uint init[32];
    
    uint max_words = min(words, 32u);
    for (uint w = 0; w < max_words; w++) {
        live[w] = live_states[state_start + w];
        moved[w] = moved_states[state_start + w];
        init[w] = init_states[state_start + w];
    }
    
    // Apply effects
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint effect = effects[block_idx * pc.effects_stride + stmt_idx];
        uint kind = effect >> 24;
        uint local = effect & 0xFFFFFF;
        
        if (local >= pc.num_locals) continue;
        
        uint word = local / 32;
        uint bit = local % 32;
        uint mask = 1u << bit;
        
        switch (kind) {
            case EFFECT_INIT:
                init[word] |= mask;
                live[word] |= mask;
                moved[word] &= ~mask;
                break;
            case EFFECT_MOVE:
                moved[word] |= mask;
                break;
            case EFFECT_BORROW_MUT:
                if ((live[word] & mask) != 0 && (moved[word] & mask) == 0) {
                    live[word] |= mask;
                }
                break;
            case EFFECT_BORROW_SHARED:
                if ((live[word] & mask) != 0) {
                    live[word] |= mask;
                }
                break;
            case EFFECT_DROP:
                live[word] &= ~mask;
                init[word] &= ~mask;
                break;
        }
    }
    
    // Write back
    bool changed = false;
    for (uint w = 0; w < max_words; w++) {
        if (live_states[state_start + w] != live[w]) {
            live_states[state_start + w] = live[w];
            changed = true;
        }
        if (moved_states[state_start + w] != moved[w]) {
            moved_states[state_start + w] = moved[w];
            changed = true;
        }
        if (init_states[state_start + w] != init[w]) {
            init_states[state_start + w] = init[w];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
