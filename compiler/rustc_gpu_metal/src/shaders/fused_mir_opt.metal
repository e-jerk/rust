#include <metal_stdlib>
using namespace metal;

// Maximum sizes for per-thread state arrays
#define MAX_BITSET_WORDS 32
#define MAX_LOCALS 128
#define BLOCK_STATE_STRIDE (MAX_BITSET_WORDS + MAX_LOCALS + MAX_LOCALS + MAX_BITSET_WORDS)

#define EFFECT_DSE_NOP  0
#define EFFECT_DSE_GEN  1
#define EFFECT_DSE_KILL 2

inline uint dseEffect(uint effect) { return effect & 0xFF; }
inline uint dseLocal(uint effect) { return (effect >> 8) & 0xFF; }
inline uint copyFact(uint effect) { return (effect >> 8) & 0xFFFF; }
inline uint constFact(uint effect) { return (effect >> 16) & 0xFFFF; }
inline uint reachDef(uint effect) { return (effect >> 24) & 0xFF; }

#define COPY_NOOP  0xFFFF
#define CONST_NOOP 0xFFFF
#define REACH_NOOP 0xFF

struct PushConstants {
    uint num_blocks;
    uint num_locals;
    uint bitset_words;
    uint effects_stride;
};

// Analysis 0: Dead Store Elimination
void transfer_dse(
    uint block_idx,
    uint stmt_count,
    uint max_bitset_words,
    uint effects_stride,
    const device uint* effects_fused,
    thread uint* dse_state
) {
    for (uint stmt_idx = stmt_count; stmt_idx-- > 0; ) {
        uint effect = effects_fused[block_idx * effects_stride + stmt_idx];
        uint kind = dseEffect(effect);
        
        if (kind == EFFECT_DSE_NOP) continue;
        
        uint local = dseLocal(effect);
        if (local >= max_bitset_words * 32) continue;
        
        uint word = local / 32;
        uint bit = local % 32;
        uint mask = 1u << bit;
        
        if (kind == EFFECT_DSE_GEN) {
            dse_state[word] |= mask;
        } else if (kind == EFFECT_DSE_KILL) {
            dse_state[word] &= ~mask;
        }
    }
}

// Analysis 1: Copy Propagation
void transfer_copy_prop(
    uint block_idx,
    uint stmt_count,
    uint max_locals,
    uint effects_stride,
    const device uint* effects_fused,
    thread uint* copy_state
) {
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint effect = effects_fused[block_idx * effects_stride + stmt_idx];
        uint fact = copyFact(effect);
        
        if (fact == COPY_NOOP) continue;
        
        uint dst = (fact >> 8) & 0xFF;
        uint src = fact & 0xFF;
        
        if (dst < max_locals) {
            copy_state[dst] = src + 1;
        }
    }
}

// Analysis 2: Constant Propagation
void transfer_const_prop(
    uint block_idx,
    uint stmt_count,
    uint max_locals,
    uint effects_stride,
    const device uint* effects_fused,
    thread uint* const_state
) {
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint effect = effects_fused[block_idx * effects_stride + stmt_idx];
        uint fact = constFact(effect);
        
        if (fact == CONST_NOOP) continue;
        
        uint local = (fact >> 8) & 0xFF;
        uint value = fact & 0xFF;
        
        if (local < max_locals) {
            const_state[local] = value + 1;
        }
    }
}

// Analysis 3: Reaching Definitions
void transfer_reaching_defs(
    uint block_idx,
    uint stmt_count,
    uint max_bitset_words,
    uint effects_stride,
    const device uint* effects_fused,
    thread uint* reach_state
) {
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint effect = effects_fused[block_idx * effects_stride + stmt_idx];
        uint def_id = reachDef(effect);
        
        if (def_id == REACH_NOOP) continue;
        
        uint word = def_id / 32;
        uint bit = def_id % 32;
        
        if (word < max_bitset_words) {
            reach_state[word] |= (1u << bit);
        }
    }
}

// Main compute kernel
kernel void fused_mir_opt(
    const device uint* configs [[buffer(0)]],
    const device uint* effects_fused [[buffer(1)]],
    device uint* entry_states_fused [[buffer(2)]],
    device uint* exit_states_fused [[buffer(3)]],
    device atomic_uint* convergence [[buffer(4)]],
    constant PushConstants& pc [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint stmt_count = configs[block_idx * 4];
    
    uint block_base = block_idx * BLOCK_STATE_STRIDE;
    uint dse_start = block_base;
    uint copy_start = block_base + MAX_BITSET_WORDS;
    uint const_start = copy_start + MAX_LOCALS;
    uint reach_start = const_start + MAX_LOCALS;
    
    uint max_bitset_words = min(pc.bitset_words, (uint)MAX_BITSET_WORDS);
    uint max_locals = min(pc.num_locals, (uint)MAX_LOCALS);
    uint effects_stride = pc.effects_stride;
    
    // Load entry states
    uint dse_state[MAX_BITSET_WORDS];
    for (uint w = 0; w < max_bitset_words; w++) {
        dse_state[w] = entry_states_fused[dse_start + w];
    }
    
    uint copy_state[MAX_LOCALS];
    for (uint i = 0; i < max_locals; i++) {
        copy_state[i] = entry_states_fused[copy_start + i];
    }
    
    uint const_state[MAX_LOCALS];
    for (uint i = 0; i < max_locals; i++) {
        const_state[i] = entry_states_fused[const_start + i];
    }
    
    uint reach_state[MAX_BITSET_WORDS];
    for (uint w = 0; w < max_bitset_words; w++) {
        reach_state[w] = entry_states_fused[reach_start + w];
    }
    
    // Apply transfer functions
    transfer_dse(block_idx, stmt_count, max_bitset_words, effects_stride, effects_fused, dse_state);
    transfer_copy_prop(block_idx, stmt_count, max_locals, effects_stride, effects_fused, copy_state);
    transfer_const_prop(block_idx, stmt_count, max_locals, effects_stride, effects_fused, const_state);
    transfer_reaching_defs(block_idx, stmt_count, max_bitset_words, effects_stride, effects_fused, reach_state);
    
    // Write exit states and track changes
    bool changed_dse = false;
    for (uint w = 0; w < max_bitset_words; w++) {
        uint old = exit_states_fused[dse_start + w];
        if (old != dse_state[w]) {
            exit_states_fused[dse_start + w] = dse_state[w];
            changed_dse = true;
        }
    }
    
    bool changed_copy = false;
    for (uint i = 0; i < max_locals; i++) {
        uint old = exit_states_fused[copy_start + i];
        if (old != copy_state[i]) {
            exit_states_fused[copy_start + i] = copy_state[i];
            changed_copy = true;
        }
    }
    
    bool changed_const = false;
    for (uint i = 0; i < max_locals; i++) {
        uint old = exit_states_fused[const_start + i];
        if (old != const_state[i]) {
            exit_states_fused[const_start + i] = const_state[i];
            changed_const = true;
        }
    }
    
    bool changed_reach = false;
    for (uint w = 0; w < max_bitset_words; w++) {
        uint old = exit_states_fused[reach_start + w];
        if (old != reach_state[w]) {
            exit_states_fused[reach_start + w] = reach_state[w];
            changed_reach = true;
        }
    }
    
    // Set convergence flags individually
    if (changed_dse) atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    if (changed_copy) atomic_fetch_or_explicit(&convergence[1], 1u, memory_order_relaxed);
    if (changed_const) atomic_fetch_or_explicit(&convergence[2], 1u, memory_order_relaxed);
    if (changed_reach) atomic_fetch_or_explicit(&convergence[3], 1u, memory_order_relaxed);
}
