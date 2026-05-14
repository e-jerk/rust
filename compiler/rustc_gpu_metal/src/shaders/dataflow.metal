#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint bitset_words;
    uint effects_stride;
};

kernel void dataflow(
    const device uint* blocks [[buffer(0)]],
    const device uint* effects [[buffer(1)]],
    device uint* entry_states [[buffer(2)]],
    device uint* exit_states [[buffer(3)]],
    device atomic_uint* convergence [[buffer(4)]],
    constant PushConstants& pc [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint block_idx = thread_position_in_grid.x;
    if (block_idx >= pc.num_blocks) return;

    uint config_base = block_idx * 5;
    uint statement_count = blocks[config_base];
    // uint terminator_kind = blocks[config_base + 1]; // unused
    // uint successor_count = blocks[config_base + 2]; // unused
    // uint successor_0 = blocks[config_base + 3]; // unused
    // uint successor_1 = blocks[config_base + 4]; // unused

    uint entry_offset = block_idx * pc.bitset_words;

    uint state[64]; // max 64 words = 2048 bits
    for (uint i = 0; i < pc.bitset_words; i++) {
        state[i] = entry_states[entry_offset + i];
    }

    // Apply statement effects
    uint effect_base = block_idx * pc.effects_stride;
    for (uint stmt = 0; stmt < statement_count; stmt++) {
        uint base = effect_base + stmt * 3u;
        uint word_idx = effects[base];
        uint gen_mask = effects[base + 1u];
        uint kill_mask = effects[base + 2u];
        state[word_idx] |= gen_mask;
        state[word_idx] &= ~kill_mask;
    }

    // Write exit state and check convergence
    uint exit_offset = block_idx * pc.bitset_words;
    bool changed = false;
    for (uint i = 0; i < pc.bitset_words; i++) {
        uint old = exit_states[exit_offset + i];
        if (old != state[i]) {
            changed = true;
        }
        exit_states[exit_offset + i] = state[i];
    }

    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
