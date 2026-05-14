#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_blocks;
    uint num_locals;
    uint facts_stride;
};

kernel void copy_prop(
    const device uint* configs [[buffer(0)]],
    const device uint* copy_facts [[buffer(1)]],
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
    uint state_start = block_idx * pc.num_locals;
    uint state[128]; // Max 128 locals per function for copy propagation
    
    uint max_locals = min(pc.num_locals, 128u);
    for (uint i = 0; i < max_locals; i++) {
        state[i] = entry_states[state_start + i];
    }
    
    // Apply copy facts forward through statements
    for (uint stmt_idx = 0; stmt_idx < stmt_count; stmt_idx++) {
        uint fact = copy_facts[block_idx * pc.facts_stride + stmt_idx];
        if (fact != 0xFFFFFFFF) {
            uint dst = (fact >> 16) & 0xFFFF;
            uint src = fact & 0xFFFF;
            if (dst < max_locals) {
                state[dst] = src + 1; // +1 to distinguish from "no copy" (0)
            }
        }
    }
    
    // Write exit state
    bool changed = false;
    for (uint i = 0; i < max_locals; i++) {
        uint old = exit_states[state_start + i];
        if (old != state[i]) {
            exit_states[state_start + i] = state[i];
            changed = true;
        }
    }
    
    if (changed) {
        atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
    }
}
