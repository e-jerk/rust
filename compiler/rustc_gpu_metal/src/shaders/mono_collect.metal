#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_bodies;
};

#define ACTION_CALL 1
#define ACTION_DROP 2
#define ACTION_CAST 3
#define ACTION_CONST 4

kernel void mono_collect(
    const device uint* actions [[buffer(0)]],
    const device uint* body_offsets [[buffer(1)]],
    device uint* edges [[buffer(2)]],
    device atomic_uint* edge_counter [[buffer(3)]],
    constant PushConstants& pc [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint body_idx = thread_position_in_grid.x;
    if (body_idx >= pc.num_bodies) return;
    
    uint offset = body_offsets[body_idx];
    uint end = body_offsets[body_idx + 1];
    
    for (uint i = offset; i < end; i += 4) {
        uint kind = actions[i];
        uint def_id = actions[i + 1];
        uint args_idx = actions[i + 2];
        uint def_id_krate = actions[i + 3];
        
        if (kind == ACTION_CALL || kind == ACTION_DROP || kind == ACTION_CAST) {
            uint edge_idx = atomic_fetch_add_explicit(&edge_counter[0], 1u, memory_order_relaxed) * 4;
            edges[edge_idx] = def_id;
            edges[edge_idx + 1] = args_idx;
            edges[edge_idx + 2] = body_idx;
            edges[edge_idx + 3] = def_id_krate;
        }
    }
}
