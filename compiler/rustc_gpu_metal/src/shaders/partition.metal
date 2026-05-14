#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_nodes;
    uint max_label;
    uint max_size;
};

kernel void partition(
    const device uint* edge_list [[buffer(0)]],
    const device uint* edge_offsets [[buffer(1)]],
    device uint* labels [[buffer(2)]],
    device uint* label_counts [[buffer(3)]],
    device atomic_uint* convergence [[buffer(4)]],
    constant PushConstants& pc [[buffer(5)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint node_idx = thread_position_in_grid.x;
    if (node_idx >= pc.num_nodes) return;
    
    uint current_label = labels[node_idx];
    
    // Get edge range for this node
    uint start = edge_offsets[node_idx];
    uint end = (node_idx + 1 < pc.num_nodes) ? edge_offsets[node_idx + 1] : pc.num_nodes;
    
    // Count neighbor labels using a small local histogram
    uint neighbor_labels[8];
    uint neighbor_counts[8];
    uint distinct = 0;
    
    for (uint e = start; e < end; e++) {
        uint neighbor = edge_list[e];
        if (neighbor >= pc.num_nodes) continue;
        
        uint neighbor_label = labels[neighbor];
        
        // Find or insert in histogram
        bool found = false;
        for (uint i = 0; i < distinct && i < 8; i++) {
            if (neighbor_labels[i] == neighbor_label) {
                neighbor_counts[i]++;
                found = true;
                break;
            }
        }
        if (!found && distinct < 8) {
            neighbor_labels[distinct] = neighbor_label;
            neighbor_counts[distinct] = 1;
            distinct++;
        }
    }
    
    // Find best label (most common among neighbors)
    uint best_label = current_label;
    uint best_count = 0;
    
    for (uint i = 0; i < distinct; i++) {
        if (neighbor_counts[i] > best_count) {
            best_count = neighbor_counts[i];
            best_label = neighbor_labels[i];
        }
    }
    
    // Only change if best label is different, not full, and has at least 2 neighbors
    if (best_label != current_label && best_count >= 2) {
        uint new_size = label_counts[best_label];
        if (new_size < pc.max_size) {
            labels[node_idx] = best_label;
            atomic_fetch_add_explicit(&label_counts[best_label], 1u, memory_order_relaxed);
            atomic_fetch_or_explicit(&convergence[0], 1u, memory_order_relaxed);
        }
    }
}
