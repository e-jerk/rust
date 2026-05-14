#include <metal_stdlib>
using namespace metal;

struct PushConstants {
    uint num_invocations;
};

#define TOK_IDENT 1
#define TOK_LIT 2
#define TOK_PUNCT 3
#define TOK_EOF 0xFFFFFFFF

kernel void macro_expand(
    device uint* invocations [[buffer(0)]],
    const device uint* input_tokens [[buffer(1)]],
    device uint* output_tokens [[buffer(2)]],
    const device uint* macro_rules [[buffer(3)]],
    constant PushConstants& pc [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_invocations) return;
    
    // Read invocation descriptor
    uint inv_offset = idx * 5;
    uint macro_id = invocations[inv_offset];
    uint num_input = invocations[inv_offset + 1];
    uint input_off = invocations[inv_offset + 2];
    uint max_output = invocations[inv_offset + 3];
    uint output_off = invocations[inv_offset + 4];
    
    // For each input token, try to expand it
    uint output_count = 0;
    for (uint i = 0; i < num_input && output_count < max_output; i++) {
        uint token = input_tokens[input_off + i];
        
        // Try macro rules (simplified: direct substitution)
        bool expanded = false;
        uint rule_base = macro_id * 1024; // Max 1024 rules per macro
        
        for (uint r = 0; r < 1024 && !expanded; r++) {
            uint pattern = macro_rules[rule_base + r * 2];
            if (pattern == TOK_EOF) break;
            
            if (pattern == token) {
                // Match! Emit template tokens
                uint template_token = macro_rules[rule_base + r * 2 + 1];
                if (template_token != TOK_EOF && output_count < max_output) {
                    output_tokens[output_off + output_count] = template_token;
                    output_count++;
                }
                expanded = true;
            }
        }
        
        if (!expanded && output_count < max_output) {
            // Pass through unchanged
            output_tokens[output_off + output_count] = token;
            output_count++;
        }
    }
    
    // Write output count back to invocation descriptor
    invocations[inv_offset + 3] = output_count;
}
