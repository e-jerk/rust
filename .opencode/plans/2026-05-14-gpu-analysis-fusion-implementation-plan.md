# GPU Analysis Fusion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fuse 4 MIR optimization analyses (dead store elimination, copy propagation, constant propagation, reaching definitions) into a single GPU compute dispatch, reducing overhead from 4×429μs to 1×429μs.

**Architecture:** Single compute shader where each thread processes one basic block across all 4 analyses simultaneously. Effects are packed into one uint32 per statement. States are interleaved per block.

**Tech Stack:** GLSL compute shaders, SPIR-V via glslangValidator, ash (Vulkan bindings), Rust

---

### Task 1: Write the Fused Compute Shader

**Files:**
- Create: `compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.comp`

**Step 1: Write shader header and bindings**

```glsl
#version 450

layout(local_size_x = 256) in;

// Shared block configuration
layout(set = 0, binding = 0) readonly buffer BlockConfigs {
    uint data[];
} configs;

// Packed effects: all 4 analyses in one uint32 per statement
layout(set = 0, binding = 1) readonly buffer Effects {
    uint data[];
} effects;

// Interleaved entry states: [dse(32) | copy(128) | const(128) | reach(32)] per block
layout(set = 0, binding = 2) buffer EntryStates {
    uint data[];
} entry_states;

// Interleaved exit states (same layout)
layout(set = 0, binding = 3) buffer ExitStates {
    uint data[];
} exit_states;

// Convergence flags: one per analysis
layout(set = 0, binding = 4) buffer Convergence {
    uint data[4];
} convergence;

layout(push_constant) uniform PushConstants {
    uint num_blocks;
    uint num_locals;       // For copy_prop / const_prop
    uint bitset_words;      // For DSE / reaching_defs
    uint effects_stride;
} pc;

// Effect kind constants
#define DSE_NOP 0
#define DSE_GEN 1
#define DSE_KILL 2
```

**Step 2: Write transfer functions**

Add after the bindings:

```glsl
// Analysis 0: Dead Store Elimination (backward, bitset)
void transfer_dse(uint block_idx, uint stmt_count, inout uint state[32]) {
    for (int stmt = int(stmt_count) - 1; stmt >= 0; stmt--) {
        uint effect = effects.data[block_idx * pc.effects_stride + uint(stmt)];
        uint kind = effect & 0xFF;
        uint local = (effect >> 8) & 0xFFFF;
        
        if (local >= pc.bitset_words * 32) continue;
        
        uint word = local / 32;
        uint bit = local % 32;
        uint mask = 1u << bit;
        
        if (kind == DSE_GEN) {
            state[word] |= mask;
        } else if (kind == DSE_KILL) {
            state[word] &= ~mask;
        }
    }
}

// Analysis 1: Copy Propagation (forward, per-local)
void transfer_copy_prop(uint block_idx, uint stmt_count, inout uint state[128]) {
    uint max_locals = min(pc.num_locals, 128);
    
    for (uint stmt = 0; stmt < stmt_count; stmt++) {
        uint effect = effects.data[block_idx * pc.effects_stride + stmt];
        uint copy_info = (effect >> 8) & 0xFFFF;
        
        if (copy_info != 0xFFFF) {
            uint dst = (copy_info >> 8) & 0xFF;
            uint src = copy_info & 0xFF;
            if (dst < max_locals) {
                state[dst] = src + 1;  // +1 to distinguish from "none"
            }
        } else {
            // Non-copy assignment: kill
            uint local = (effect >> 8) & 0xFFFF;
            if (local < max_locals) {
                state[local] = 0;
            }
        }
    }
}

// Analysis 2: Constant Propagation (forward, per-local)
void transfer_const_prop(uint block_idx, uint stmt_count, inout uint state[128]) {
    uint max_locals = min(pc.num_locals, 128);
    
    for (uint stmt = 0; stmt < stmt_count; stmt++) {
        uint effect = effects.data[block_idx * pc.effects_stride + stmt];
        uint const_info = (effect >> 16) & 0xFFFF;
        
        if (const_info != 0xFFFF) {
            uint local = (const_info >> 8) & 0xFF;
            uint value = const_info & 0xFF;
            if (local < max_locals) {
                state[local] = value + 1;
            }
        } else {
            // Non-constant assignment: kill
            uint local = (effect >> 8) & 0xFFFF;
            if (local < max_locals) {
                state[local] = 0;
            }
        }
    }
}

// Analysis 3: Reaching Definitions (forward, bitset)
void transfer_reaching_defs(uint block_idx, uint stmt_count, inout uint state[32]) {
    for (uint stmt = 0; stmt < stmt_count; stmt++) {
        uint effect = effects.data[block_idx * pc.effects_stride + stmt];
        uint def_id = (effect >> 24) & 0xFF;
        
        if (def_id != 0xFF) {
            uint word = def_id / 32;
            uint bit = def_id % 32;
            state[word] |= (1u << bit);
        }
    }
}
```

**Step 3: Write main function**

Add after transfer functions:

```glsl
void main() {
    uint block_idx = gl_GlobalInvocationID.x;
    if (block_idx >= pc.num_blocks) return;
    
    uint config_base = block_idx * 4;
    uint stmt_count = configs.data[config_base];
    
    // State arrays for all 4 analyses
    uint dse_state[32];
    uint copy_state[128];
    uint const_state[128];
    uint reach_state[32];
    
    uint max_locals = min(pc.num_locals, 128);
    uint state_base = block_idx * (32 + 128 + 128 + 32);
    
    // Load all 4 entry states
    for (uint i = 0; i < pc.bitset_words; i++) {
        dse_state[i] = entry_states.data[state_base + i];
        reach_state[i] = entry_states.data[state_base + 32 + 128 + 128 + i];
    }
    for (uint i = 0; i < max_locals; i++) {
        copy_state[i] = entry_states.data[state_base + 32 + i];
        const_state[i] = entry_states.data[state_base + 32 + 128 + i];
    }
    
    // Apply all 4 transfer functions
    transfer_dse(block_idx, stmt_count, dse_state);
    transfer_copy_prop(block_idx, stmt_count, copy_state);
    transfer_const_prop(block_idx, stmt_count, const_state);
    transfer_reaching_defs(block_idx, stmt_count, reach_state);
    
    // Write all 4 exit states and check convergence
    bool any_changed = false;
    
    // DSE exit state (offset: 32 + 128 + 128 = 288)
    for (uint i = 0; i < pc.bitset_words; i++) {
        uint idx = state_base + 288 + i;
        if (exit_states.data[idx] != dse_state[i]) {
            exit_states.data[idx] = dse_state[i];
            any_changed = true;
        }
    }
    
    // Copy propagation exit state (offset: 32)
    for (uint i = 0; i < max_locals; i++) {
        uint idx = state_base + 32 + i;
        if (exit_states.data[idx] != copy_state[i]) {
            exit_states.data[idx] = copy_state[i];
            any_changed = true;
        }
    }
    
    // Constant propagation exit state (offset: 32 + 128 = 160)
    for (uint i = 0; i < max_locals; i++) {
        uint idx = state_base + 160 + i;
        if (exit_states.data[idx] != const_state[i]) {
            exit_states.data[idx] = const_state[i];
            any_changed = true;
        }
    }
    
    // Reaching definitions exit state (offset: 32 + 128 + 128 = 288)
    for (uint i = 0; i < pc.bitset_words; i++) {
        uint idx = state_base + 288 + i;
        if (exit_states.data[idx] != reach_state[i]) {
            exit_states.data[idx] = reach_state[i];
            any_changed = true;
        }
    }
    
    if (any_changed) {
        atomicOr(convergence.data[0], 1);
        atomicOr(convergence.data[1], 1);
        atomicOr(convergence.data[2], 1);
        atomicOr(convergence.data[3], 1);
    }
}
```

**Step 4: Commit**

```bash
git add compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.comp
git commit -m "feat: add fused MIR optimization compute shader (4 analyses in 1 dispatch)"
```

---

### Task 2: Compile Shader to SPIR-V

**Files:**
- Modify: `compiler/rustc_gpu_vulkan/build.rs` (to compile new shader)
- Create: `compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.spv`

**Step 1: Check build.rs for shader compilation**

Read `compiler/rustc_gpu_vulkan/build.rs` to understand how other shaders are compiled.

**Step 2: Add compilation for fused shader**

If build.rs uses a loop or explicit compilation, add:

```rust
// In build.rs, add to shader list
compile_shader("fused_mir_opt");
```

Or if it's manual:

```rust
Command::new("glslangValidator")
    .args(&["-V", "-o", "src/shaders/fused_mir_opt.spv", "src/shaders/fused_mir_opt.comp"])
    .status()?;
```

**Step 3: Verify compilation**

```bash
glslangValidator -V -o compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.spv compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.comp
```

Expected output: `Linked...` with no errors.

**Step 4: Commit**

```bash
git add compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.spv compiler/rustc_gpu_vulkan/build.rs
git commit -m "feat: compile fused MIR optimization shader to SPIR-V"
```

---

### Task 3: Add Shader Loading Function

**Files:**
- Modify: `compiler/rustc_gpu_vulkan/src/lib.rs`

**Step 1: Add load function**

Find where other `load_*_shader()` functions are defined (likely around line 80). Add:

```rust
/// Load the fused MIR optimization shader (4 analyses in 1 dispatch).
pub fn load_fused_mir_opt_shader() -> Option<Vec<u8>> {
    Some(include_bytes!("shaders/fused_mir_opt.spv").to_vec())
}
```

**Step 2: Verify it compiles**

```bash
./x.py check --stage 1 compiler/rustc_gpu_vulkan
```

Expected: Build successful.

**Step 3: Commit**

```bash
git add compiler/rustc_gpu_vulkan/src/lib.rs
git commit -m "feat: add fused MIR optimization shader loader"
```

---

### Task 4: Add Rust Dispatch Method

**Files:**
- Modify: `compiler/rustc_gpu_vulkan/src/dataflow.rs`

**Step 1: Add dispatch method to GpuDataflowEngine**

Find where `dispatch_ssa_round` ends (around line 1000). Add after it:

```rust
    /// Dispatch fused MIR optimization round (4 analyses in 1 kernel).
    ///
    /// Fuses dead store elimination, copy propagation, constant propagation,
    /// and reaching definitions into a single compute dispatch.
    ///
    /// Buffer layout:
    /// - Binding 0: block_configs (shared CFG)
    /// - Binding 1: effects_fused (packed effects, 1 uint32 per stmt)
    /// - Binding 2: entry_states_fused (interleaved: dse|copy|const|reach)
    /// - Binding 3: exit_states_fused (same interleaving)
    /// - Binding 4: convergence (4 flags)
    pub fn dispatch_fused_mir_opt_round(
        &self,
        config_buf: &GpuBuffer,
        effects_buf: &GpuBuffer,
        entry_buf: &GpuBuffer,
        exit_buf: &GpuBuffer,
        convergence_buf: &GpuBuffer,
        num_blocks: u32,
        num_locals: u32,
        bitset_words: u32,
        effects_stride: u32,
    ) -> Result<(), vk::Result> {
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(config_buf.buffer)
                .range(config_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(effects_buf.buffer)
                .range(effects_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(entry_buf.buffer)
                .range(entry_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(exit_buf.buffer)
                .range(exit_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(convergence_buf.buffer)
                .range(convergence_buf.size),
        ];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2])),
            vk::WriteDescriptorSet::default()
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[3])),
            vk::WriteDescriptorSet::default()
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[4])),
        ];

        self.dispatch_persistent(&writes, |cmd_buf, device| {
            let push_constants = [num_blocks, num_locals, bitset_words, effects_stride];
            let push_bytes = unsafe {
                std::slice::from_raw_parts(
                    push_constants.as_ptr() as *const u8,
                    push_constants.len() * std::mem::size_of::<u32>(),
                )
            };
            unsafe {
                device.cmd_push_constants(
                    cmd_buf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_bytes,
                );
            }

            let workgroup_count = (num_blocks + 255) / 256;
            unsafe { device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1); }

            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .buffer(exit_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .buffer(convergence_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
            ];
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd_buf,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &[],
                    &barriers,
                    &[],
                );
            }

            Ok(())
        })
    }
```

**Step 2: Verify compilation**

```bash
./x.py check --stage 1 compiler/rustc_gpu_vulkan
```

Expected: Build successful.

**Step 3: Commit**

```bash
git add compiler/rustc_gpu_vulkan/src/dataflow.rs
git commit -m "feat: add fused MIR optimization dispatch method"
```

---

### Task 5: Update Validation Example

**Files:**
- Modify: `compiler/rustc_gpu_vulkan/examples/validate_m1.rs`

**Step 1: Add fused shader to validation test**

Find where shaders are listed (around line 25). Add:

```rust
("fused_mir_opt", rustc_gpu_vulkan::load_fused_mir_opt_shader()),
```

**Step 2: Add fused dispatch benchmark**

After the existing Test 5 (around line 162), add:

```rust
    // Test 6: Fused Dispatch Benchmark
    println!("\nTest 6: Fused GPU Dispatch (4 analyses in 1)");
    if let Some(spv) = rustc_gpu_vulkan::load_fused_mir_opt_shader() {
        if let Ok(pipeline) = rustc_gpu_vulkan::shader::ComputePipeline::from_spirv(
            &backend.context.device,
            &spv,
        ) {
            let num_blocks = 100u32;
            let num_locals = 10u32;
            let bitset_words = ((num_locals + 31) / 32) as u32;
            let effects_stride = 20u32;
            
            // Allocate buffers for fused layout
            let config_size = (num_blocks as usize * 4 * std::mem::size_of::<u32>()) as u64;
            let effects_size = (num_blocks as usize * effects_stride as usize * std::mem::size_of::<u32>()) as u64;
            let state_size = (num_blocks as usize * (32 + 128 + 128 + 32) * std::mem::size_of::<u32>()) as u64;
            
            let config_buf = backend.create_buffer(config_size);
            let effects_buf = backend.create_buffer(effects_size);
            let entry_buf = backend.create_buffer(state_size);
            let exit_buf = backend.create_buffer(state_size);
            let convergence_buf = backend.create_buffer(4 * std::mem::size_of::<u32>() as u64);
            
            if let (Some(cb), Some(eb), Some(enb), Some(exb), Some(conb)) = 
                (config_buf, effects_buf, entry_buf, exit_buf, convergence_buf) {
                
                // Initialize with synthetic data
                let configs: Vec<u32> = (0..num_blocks).flat_map(|i| {
                    vec![5u32, i, u32::MAX, 0]
                }).collect();
                cb.write(&configs);
                
                let effects: Vec<u32> = vec![0; num_blocks as usize * effects_stride as usize];
                eb.write(&effects);
                
                let states: Vec<u32> = vec![0; num_blocks as usize * (32 + 128 + 128 + 32)];
                enb.write(&states);
                exb.write(&states);
                conb.write(&[0u32, 0, 0, 0]);
                
                // Create dataflow engine with fused shader
                let mut gpu = rustc_gpu_vulkan::dataflow::GpuDataflowEngine::new(
                    &backend.context,
                    &spv,
                ).ok();
                
                if let Some(ref mut gpu_engine) = gpu {
                    let num_iterations = 100;
                    let start = std::time::Instant::now();
                    
                    for _ in 0..num_iterations {
                        let _ = gpu_engine.dispatch_fused_mir_opt_round(
                            &cb, &eb, &enb, &exb, &conb,
                            num_blocks, num_locals, bitset_words, effects_stride,
                        );
                    }
                    
                    let total_elapsed = start.elapsed();
                    let per_dispatch = total_elapsed / num_iterations;
                    
                    println!("  ✅ {} fused dispatches completed", num_iterations);
                    println!("  Total time: {:?}", total_elapsed);
                    println!("  Per dispatch: {:?}", per_dispatch);
                    println!("  Effective per-analysis overhead: ~{}μs", 
                        per_dispatch.as_micros() / 4);
                }
            }
        }
    }
```

**Step 3: Test compilation**

```bash
cargo check --manifest-path compiler/rustc_gpu_vulkan/Cargo.toml --example validate_m1
```

Expected: Build successful.

**Step 4: Commit**

```bash
git add compiler/rustc_gpu_vulkan/examples/validate_m1.rs
git commit -m "feat: add fused dispatch benchmark to validation test"
```

---

### Task 6: Update Benchmark Script

**Files:**
- Modify: `benchmark_gpu_speedups.py`

**Step 1: Add fused analysis to GPU_PHASES**

Find where `mir_optimizations` is defined. Update or add:

```python
    "fused_mir_optimizations": {
        "cpu_fraction": 0.12,
        "gpu_speedup": 3.0,  # Same compute, but 4x less overhead
        "kernel_launch_us": 429,  # One dispatch instead of 4
        "batch_size": 65536,
        "amortization_threshold": 2000,  # Lower threshold due to fused overhead
    },
```

**Step 2: Update speedup calculation**

In `calculate_total_speedup()`, replace the 4 separate MIR optimization phases with the fused one, or add it as an alternative path.

**Step 3: Run updated benchmark**

```bash
python3 benchmark_gpu_speedups.py
```

Expected: Updated speedup table shows improved numbers for medium crates.

**Step 4: Commit**

```bash
git add benchmark_gpu_speedups.py
git commit -m "feat: update benchmark script with fused analysis speedups"
```

---

### Task 7: Integration Test - Compile All Crates

**Files:**
- All modified files

**Step 1: Compile all GPU-related crates**

```bash
./x.py check --stage 1 compiler/rustc_gpu_vulkan compiler/rustc_mir_dataflow compiler/rustc_monomorphize
```

Expected: All 3 crates compile successfully.

**Step 2: Run validation test**

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH \
  cargo run --manifest-path compiler/rustc_gpu_vulkan/Cargo.toml --example validate_m1 --release
```

Expected:
- All 17 shaders load (16 original + 1 fused)
- Fused dispatch benchmark shows ~429μs per dispatch
- Effective per-analysis overhead shows ~107μs

**Step 3: Commit final changes**

```bash
git add -A
git commit -m "feat: GPU analysis fusion - 4 MIR optimization analyses in 1 dispatch"
```

---

### Task 8: Documentation Update

**Files:**
- Modify: `GPU_IMPLEMENTATION_SUMMARY.md` (if exists)
- Modify: `AGENTS.md`

**Step 1: Update implementation summary**

Add fused analysis to the list of GPU-accelerated phases.

**Step 2: Update AGENTS.md progress section**

Update the "In Progress" and "Done" sections to reflect the fused analysis work.

**Step 3: Commit**

```bash
git add GPU_IMPLEMENTATION_SUMMARY.md AGENTS.md
git commit -m "docs: update GPU implementation summary with fused analysis"
```

---

## Summary

This plan creates a fused compute shader that runs 4 MIR optimization analyses in a single GPU dispatch, reducing overhead from 1,716μs to 429μs for the MIR optimization phase.

**Key files created:**
- `compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.comp`
- `compiler/rustc_gpu_vulkan/src/shaders/fused_mir_opt.spv`

**Key files modified:**
- `compiler/rustc_gpu_vulkan/src/lib.rs`
- `compiler/rustc_gpu_vulkan/src/dataflow.rs`
- `compiler/rustc_gpu_vulkan/build.rs`
- `compiler/rustc_gpu_vulkan/examples/validate_m1.rs`
- `benchmark_gpu_speedups.py`

**Expected outcome:**
- 4× overhead reduction for MIR optimization dispatches
- Improved speedups for medium-sized crates (5K-20K items)
- All crates compile successfully
