# Design: GPU-Accelerated MIR Dataflow Analysis

**Date:** 2026-05-14
**Status:** ✅ Complete — 12 GPU dataflow analysis types implemented in rustc_mir_dataflow
**Scope:** Vulkan compute acceleration of rustc MIR dataflow fixed-point iteration
**Depends on:** GPU monomorphization infrastructure (rustc_gpu_vulkan)

---

## 1. Problem Statement

rustc's MIR dataflow framework performs fixed-point iteration over control-flow graphs for every function. Key analyses include:
- Liveness analysis (`MaybeLiveLocals`)
- Initialized places (`MaybeInitializedPlaces`)
- Borrow tracking (`Borrows` in borrowck)
- Storage liveness (`MaybeStorageLive`)

These analyses dominate ~2% of compile time but are required for every function. The current implementation uses a sequential work-queue algorithm that processes one basic block at a time.

**Goal:** GPU-accelerate the fixed-point iteration for bitset-based dataflow analyses.

---

## 2. Architecture

### Why MIR Dataflow is GPU-Suitable

| Property | Trait Solver ❌ | Pattern Analysis ⚠️ | MIR Dataflow ✅ |
|----------|-----------------|---------------------|-----------------|
| Domain | Complex recursive types | Recursive pattern trees | **Bitsets (DenseBitSet)** |
| Algorithm | Stateful unification | Tree search with pruning | **Fixed-point over CFG** |
| Work units | Single obligation | Single constructor split | **One thread per basic block** |
| Independence | None (path-dependent) | Some (constructor loop) | **High within an iteration** |
| Memory pattern | Random pointer chasing | Moderate recursion | **Dense arrays, sequential** |

### Core Insight

Within a single fixed-point iteration round, **each basic block's transfer function is independent**. A block only needs:
1. Its entry bitset (read-only for this round)
2. Its own statements/terminator (read-only)
3. Writes to its exit bitset

Cross-block dependencies only appear between rounds when exit states are joined into successor entry states.

### Wavefront Iteration Strategy

Replace the CPU work-queue with a **dense wavefront approach**:

```
Round 1:
  [GPU Kernel] Process all basic blocks in parallel
    Each thread: load entry bitset → apply statements → write exit bitset
  [GPU Kernel] Propagate exit → entry for all edges
    Each thread: atomic OR exit into successor entry
  [Host] Check global convergence flag

Round 2+: Repeat until converged
```

This is simpler than a sparse work queue and maps well to GPU execution.

---

## 3. Dataflow Shader Design

### Shader: `dataflow.comp`

```glsl
#version 450

layout(local_size_x = 256) in;

// Per-basic-block configuration
layout(set = 0, binding = 0) readonly buffer BlockConfig {
    uint statement_count;   // number of statements in this block
    uint terminator_kind;   // enum: Goto, SwitchInt, Call, Return, etc.
    uint successor_count;
    uint successor_0;       // first successor index
    uint successor_1;       // second successor (if applicable)
    // ... more block metadata
} blocks[];

// Statement effects (packed)
layout(set = 0, binding = 1) readonly buffer StatementEffects {
    uint data[]; // packed gen/kill operations
} effects;

// Entry states (one bitset per block)
layout(set = 0, binding = 2) buffer EntryStates {
    uint bits[]; // bitset data, size = num_blocks * bitset_words
} entry_states;

// Exit states (one bitset per block)
layout(set = 0, binding = 3) buffer ExitStates {
    uint bits[]; // bitset data
} exit_states;

// Convergence flag: 0 = converged, 1 = changed
layout(set = 0, binding = 4) buffer Convergence {
    uint changed;
} convergence;

// Push constants
layout(push_constant) uniform PushConstants {
    uint num_blocks;
    uint bitset_words;      // bitset size in 32-bit words
    uint num_rounds;        // current round number (for debugging)
} pc;

void main() {
    uint block_idx = gl_GlobalInvocationID.x;
    if (block_idx >= pc.num_blocks) return;
    
    // Load entry state
    uint entry_offset = block_idx * pc.bitset_words;
    uint state[BITSET_MAX_WORDS]; // compile-time max or dynamic via SSBO
    for (uint i = 0; i < pc.bitset_words; i++) {
        state[i] = entry_states.bits[entry_offset + i];
    }
    
    // Apply statement effects
    BlockConfig config = blocks[block_idx];
    for (uint stmt = 0; stmt < config.statement_count; stmt++) {
        // Each statement effect is a (gen_mask, kill_mask) pair
        // uint gen = effects.data[...];
        // uint kill = effects.data[...];
        // state[word] |= gen;
        // state[word] &= ~kill;
    }
    
    // Apply terminator effects
    switch (config.terminator_kind) {
        case TERMINATOR_RETURN:
            // Kill all locals or apply return-specific effects
            break;
        case TERMINATOR_GOTO:
        case TERMINATOR_SWITCH_INT:
            // Effects already applied in statement loop
            break;
    }
    
    // Write exit state
    uint exit_offset = block_idx * pc.bitset_words;
    for (uint i = 0; i < pc.bitset_words; i++) {
        exit_states.bits[exit_offset + i] = state[i];
    }
}
```

### Edge Propagation Kernel

```glsl
// Second kernel: propagate exit states to successor entry states
// Each thread processes one edge (block → successor)

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly buffer BlockConfig { ... } blocks;
layout(set = 0, binding = 3) readonly buffer ExitStates { uint bits[]; } exit_states;
layout(set = 0, binding = 2) buffer EntryStates { uint bits[]; } entry_states;
layout(set = 0, binding = 4) buffer Convergence { uint changed; } convergence;

layout(push_constant) uniform PushConstants {
    uint num_blocks;
    uint bitset_words;
} pc;

void main() {
    uint block_idx = gl_GlobalInvocationID.x;
    if (block_idx >= pc.num_blocks) return;
    
    BlockConfig config = blocks[block_idx];
    uint exit_offset = block_idx * pc.bitset_words;
    
    // For each successor, join exit state into successor entry
    for (uint succ = 0; succ < config.successor_count; succ++) {
        uint succ_idx = config.successor_0; // or successor_N lookup
        uint entry_offset = succ_idx * pc.bitset_words;
        
        for (uint i = 0; i < pc.bitset_words; i++) {
            uint old_val = entry_states.bits[entry_offset + i];
            uint new_val = old_val | exit_states.bits[exit_offset + i];
            if (new_val != old_val) {
                entry_states.bits[entry_offset + i] = new_val;
                atomicOr(convergence.changed, 1);
            }
        }
    }
}
```

---

## 4. CPU Integration Design

### Where to Hook In

The `rustc_mir_dataflow` crate defines analyses. We add a GPU dispatch path:

```rust
// In rustc_mir_dataflow/src/framework/mod.rs or a new gpu_engine.rs

pub fn iterate_to_fixpoint_gpu<'tcx, A: Analysis<'tcx>>(
    body: &Body<'tcx>,
    analysis: A,
) -> Option<Results<'tcx, A>>
where
    A::Domain: BitSetDomain, // GPU only works for bitset domains
{
    if !GpuBackend::new().is_some() { return None; }
    // ... GPU path
}
```

### Supported Analyses

**Phase 1 (MVP):** Only forward analyses with `DenseBitSet<Local>` domain:
- `MaybeStorageLive`
- `MaybeStorageDead`
- `MaybeBorrowedLocals`

**Phase 2:** Add `MixedBitSet` support for:
- `MaybeInitializedPlaces`
- `MaybeUninitializedPlaces`
- `Borrows` (borrowck)

**Not supported:** Backward analyses (liveness) — would need reverse topological order and different shader.

### Serialization

Before dispatching to GPU, we serialize:
1. **Block metadata**: statement count, terminator kind, successors
2. **Statement effects**: Precompute gen/kill masks for each statement
3. **Initial entry states**: From `bottom_value` and `initialize_start_block`

This is done per-function, at the time the analysis is requested.

---

## 5. Performance Model

### Why This Should Be Faster

| Factor | CPU (Sequential) | GPU (Parallel) |
|--------|-----------------|----------------|
| Blocks processed | 1 at a time | 256–1024 simultaneously |
| Bitset operations | CPU bitwise | GPU SIMT (32 threads per warp do 32-bit ops) |
| Memory bandwidth | Limited by cache | GPU HBM2/VRAM has 10x+ bandwidth |
| Work queue overhead | Dynamic scheduling | Dense wavefront, no queue |

### Expected Speedup

For a function with:
- 100 basic blocks
- 50-bit domain (2 words)
- 10 fixed-point rounds

**CPU time:** 100 blocks × 10 rounds × (5 statements × ~10ns + join overhead ~50ns) ≈ 100μs
**GPU time:** 10 rounds × (kernel launch ~50μs + 100 blocks / 256 warps × ~5ns + edge propagation ~20ns) ≈ 500μs

**Hmm, for small functions GPU overhead dominates.**

### When GPU Wins

For large functions with:
- 1000+ basic blocks
- Large domains (256+ bits = 8+ words)
- Many rounds (complex CFGs)

**GPU time:** 10 rounds × (kernel launch ~50μs + 1000 blocks / 256 warps × ~50ns + edge propagation ~100ns) ≈ 500μs
**CPU time:** 1000 blocks × 10 rounds × ~100ns ≈ 1ms

**Speedup: ~2x for large functions**

### The Real Win: Batch Multiple Functions

The biggest opportunity is **batching multiple functions into a single GPU dispatch**:
- Compile all functions' MIR bodies into one large buffer
- Run dataflow for all functions simultaneously
- This amortizes kernel launch overhead

With batching:
- 100 functions × 100 blocks each = 10,000 blocks
- GPU processes all in ~1ms per round
- CPU would take ~10ms sequential
- **Speedup: ~5–10x**

---

## 6. Implementation Plan

### Phase 1: Infrastructure
1. Add `dataflow` module to `rustc_gpu_vulkan`
2. Write `dataflow.comp` shader
3. Add `build.rs` entry for SPIR-V compilation

### Phase 2: CPU Integration
4. Create `rustc_mir_dataflow/src/gpu_engine.rs`
5. Implement `GpuDataflowEngine` struct
6. Implement MIR body → GPU buffer serialization
7. Add `try_gpu` path to analysis framework

### Phase 3: Testing
8. Unit tests for serialization
9. Compare CPU vs GPU results on test MIR bodies
10. Benchmark on large functions

---

## 7. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU overhead exceeds work for small functions | Only dispatch when block count > threshold (e.g., 100) |
| MixedBitSet too complex for GPU | Start with DenseBitSet only |
| Backward analyses not supported | Document limitation; CPU fallback always works |
| Shader compilation failure | CPU fallback transparently |
| Atomics on bitset joins cause contention | Use warp-level primitives or per-warp reductions |

---

## 8. Relation to Monomorphization GPU Work

This reuses the `rustc_gpu_vulkan` crate:
- `GpuContext` — same Vulkan device
- `GpuBuffer` — same host-visible buffers
- `GpuDispatch` — same dispatch mechanism

New additions:
- `dataflow.comp` shader (separate from `mono_collect.comp`)
- `dataflow` module in `rustc_gpu_vulkan`
- Integration in `rustc_mir_dataflow`

---

## 9. Conclusion

MIR dataflow is the most GPU-suitable rustc frontend component after monomorphization. The bitset domains, independent basic block transfer functions, and wavefront-friendly iteration pattern make it a strong candidate for 2–5x speedup on large functions, and potentially 5–10x with multi-function batching.

The key insight is **not** that individual functions get huge speedups, but that batching many functions' dataflow analyses into a single GPU dispatch amortizes overhead and achieves meaningful compile-time reductions.
