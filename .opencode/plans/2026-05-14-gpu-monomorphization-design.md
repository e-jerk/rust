# Design: GPU-Accelerated Monomorphization Collection in rustc

**Date:** 2026-05-14  
**Status:** Approved  
**Scope:** Vulkan compute acceleration of rustc frontend monomorphization  
**Constraint:** Do not touch LLVM backend

---

## 1. Problem Statement

The Rust compiler spends a significant portion of compile time in monomorphization collection, especially for generic-heavy crates. The `collect_and_partition_mono_items` query recursively walks MIR bodies to discover all concrete instantiations of generic functions. This walk is currently sequential and CPU-bound.

For generic-heavy crates (e.g., those using `serde`, `tokio`, `futures`), monomorphization collection can consume 5–20% of total compile time, with known superlinear blowups (Issue #135477).

The goal is to offload the parallelizable portion of this work to a GPU using Vulkan compute shaders, reducing compile times for the most affected workloads.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    rustc_monomorphize                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ collector.rs │  │ gpu_collector│  │  rustc_gpu_vulkan   │  │
│  │  (existing)  │  │    (new)     │  │    (new crate)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│         │                 │                    │                │
│         └─────────────────┴────────────────────┘                │
│                           │                                     │
│                    ┌──────────────┐                             │
│                    │   Vulkan     │                             │
│                    │  GPU Device  │                             │
│                    └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. CPU discovers roots via HIR walk (unchanged)
2. CPU serializes each root's MIR into a flat GPU buffer
3. CPU dispatches a Vulkan compute shader that walks the MIR and emits discovered `Instance` edges
4. CPU reads back edge buffers, resolves `Instance` objects in `TyCtxt`, deduplicates via `visited` set
5. Newly discovered items are batched and sent back to the GPU
6. Repeat until fixed point

---

## 3. Crate Layout

We add **one new crate** inside `compiler/`:

```
compiler/rustc_gpu_vulkan/
├── Cargo.toml
├── src/
│   ├── lib.rs                 # Crate entry, feature gating
│   ├── context.rs             # Vulkan instance, device, queue, command pool
│   ├── buffer.rs              # Host-visible & device-local buffer management
│   ├── shader.rs              # SPIR-V shader loading & pipeline creation
│   ├── dispatch.rs            # Compute dispatch, synchronization
│   └── shaders/
│       └── mono_collect.comp  # GLSL compute shader for MIR edge discovery
```

**Modified crates:**
- `rustc_monomorphize/Cargo.toml` — add `rustc_gpu_vulkan` as optional dependency
- `rustc_monomorphize/src/collector.rs` — add GPU path in `collect_crate_mono_items`
- `rustc_monomorphize/src/gpu_collector.rs` — new module, orchestrates GPU batching

---

## 4. GPU Compute Infrastructure (Vulkan)

**Crate: `rustc_gpu_vulkan`**

| Component | Responsibility |
|---|---|
| `GpuContext` | Manages `VkInstance`, `VkPhysicalDevice`, `VkDevice`, `VkQueue`. Uses validation layers in debug builds. Falls back to CPU if no GPU available. |
| `GpuBuffer` | Wraps `VkBuffer` with host-visible memory for CPU↔GPU transfer. Uses `vkFlushMappedMemoryRanges` / `vkInvalidateMappedMemoryRanges`. |
| `GpuShader` | Loads precompiled SPIR-V from `&[u8]`. In development, we compile `.comp` → SPIR-V at build time via `glslangValidator` in `build.rs`. |
| `GpuDispatch` | Records command buffer, dispatches compute, inserts memory barriers, submits to queue. |

**Vulkan strategy:**
- Use **compute shaders only** (no graphics pipeline).
- One compute dispatch per batch of MIR bodies.
- `local_size_x = 64` — one workgroup processes 64 MIR bodies.
- Memory: host-visible COHERENT buffers for simplicity. We can optimize to device-local later.

**Why Vulkan:**
- Cross-platform (Linux, Windows, macOS via MoltenVK).
- No proprietary runtime dependency (unlike CUDA).
- Compute shaders are mature and well-supported.
- Rust ecosystem has `ash` (raw Vulkan bindings) and `gpu-alloc` (memory management).

**Dependency choice:**
- Use `ash` (raw Vulkan bindings) for minimal overhead. We don't need `vulkano`'s safety abstractions inside a compiler.
- Use `gpu-alloc` for simple memory allocation.

---

## 5. MIR-to-GPU Serialization

This is the hardest part. MIR is a complex nested Rust data structure. We need a flat, GPU-friendly representation.

**Approach: MIR Instruction Stream**

Instead of serializing the full MIR `Body`, we serialize a **stream of "mono actions"** — the minimal information needed to discover outgoing edges:

```rust
// CPU-side: packed into a GPU buffer
#[repr(C)]
struct MonoAction {
    kind: u32,           // enum: Call, Drop, Cast, Const, etc.
    def_id: u32,         // index into a def-id table
    generic_args_len: u32,
    generic_args_offset: u32, // offset into args buffer
}
```

**Serialization happens in `collector.rs`:**
- Before dispatching a batch to the GPU, the CPU calls `tcx.instance_mir(instance.def)` to get the MIR body.
- It runs a **lightweight CPU visitor** that flattens the body into a `Vec<MonoAction>` + auxiliary buffers (generic args, types as indices).
- This flattening is single-threaded per body, but we do it for N bodies in parallel on the CPU (using the existing `par_iter` infrastructure if available).

**Why this approach:**
- The GPU doesn't need to understand full MIR types, HIR, or the `TyCtxt` arena. It only needs to know: "this body references these `DefId`s with these generic args."
- The CPU is still responsible for resolving `DefId` → `Instance`, evaluating constants, and building the actual `MonoItem` graph.
- This keeps the GPU shader simple and maintainable.

**Generic args encoding:**
- Generic arguments (`GenericArgsRef<'tcx>`) are interned in `TyCtxt`. We can't pass pointers to the GPU.
- Instead, we assign each unique `GenericArgs` an index into a per-batch lookup table.
- The GPU shader only manipulates indices. The CPU resolves indices back to actual `GenericArgs` after readback.

---

## 6. GPU Monomorphization Collection Algorithm

**Compute shader (`mono_collect.comp`):**

```glsl
layout(local_size_x = 64) in;

// Input: flattened MIR action streams
layout(set = 0, binding = 0) readonly buffer Actions { uint data[]; } actions;
layout(set = 0, binding = 1) readonly buffer Offsets { uint data[]; } body_offsets;

// Output: discovered edges
layout(set = 0, binding = 2) writeonly buffer Edges { 
    uint def_id; 
    uint args_idx; 
} edges[];

void main() {
    uint body_idx = gl_GlobalInvocationID.x;
    if (body_idx >= num_bodies) return;
    
    uint offset = body_offsets.data[body_idx];
    uint end = body_offsets.data[body_idx + 1];
    
    uint edge_write_idx = atomicAdd(edge_counter, 0); // reserve space
    
    for (uint i = offset; i < end; i++) {
        uint kind = actions.data[i].kind;
        // Handle different action kinds in parallel within the workgroup
        if (kind == ACTION_CALL || kind == ACTION_DROP || kind == ACTION_CAST) {
            edges[edge_write_idx].def_id = actions.data[i].def_id;
            edges[edge_write_idx].args_idx = actions.data[i].args_idx;
            edge_write_idx++;
        }
    }
}
```

**CPU orchestration (`gpu_collector.rs`):**

```rust
pub fn gpu_collect_mono_items(
    tcx: TyCtxt<'_>,
    roots: Vec<MonoItem<'_>>,
) -> (Vec<MonoItem<'_>>, UsageMap<'_>) {
    let gpu = GpuContext::new().ok()?; // fallback to CPU if GPU unavailable
    
    let mut visited = FxHashSet::default();
    let mut queue = VecDeque::from(roots);
    let mut usage_map = UsageMap::new();
    
    while !queue.is_empty() {
        // Batch up to GPU_BATCH_SIZE items
        let batch: Vec<_> = queue.drain(..GPU_BATCH_SIZE.min(queue.len())).collect();
        
        // Serialize MIR bodies to GPU buffers
        let (actions_buf, offsets_buf) = serialize_batch(tcx, &batch);
        
        // Dispatch compute shader
        let edges_buf = gpu.dispatch(actions_buf, offsets_buf);
        
        // Read back edges
        let edges = edges_buf.read::<GpuEdge>();
        
        // Resolve edges on CPU: def_id + args_idx → Instance → MonoItem
        for edge in edges {
            let instance = resolve_edge(tcx, edge);
            let mono_item = MonoItem::Fn(instance);
            
            usage_map.record_usage(batch[edge.source_idx], mono_item);
            
            if visited.insert(mono_item) {
                queue.push_back(mono_item);
            }
        }
    }
    
    (visited.into_iter().collect(), usage_map)
}
```

**Key design decisions:**
- **Batched BFS:** We don't do a pure GPU BFS because the graph is dynamic. Instead, we do batched rounds: CPU → GPU → CPU → GPU.
- **CPU resolution:** The GPU only emits raw `(def_id, args_idx)` edges. The CPU turns these into proper `Instance` objects using `TyCtxt`. This avoids putting the entire type system on the GPU.
- **Deduplication on CPU:** The `visited` set stays on the CPU because it requires `MonoItem` equality and hashing, which depends on `TyCtxt` interning.

---

## 7. Integration with rustc Query System

**Entry point:** `collect_crate_mono_items` in `collector.rs`.

**Integration strategy:**
- Add a compiler flag `-Z gpu-mono=on` (unstable, nightly only).
- In `collect_crate_mono_items`, if the flag is set and `rustc_gpu_vulkan` compiled successfully:
  - Run root collection on CPU (unchanged).
  - Call `gpu_collector::gpu_collect_mono_items(tcx, roots)` instead of the recursive `collect_items_rec`.
- If the GPU path fails (no Vulkan device, out of memory, shader error), **fall back transparently** to the existing CPU collector.

**Why this integration is clean:**
- `collect_and_partition_mono_items` is already the boundary between frontend and codegen. The GPU collector produces the exact same output (`Vec<MonoItem>`, `UsageMap`) as the CPU collector.
- No other queries need to change. The partitioning step (`partition.rs`) runs on the CPU using the GPU-collected items.

---

## 8. Fallback & Correctness Strategy

| Risk | Mitigation |
|---|---|
| GPU produces different edges than CPU | **Fuzzer:** Run both collectors on the same crate, compare output. Enable GPU only after 100% match on rustc test suite. |
| GPU out-of-memory on large crates | Batch size tuning. If a batch exceeds GPU memory, split it. If total memory exceeded, fallback to CPU. |
| Vulkan driver bugs | Validation layers in debug builds. CI tests on multiple GPU vendors (NVIDIA, AMD, Intel). |
| Wrong generic args resolution | CPU always resolves `Instance` from raw GPU indices. The GPU never interprets types directly. |

**Correctness proof sketch:**
- The GPU shader is a pure function: input = flattened MIR actions, output = `(def_id, args_idx)` pairs.
- The CPU `serialize_batch` visitor is derived from the existing `MirUsedCollector` logic. We can verify it produces the same action stream as the CPU collector would process.
- The CPU `resolve_edge` function uses existing rustc APIs (`Instance::expect_resolve`, etc.) — no new logic.
- Therefore, the GPU path discovers the same graph as the CPU path, just in parallel batches.

---

## 9. Performance Model & Expected Gains

**GPU advantages:**
- Parallel MIR body processing: 64–1024 bodies per dispatch.
- No CPU cache thrashing from walking deep MIR trees.
- Memory bandwidth for reading action streams is GPU-optimized.

**Overheads:**
- CPU serialization: O(N) per body, but N is small (flattened actions only).
- CPU↔GPU memory transfer: PCIe bandwidth. For a batch of 1000 bodies (~1–10 MB), transfer is negligible compared to CPU walk time.
- CPU resolution of edges: O(E) where E is edges discovered. This is unavoidable but parallelizable with `par_iter`.

**Expected speedup (back-of-envelope):**
- Generic-heavy crate with 10,000 instantiations.
- CPU: 10,000 sequential MIR walks × ~0.1 ms = 1.0 s.
- GPU: 10,000 / 1024 ≈ 10 batches × (0.5 ms serialize + 0.1 ms dispatch + 0.2 ms readback) + 0.3 s CPU resolution = ~0.4 s.
- **Net speedup: ~2.5× for monomorphization phase.**
- Since monomorphization is ~5–20% of total compile time for generic-heavy crates, **total compile time improvement: ~5–15%.**

---

## 10. Testing & Validation Strategy

| Test Type | How |
|---|---|
| Correctness fuzzer | Run both CPU and GPU collectors on every `rustc-perf` benchmark. Assert identical `MonoItem` sets. |
| Unit tests | `rustc_gpu_vulkan` crate tests for buffer management, shader dispatch. |
| Integration tests | Add `tests/ui/gpu-mono/` with `-Z gpu-mono=on` flag. |
| Performance regression | `rustc-perf` compare with/without GPU flag. |
| CI coverage | Run on GitHub Actions with software Vulkan (SwiftShader) for basic correctness. Require physical GPU for perf tests. |

---

## 11. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| SPIR-V shader complexity | Medium | Keep shader minimal. Only action matching, no type system. |
| Cross-platform Vulkan availability | Low | Fallback to CPU always works. MoltenVK on macOS. |
| rustc build system integration | Medium | Add to `x.py` build, optional dependency. |
| Maintenance burden | Medium | Isolate in `rustc_gpu_vulkan` crate. Feature-gated. |

---

## 12. Research Findings: Other rustc Phases

### Trait / Type Constraint Solving
After deep research into `rustc_trait_selection`, we found the trait solver is **NOT GPU-parallelizable**. The core algorithm is a stateful, recursive graph-search with unification (union-find). Every step mutates inference variables, and later steps depend on earlier mutations. This makes the core loop inherently sequential.

**Key blocker:** The `ena` unification engine uses path-compression union-find, which is not commutative. Order of operations affects results.

### Pattern Analysis (Exhaustiveness Checking)
The pattern analysis algorithm in `rustc_pattern_analysis` has **parallel potential at the constructor level**:
- `compute_exhaustiveness_and_usefulness` splits constructors for each matrix column
- Each constructor produces an **independent** specialized sub-matrix
- The recursive calls for different constructors are embarrassingly parallel

**However, GPU is the wrong tool:**
- Pattern analysis is only ~1-3% of compile time
- Matrices are small (typically <100 rows)
- Work per constructor is tiny (a few pattern comparisons)
- Complex recursive data structures (`DeconstructedPat`, `Constructor`, `Matrix`) are painful to serialize to GPU
- GPU kernel launch overhead would exceed the computation time

**Better approach:** CPU-level parallelism using `rayon` to parallelize the constructor loop. This would be much simpler and have lower overhead.

### Recommended Future Targets
1. **CPU Parallelism for Pattern Analysis** — Parallelize constructor splitting loop with `rayon`
2. **MIR Dataflow & Borrow Checking** — GPU-accelerated fixed-point iteration (2-5x research precedent)
3. **LLVM Backend Replacement (long-term)** — GPU-native code generator for specific targets

---

## 13. Approval

- **Architecture approved by:** user
- **Date:** 2026-05-14
- **Next step:** Implementation planning via `writing-plans` skill
