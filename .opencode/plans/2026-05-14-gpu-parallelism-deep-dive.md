# GPU Acceleration Deep Dive - Where the Real Parallelism Hides

## Current State Analysis

We've implemented 12 GPU shaders covering:
- Monomorphization (20% of compile time, 2.5x speedup)
- Dataflow analyses (12% of compile time, 3-4x speedup)
- SSA construction (2%)
- Dominance, loops, alias, GVN, induction vars (combined ~5%)

**Total frontend GPU coverage: ~44% of compile time**
**Theoretical max speedup: 1.31x** (Amdahl's Law with 56% serial)

## The Untapped 56%

### 1. Type Checking (15% of compile time)
- **Challenge**: Unification is sequential
- **Opportunity**: Trait resolution, coherence checking, method lookup are parallelizable
- **Potential**: 2-3x on trait-heavy crates (serde, async traits)

### 2. Parsing (5%) + Expansion (8%) = 13%
- **Opportunity**: All files are independent, all macros are independent
- **Potential**: Near-linear speedup with file count. 100 files = ~10x parsing speedup
- **GPU angle**: Parse token streams in parallel on GPU, assemble AST on CPU

### 3. Codegen (20%)
- **Off limits** (user said don't touch LLVM)
- But: we could GPU-accelerate codegen unit partitioning and optimization passes

### 4. The Real Bottleneck We're Missing
**We're doing ONE function at a time on GPU.**

Current approach:
```
for each function:
    CPU: serialize
    GPU: run dataflow
    CPU: read back + apply
    GPU: run DSE
    CPU: read back + apply
    GPU: run copy prop
    CPU: read back + apply
```

**Massive waste**: Each kernel launch costs ~50μs. For 1000 functions, that's 50ms of pure overhead. Plus we're only using 1/1000th of the GPU.

**Better approach**:
```
CPU: serialize 100 functions into one mega-buffer
GPU: run ALL dataflow analyses on ALL 100 functions simultaneously
CPU: read back + apply all at once
```

This gives us:
- **100x more parallelism** (process 100 functions at once)
- **100x less overhead** (1 kernel launch instead of 100)
- **Better GPU utilization** (use thousands of threads instead of tens)

### 5. Multi-Pass Analysis Fusion

Instead of separate kernels for:
- Liveness → Dead store elimination
- Copy facts → Copy propagation
- Const facts → Constant propagation
- Def sites → Reaching definitions

We could run ALL FOUR in a SINGLE kernel dispatch:
```glsl
// One mega-kernel that does 4 analyses per function
layout(local_size_x = 256) in;
// Threads 0-63:   liveness
// Threads 64-127: copy propagation
// Threads 128-191: constant propagation
// Threads 192-255: reaching definitions
```

This is a **4x reduction in kernel launches**.

### 6. GPU-Accelerated Macro Expansion

Rust macros are token transformers. They're embarrassingly parallel:
- Each macro invocation is independent
- Token matching is pattern-based (good for GPU)
- Expansion produces token trees

**Potential**: For crates with heavy macro use (derive macros, async/await), this could be **20-40% of expansion time**.

### 7. GPU-Accelerated Borrow Check

The borrow checker is actually a suite of dataflow analyses:
- Liveness analysis (we have this on GPU)
- Move analysis
- Initialization analysis
- Drop elaboration

These are ALL bitset-based and could run on GPU.

### 8. Theoretical Maximum Revisited

If we add:
- Multi-function batching: +15% total compile time coverage
- Macro expansion: +8% coverage
- Borrow check: +10% coverage
- Type check (trait resolution): +10% coverage
- Parsing: +5% coverage

**New total GPU coverage: ~52% of compile time**
**New theoretical max: 1.45x** (up from 1.31x)

Still not "massive" but getting closer to meaningful.

## Implementation Strategy

1. **Multi-function batching** (biggest immediate win)
2. **Analysis fusion** (reduce overhead)
3. **Macro expansion** (parallel token processing)
4. **Borrow check** (dataflow suite)
5. **Type check trait resolution** (parallel constraint solving)
