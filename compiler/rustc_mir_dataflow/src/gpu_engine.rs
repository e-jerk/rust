use rustc_gpu_vulkan::dataflow::{GpuDataflowConfig, GpuDataflowEngine};
use rustc_gpu_vulkan::{GpuBackend, load_dataflow_shader};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{self, visit::Visitor, BasicBlock, Body, Local, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;

/// GPU-accelerated dataflow engine for bitset-based forward analyses.
///
/// This is an MVP skeleton that provides a minimal integration path
/// between `rustc_mir_dataflow` and the `rustc_gpu_vulkan` compute
/// backend.  Only large functions (>100 basic blocks) are considered,
/// and only a simplified liveness-like analysis is supported.
pub struct GpuEngine<'tcx> {
    _tcx: TyCtxt<'tcx>,
    body: &'tcx Body<'tcx>,
}

impl<'tcx> GpuEngine<'tcx> {
    /// Try to construct a GPU engine for the given function body.
    ///
    /// Returns `None` if the body is too small to benefit from GPU
    /// offload or if the GPU backend is unavailable.
    pub fn new(tcx: TyCtxt<'tcx>, body: &'tcx Body<'tcx>) -> Option<Self> {
        // Only enable for large functions (>100 basic blocks)
        if body.basic_blocks.len() < 100 {
            return None;
        }
        Some(GpuEngine { _tcx: tcx, body })
    }

    /// Run a forward dataflow analysis on GPU.
    ///
    /// For the MVP this is a simplified version that tracks which
    /// locals have storage (are "live" in the loosest sense) using
    /// only `StorageLive` / `StorageDead` effects.
    pub fn run_forward_live_locals(&self) -> Option<Vec<DenseBitSet<Local>>> {
        let backend = GpuBackend::new()?;
        let spirv = load_dataflow_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let bitset_words = (num_locals + 31) / 32;

        // Serialize MIR into GPU-friendly buffers.
        let configs = self.serialize_block_configs();
        let (effects, effects_stride) = self.serialize_effects();

        // Initialize states to bottom (all zeros for this simplified analysis).
        let entry_states: Vec<u32> = vec![0; num_blocks * bitset_words];
        let exit_states: Vec<u32> = vec![0; num_blocks * bitset_words];

        // Upload to GPU-visible buffers.
        let config_buf = backend
            .create_buffer((configs.len() * std::mem::size_of::<GpuDataflowConfig>()) as u64)?;
        config_buf.write(&configs);

        let effects_buf =
            backend.create_buffer((effects.len() * std::mem::size_of::<u32>()) as u64)?;
        effects_buf.write(&effects);

        let entry_buf =
            backend.create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf =
            backend.create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;
        convergence_buf.write(&[0u32]);

        // Fixed-point iteration on GPU.
        let mut round = 0;
        const MAX_ROUNDS: u32 = 100;

        loop {
            gpu.dispatch_round(
                &config_buf,
                &effects_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_blocks as u32,
                bitset_words as u32,
                effects_stride,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }

            // Propagate exit states to successor entry states on CPU.
            let exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            self.propagate_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        // Read back final entry states.
        let final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
        Some(self.parse_results(&final_entry, bitset_words))
    }

    // ------------------------------------------------------------------
    // Serialization helpers
    // ------------------------------------------------------------------

    /// Build a `GpuDataflowConfig` for every basic block.
    fn serialize_block_configs(&self) -> Vec<GpuDataflowConfig> {
        let mut configs = Vec::with_capacity(self.body.basic_blocks.len());
        for block in self.body.basic_blocks.iter() {
            let terminator = block.terminator();
            let successors: Vec<BasicBlock> = terminator.successors().collect();
            let terminator_kind = match terminator.kind {
                TerminatorKind::Goto { .. } => 0,
                TerminatorKind::SwitchInt { .. } => 1,
                TerminatorKind::Return => 2,
                TerminatorKind::Unreachable => 3,
                TerminatorKind::Call { .. } => 4,
                TerminatorKind::Drop { .. } => 5,
                _ => 6,
            };
            configs.push(GpuDataflowConfig {
                statement_count: block.statements.len() as u32,
                terminator_kind,
                successor_count: successors.len() as u32,
                successor_0: successors.get(0).map_or(u32::MAX, |b| b.as_u32()),
                successor_1: successors.get(1).map_or(u32::MAX, |b| b.as_u32()),
            });
        }
        configs
    }

    /// Build a flat effect buffer and the per-block stride.
    ///
    /// Effects are encoded as a single `u32` per statement:
    ///   * lower 24 bits – local index
    ///   * upper 8 bits  – operation kind (0 = nop, 1 = gen/StorageLive, 2 = kill/StorageDead)
    ///
    /// The buffer is padded so every block occupies `effects_stride`
    /// entries, allowing the GPU shader to index it as
    /// `block_idx * effects_stride + stmt_idx`.
    fn serialize_effects(&self) -> (Vec<u32>, u32) {
        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0) as u32;

        let mut effects = vec![0u32; num_blocks * max_statements as usize];

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                let encoded = match stmt.kind {
                    StatementKind::StorageLive(local) => (1u32 << 24) | local.as_u32(),
                    StatementKind::StorageDead(local) => (2u32 << 24) | local.as_u32(),
                    _ => 0u32,
                };
                effects[block_idx.index() * max_statements as usize + stmt_idx] = encoded;
            }
        }

        (effects, max_statements)
    }

    // ------------------------------------------------------------------
    // CPU-side edge propagation
    // ------------------------------------------------------------------

    /// Union the exit state of every block into the entry states of its
    /// successors.  This is the join step for a forward "may" analysis.
    fn propagate_edges(&self, exit_states: &[u32], entry_states: &mut [u32]) {
        let num_blocks = self.body.basic_blocks.len();
        let bitset_words = (self.body.local_decls.len() + 31) / 32;

        for block_idx in 0..num_blocks {
            let bb = BasicBlock::from_usize(block_idx);
            let block = &self.body.basic_blocks[bb];
            let exit_start = block_idx * bitset_words;
            let exit_slice = &exit_states[exit_start..exit_start + bitset_words];

            for succ in block.terminator().successors() {
                let succ_idx = succ.index();
                let entry_start = succ_idx * bitset_words;
                for w in 0..bitset_words {
                    entry_states[entry_start + w] |= exit_slice[w];
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Result parsing
    // ------------------------------------------------------------------

    /// Convert flat `u32` words back into per-block `DenseBitSet<Local>`.
    fn parse_results(&self, flat_states: &[u32], bitset_words: usize) -> Vec<DenseBitSet<Local>> {
        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let mut results = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * bitset_words;
            let words = &flat_states[start..start + bitset_words];
            let mut bitset = DenseBitSet::new_empty(num_locals);

            for (word_idx, &word) in words.iter().enumerate() {
                if word == 0 {
                    continue;
                }
                let base_local = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let local_idx = base_local + bit;
                        if local_idx < num_locals {
                            bitset.insert(Local::from_usize(local_idx));
                        }
                    }
                }
            }
            results.push(bitset);
        }
        results
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Dead Store Elimination (backward liveness)
    // ------------------------------------------------------------------

    /// Run backward liveness analysis on GPU to identify dead stores.
    ///
    /// Returns a vector of (block, statement_idx) pairs indicating dead stores.
    /// Only runs for large functions (>50 basic blocks) to amortize GPU overhead.
    pub fn run_backward_liveness_for_dse(&self) -> Option<Vec<(BasicBlock, usize)>> {
        // Use lower threshold for DSE since backward analysis is more expensive on CPU
        if self.body.basic_blocks.len() < 50 {
            return None;
        }

        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_dead_store_elim_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let bitset_words = (num_locals + 31) / 32;

        // Serialize for backward analysis: assignments = KILL, uses = GEN
        let (configs, effects, effects_stride) = self.serialize_backward_effects();

        // Initialize: all locals are dead at function exit (except return value)
        let mut exit_states: Vec<u32> = vec![0; num_blocks * bitset_words];
        let entry_states: Vec<u32> = vec![0; num_blocks * bitset_words];

        // Mark return value as live at return blocks
        for (bb, block) in self.body.basic_blocks.iter_enumerated() {
            if matches!(block.terminator().kind, TerminatorKind::Return) {
                let start = bb.index() * bitset_words;
                // Local_0 is the return value
                exit_states[start] |= 1u32;
            }
        }

        // Upload to GPU
        let config_buf = backend
            .create_buffer((configs.len() * std::mem::size_of::<u32>()) as u64)?;
        config_buf.write(&configs);

        let effects_buf =
            backend.create_buffer((effects.len() * std::mem::size_of::<u32>()) as u64)?;
        effects_buf.write(&effects);

        let entry_buf =
            backend.create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf =
            backend.create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        // Backward fixed-point iteration on GPU
        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            // Reset convergence flag
            convergence_buf.write(&[0u32]);

            // Propagate entry states to predecessor exit states on CPU
            let entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            let mut exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            self.propagate_backward_edges(&entry_data, &mut exit_data);
            exit_buf.write(&exit_data);

            // GPU computes entry states from exit states
            gpu.dispatch_round(
                &config_buf,
                &effects_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_blocks as u32,
                bitset_words as u32,
                effects_stride,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }
        }

        // Read back final entry states
        let final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
        let live_sets = self.parse_results(&final_entry, bitset_words);

        // Identify dead stores on CPU
        Some(self.identify_dead_stores(&live_sets))
    }

    /// Serialize block configs and effects for backward liveness.
    /// Effects: KILL=1 for assignments, GEN=2 for uses.
    fn serialize_backward_effects(&self) -> (Vec<u32>, Vec<u32>, u32) {
        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0) as u32;

        let mut configs = Vec::with_capacity(num_blocks * 4);
        let mut effects = vec![0u32; num_blocks * max_statements as usize];

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let terminator = block.terminator();
            let successors: Vec<BasicBlock> = terminator.successors().collect();
            let predecessors: Vec<BasicBlock> = self
                .body
                .basic_blocks
                .predecessors()[block_idx]
                .iter()
                .copied()
                .collect();

            // Config encoding: [stmt_count, terminator_kind, successors, predecessors]
            let terminator_kind = match terminator.kind {
                TerminatorKind::Goto { .. } => 0,
                TerminatorKind::SwitchInt { .. } => 1,
                TerminatorKind::Return => 2,
                TerminatorKind::Unreachable => 3,
                TerminatorKind::Call { .. } => 4,
                TerminatorKind::Drop { .. } => 5,
                _ => 6,
            };

            configs.push(block.statements.len() as u32);
            configs.push(terminator_kind);
            configs.push(
                (successors.len() as u32 & 0xFFFF)
                    | ((successors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF) << 16),
            );
            configs.push(
                (predecessors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF)
                    | ((predecessors.len() as u32 & 0xFFFF) << 16),
            );

            // Encode effects for each statement
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                let encoded = match &stmt.kind {
                    StatementKind::Assign((place, _)) => {
                        // Assignment KILLS the local (it's no longer live before this)
                        let local = place.local;
                        (1u32 << 24) | local.as_u32()
                    }
                    StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => 0,
                    _ => {
                        // For non-assignment statements, encode as GEN for the statement
                        // The GPU shader will process them; detailed local scanning is done on CPU
                        0
                    }
                };
                effects[block_idx.index() * max_statements as usize + stmt_idx] = encoded;
            }
        }

        (configs, effects, max_statements)
    }

    /// Propagate entry states to predecessor exit states for backward analysis.
    fn propagate_backward_edges(&self, entry_states: &[u32], exit_states: &mut [u32]) {
        let num_blocks = self.body.basic_blocks.len();
        let bitset_words = (self.body.local_decls.len() + 31) / 32;

        for block_idx in 0..num_blocks {
            let bb = BasicBlock::from_usize(block_idx);
            let preds = &self.body.basic_blocks.predecessors()[bb];

            let entry_start = block_idx * bitset_words;
            let entry_slice = &entry_states[entry_start..entry_start + bitset_words];

            // Union this block's entry into all predecessors' exit states
            for &pred in preds {
                let pred_idx = pred.index();
                let exit_start = pred_idx * bitset_words;
                for w in 0..bitset_words {
                    exit_states[exit_start + w] |= entry_slice[w];
                }
            }
        }
    }

    /// Identify dead stores using computed liveness information.
    fn identify_dead_stores(
        &self,
        live_at_entry: &[DenseBitSet<Local>],
    ) -> Vec<(BasicBlock, usize)> {
        let mut dead_stores = Vec::new();

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            // Compute live set at each statement position by applying effects forward
            let mut live = live_at_entry[block_idx.index()].clone();
            let _bitset_words = (self.body.local_decls.len() + 31) / 32;

            // Walk statements backward to determine liveness at each point
            for stmt_idx in (0..block.statements.len()).rev() {
                let stmt = &block.statements[stmt_idx];

                if let StatementKind::Assign((place, _)) = &stmt.kind {
                    let local = place.local;

                    // If the assigned local is NOT live after this statement,
                    // the store is dead
                    if !live.contains(local) {
                        dead_stores.push((block_idx, stmt_idx));
                    }

                    // Apply KILL effect: local is no longer live before this assignment
                    live.remove(local);
                }

                // Apply GEN effects: mark all locals used in the statement as live
                // Use a visitor to find all locals in the statement
                struct LocalCollector<'a> {
                    live: &'a mut DenseBitSet<Local>,
                }
                impl<'tcx> mir::visit::Visitor<'tcx> for LocalCollector<'_> {
                    fn visit_local(&mut self, local: Local, _ctx: mir::visit::PlaceContext, _loc: mir::Location) {
                        self.live.insert(local);
                    }
                }
                let mut collector = LocalCollector { live: &mut live };
                collector.visit_statement(stmt, mir::Location { block: block_idx, statement_index: stmt_idx });
            }
        }

        dead_stores
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Copy Propagation (forward dataflow)
    // ------------------------------------------------------------------

    /// Run copy propagation analysis on GPU.
    ///
    /// Returns a vector of (block, statement_idx, dst_local, src_local) for copy statements
    /// that can be propagated.
    pub fn run_copy_propagation(&self) -> Option<Vec<(BasicBlock, usize, Local, Local)>> {
        if self.body.basic_blocks.len() < 50 {
            return None;
        }

        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_copy_prop_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();

        // Serialize copy facts
        let (configs, copy_facts, facts_stride) = self.serialize_copy_facts();

        // Initialize entry states to 0 (no known copies)
        let entry_states: Vec<u32> = vec![0; num_blocks * num_locals];
        let exit_states: Vec<u32> = vec![0; num_blocks * num_locals];

        let config_buf = backend
            .create_buffer((configs.len() * std::mem::size_of::<u32>()) as u64)?;
        config_buf.write(&configs);

        let facts_buf = backend
            .create_buffer((copy_facts.len() * std::mem::size_of::<u32>()) as u64)?;
        facts_buf.write(&copy_facts);

        let entry_buf = backend
            .create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf = backend
            .create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        // Forward fixed-point iteration on GPU
        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            // GPU computes exit states from entry states
            gpu.dispatch_round(
                &config_buf,
                &facts_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_blocks as u32,
                num_locals as u32,
                facts_stride,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }

            // Propagate exit states to successor entry states on CPU
            let exit_data: Vec<u32> = exit_buf.read(num_blocks * num_locals);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * num_locals);
            self.propagate_copy_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        // Read back final states
        let final_entry: Vec<u32> = entry_buf.read(num_blocks * num_locals);
        Some(self.identify_copy_propagations(&final_entry, &copy_facts, facts_stride))
    }

    /// Serialize block configs and copy facts for copy propagation.
    fn serialize_copy_facts(&self) -> (Vec<u32>, Vec<u32>, u32) {
        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0) as u32;

        let mut configs = Vec::with_capacity(num_blocks * 4);
        let mut facts = vec![0xFFFFFFFFu32; num_blocks * max_statements as usize];

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let terminator = block.terminator();
            let successors: Vec<BasicBlock> = terminator.successors().collect();

            let terminator_kind = match terminator.kind {
                TerminatorKind::Goto { .. } => 0,
                TerminatorKind::SwitchInt { .. } => 1,
                TerminatorKind::Return => 2,
                TerminatorKind::Unreachable => 3,
                TerminatorKind::Call { .. } => 4,
                TerminatorKind::Drop { .. } => 5,
                _ => 6,
            };

            configs.push(block.statements.len() as u32);
            configs.push(terminator_kind);
            configs.push(
                (successors.len() as u32 & 0xFFFF)
                    | ((successors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF) << 16),
            );
            configs.push(
                successors.get(1).map_or(u32::MAX, |b| b.as_u32() & 0xFFFF),
            );

            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((place, rvalue)) = &stmt.kind {
                    if let rustc_middle::mir::Rvalue::Use(rustc_middle::mir::Operand::Copy(src) |
                        rustc_middle::mir::Operand::Move(src), _) = rvalue {
                        if let Some(dst_local) = place.as_local() {
                            if let Some(src_local) = src.as_local() {
                                let encoded = ((dst_local.as_u32() & 0xFFFF) << 16)
                                    | (src_local.as_u32() & 0xFFFF);
                                facts[block_idx.index() * max_statements as usize + stmt_idx] = encoded;
                            }
                        }
                    }
                }
            }
        }

        (configs, facts, max_statements)
    }

    /// Propagate exit copy states to successor entry states.
    fn propagate_copy_edges(&self, exit_states: &[u32], entry_states: &mut [u32]) {
        let num_locals = self.body.local_decls.len();

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let exit_start = block_idx.index() * num_locals;

            for succ in block.terminator().successors() {
                let succ_idx = succ.index();
                let entry_start = succ_idx * num_locals;

                for l in 0..num_locals {
                    let exit_val = exit_states[exit_start + l];
                    let entry_val = entry_states[entry_start + l];

                    // Join: if both agree, keep the value; otherwise, mark as unknown (0)
                    if entry_val == 0 {
                        entry_states[entry_start + l] = exit_val;
                    } else if entry_val != exit_val {
                        entry_states[entry_start + l] = 0; // conflicting sources -> unknown
                    }
                }
            }
        }
    }

    /// Identify copy statements that can be propagated.
    fn identify_copy_propagations(
        &self,
        entry_states: &[u32],
        _copy_facts: &[u32],
        _facts_stride: u32,
    ) -> Vec<(BasicBlock, usize, Local, Local)> {
        let num_locals = self.body.local_decls.len();
        let mut propagations = Vec::new();

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let entry_start = block_idx.index() * num_locals;

            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((place, rvalue)) = &stmt.kind {
                    if let rustc_middle::mir::Rvalue::Use(rustc_middle::mir::Operand::Copy(src) |
                        rustc_middle::mir::Operand::Move(src), _) = rvalue {
                        if let Some(dst_local) = place.as_local() {
                            if let Some(src_local) = src.as_local() {
                                // Check if src is itself a copy of something else
                                let src_copy_source = entry_states[entry_start + src_local.as_usize()];
                                if src_copy_source > 0 {
                                    // src holds a copy of (src_copy_source - 1)
                                    // We can propagate: dst = original_source
                                    let original = Local::from_usize((src_copy_source - 1) as usize);
                                    propagations.push((block_idx, stmt_idx, dst_local, original));
                                }
                            }
                        }
                    }
                }
            }
        }

        propagations
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Constant Propagation (forward dataflow)
    // ------------------------------------------------------------------

    /// Run constant propagation analysis on GPU.
    ///
    /// Returns a vector of (block, statement_idx, local, constant_value) for assignments
    /// that can be constant-folded.
    pub fn run_constant_propagation(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
        if self.body.basic_blocks.len() < 50 {
            return None;
        }

        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_const_prop_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();

        // Serialize constant facts
        let (configs, const_facts, facts_stride) = self.serialize_const_facts();

        // Initialize entry states to 0 (no known constants)
        let entry_states: Vec<u32> = vec![0; num_blocks * num_locals];
        let exit_states: Vec<u32> = vec![0; num_blocks * num_locals];

        let config_buf = backend
            .create_buffer((configs.len() * std::mem::size_of::<u32>()) as u64)?;
        config_buf.write(&configs);

        let facts_buf = backend
            .create_buffer((const_facts.len() * std::mem::size_of::<u32>()) as u64)?;
        facts_buf.write(&const_facts);

        let entry_buf = backend
            .create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf = backend
            .create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        // Forward fixed-point iteration on GPU
        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            // GPU computes exit states from entry states
            gpu.dispatch_round(
                &config_buf,
                &facts_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_blocks as u32,
                num_locals as u32,
                facts_stride,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }

            // Propagate exit states to successor entry states on CPU
            let exit_data: Vec<u32> = exit_buf.read(num_blocks * num_locals);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * num_locals);
            self.propagate_const_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        // Read back final states
        let final_entry: Vec<u32> = entry_buf.read(num_blocks * num_locals);
        Some(self.identify_constant_propagations(&final_entry))
    }

    /// Serialize block configs and constant facts for constant propagation.
    fn serialize_const_facts(&self) -> (Vec<u32>, Vec<u32>, u32) {
        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0) as u32;

        let mut configs = Vec::with_capacity(num_blocks * 4);
        let mut facts = vec![0xFFFFFFFFu32; num_blocks * max_statements as usize];

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let terminator = block.terminator();
            let successors: Vec<BasicBlock> = terminator.successors().collect();

            let terminator_kind = match terminator.kind {
                TerminatorKind::Goto { .. } => 0,
                TerminatorKind::SwitchInt { .. } => 1,
                TerminatorKind::Return => 2,
                TerminatorKind::Unreachable => 3,
                TerminatorKind::Call { .. } => 4,
                TerminatorKind::Drop { .. } => 5,
                _ => 6,
            };

            configs.push(block.statements.len() as u32);
            configs.push(terminator_kind);
            configs.push(
                (successors.len() as u32 & 0xFFFF)
                    | ((successors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF) << 16),
            );
            configs.push(
                successors.get(1).map_or(u32::MAX, |b| b.as_u32() & 0xFFFF),
            );

            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((place, rvalue)) = &stmt.kind {
                    if let rustc_middle::mir::Rvalue::Use(rustc_middle::mir::Operand::Constant(cst), _) = rvalue {
                        if let Some(dst_local) = place.as_local() {
                            // Try to extract a small integer constant (0-65534)
                            let value = match cst.const_ {
                                rustc_middle::mir::Const::Val(val, _) => {
                                    if let Some(scalar) = val.try_to_scalar_int() {
                                        let v = scalar.to_u32();
                                        if v <= 65534 { v } else { continue; }
                                    } else {
                                        continue;
                                    }
                                }
                                rustc_middle::mir::Const::Ty(_, ty_const) => {
                                    if let Some(scalar) = ty_const.try_to_leaf() {
                                        let v = scalar.to_u32();
                                        if v <= 65534 { v } else { continue; }
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            };
                            let encoded = ((dst_local.as_u32() & 0xFFFF) << 16) | (value & 0xFFFF);
                            facts[block_idx.index() * max_statements as usize + stmt_idx] = encoded;
                        }
                    }
                }
            }
        }

        (configs, facts, max_statements)
    }

    /// Propagate exit constant states to successor entry states.
    fn propagate_const_edges(&self, exit_states: &[u32], entry_states: &mut [u32]) {
        let num_locals = self.body.local_decls.len();

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let exit_start = block_idx.index() * num_locals;

            for succ in block.terminator().successors() {
                let succ_idx = succ.index();
                let entry_start = succ_idx * num_locals;

                for l in 0..num_locals {
                    let exit_val = exit_states[exit_start + l];
                    let entry_val = entry_states[entry_start + l];

                    // Join: if both agree, keep the value; otherwise, mark as unknown (0)
                    if entry_val == 0 {
                        entry_states[entry_start + l] = exit_val;
                    } else if entry_val != exit_val {
                        entry_states[entry_start + l] = 0; // conflicting constants -> unknown
                    }
                }
            }
        }
    }

    /// Identify constant assignments that can be propagated.
    fn identify_constant_propagations(
        &self,
        entry_states: &[u32],
    ) -> Vec<(BasicBlock, usize, Local, u32)> {
        let num_locals = self.body.local_decls.len();
        let mut propagations = Vec::new();

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let entry_start = block_idx.index() * num_locals;

            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                // Check if this statement uses a local that holds a constant
                // For simplicity, we look for Rvalues that are copies of locals
                if let StatementKind::Assign((place, rvalue)) = &stmt.kind {
                    if let rustc_middle::mir::Rvalue::Use(rustc_middle::mir::Operand::Copy(src) |
                        rustc_middle::mir::Operand::Move(src), _) = rvalue {
                        if let Some(src_local) = src.as_local() {
                            let const_val = entry_states[entry_start + src_local.as_usize()];
                            if const_val > 0 {
                                // src holds constant (const_val - 1)
                                let value = const_val - 1;
                                if let Some(dst_local) = place.as_local() {
                                    propagations.push((block_idx, stmt_idx, dst_local, value));
                                }
                            }
                        }
                    }
                }
            }
        }

        propagations
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Reaching Definitions (forward dataflow)
    // ------------------------------------------------------------------

    /// Run reaching definitions analysis on GPU.
    ///
    /// Returns a vector of (block, statement_idx, local, def_id) for all definition sites.
    pub fn run_reaching_definitions(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
        if self.body.basic_blocks.len() < 50 {
            return None;
        }

        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_reaching_defs_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let _num_locals = self.body.local_decls.len();
        let max_defs = ((num_blocks * 16).min(1024)) as u32; // Cap at 1024 definitions
        let bitset_words = ((max_defs + 31) / 32) as usize;

        // Serialize definition facts
        let (configs, def_facts, facts_stride, def_map) = self.serialize_def_facts(max_defs);

        // Initialize entry states to 0 (no definitions reach)
        let entry_states: Vec<u32> = vec![0; num_blocks * bitset_words];
        let exit_states: Vec<u32> = vec![0; num_blocks * bitset_words];

        let config_buf = backend
            .create_buffer((configs.len() * std::mem::size_of::<u32>()) as u64)?;
        config_buf.write(&configs);

        let facts_buf = backend
            .create_buffer((def_facts.len() * std::mem::size_of::<u32>()) as u64)?;
        facts_buf.write(&def_facts);

        let entry_buf = backend
            .create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf = backend
            .create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        // Forward fixed-point iteration on GPU
        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            // GPU computes exit states from entry states
            gpu.dispatch_round(
                &config_buf,
                &facts_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_blocks as u32,
                bitset_words as u32,
                facts_stride,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }

            // Propagate exit states to successor entry states on CPU
            let exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            self.propagate_def_edges(&exit_data, &mut entry_data, bitset_words);
            entry_buf.write(&entry_data);
        }

        // Read back final states
        let _final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);

        // Return the definition map
        Some(def_map)
    }

    /// Serialize block configs and definition facts for reaching definitions.
    fn serialize_def_facts(&self, max_defs: u32) -> (Vec<u32>, Vec<u32>, u32, Vec<(BasicBlock, usize, Local, u32)>) {
        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0) as u32;

        let mut configs = Vec::with_capacity(num_blocks * 4);
        let mut facts = vec![0xFFFFFFFFu32; num_blocks * max_statements as usize];
        let mut def_map = Vec::new();
        let mut next_def_id: u32 = 0;

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let terminator = block.terminator();
            let successors: Vec<BasicBlock> = terminator.successors().collect();

            let terminator_kind = match terminator.kind {
                TerminatorKind::Goto { .. } => 0,
                TerminatorKind::SwitchInt { .. } => 1,
                TerminatorKind::Return => 2,
                TerminatorKind::Unreachable => 3,
                TerminatorKind::Call { .. } => 4,
                TerminatorKind::Drop { .. } => 5,
                _ => 6,
            };

            configs.push(block.statements.len() as u32);
            configs.push(terminator_kind);
            configs.push(
                (successors.len() as u32 & 0xFFFF)
                    | ((successors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF) << 16),
            );
            configs.push(
                successors.get(1).map_or(u32::MAX, |b| b.as_u32() & 0xFFFF),
            );

            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((place, _)) = &stmt.kind {
                    if let Some(dst_local) = place.as_local() {
                        if next_def_id < max_defs {
                            let def_id = next_def_id;
                            next_def_id += 1;
                            let encoded = ((dst_local.as_u32() & 0xFFFF) << 16) | (def_id & 0xFFFF);
                            facts[block_idx.index() * max_statements as usize + stmt_idx] = encoded;
                            def_map.push((block_idx, stmt_idx, dst_local, def_id));
                        }
                    }
                }
            }
        }

        (configs, facts, max_statements, def_map)
    }

    /// Propagate exit definition states to successor entry states.
    fn propagate_def_edges(&self, exit_states: &[u32], entry_states: &mut [u32], bitset_words: usize) {
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let exit_start = block_idx.index() * bitset_words;

            for succ in block.terminator().successors() {
                let succ_idx = succ.index();
                let entry_start = succ_idx * bitset_words;

                for w in 0..bitset_words {
                    entry_states[entry_start + w] |= exit_states[exit_start + w];
                }
            }
        }
    }
}
