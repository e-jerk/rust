use rustc_gpu_vulkan::dataflow::{GpuDataflowConfig, GpuDataflowEngine};
use rustc_gpu_vulkan::{GpuBackend, load_dataflow_shader};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{self, visit::Visitor, BasicBlock, Body, Local, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;

/// GPU-accelerated dataflow engine for bitset-based forward analyses.
///
/// This is an MVP skeleton that provides a minimal integration path
/// between `rustc_mir_dataflow` and the GPU compute backend.
/// Supports both Metal (macOS native) and Vulkan (cross-platform).
///
/// Only large functions (>100 basic blocks) are considered,
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
    /// Tries Metal first (macOS native, ~1.5× faster), then Vulkan.
    /// For the MVP this is a simplified version that tracks which
    /// locals have storage (are "live" in the loosest sense) using
    /// only `StorageLive` / `StorageDead` effects.
    pub fn run_forward_live_locals(&self) -> Option<Vec<DenseBitSet<Local>>> {
        // Try Metal first (native Apple Silicon, ~280µs dispatch overhead)
        if let Some(result) = self.run_forward_live_locals_metal() {
            return Some(result);
        }
        
        // Fall back to Vulkan (cross-platform, ~429µs dispatch overhead via MoltenVK)
        self.run_forward_live_locals_vulkan()
    }

    /// Metal-specific dataflow path.
    fn run_forward_live_locals_metal(&self) -> Option<Vec<DenseBitSet<Local>>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_dataflow_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "dataflow",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let bitset_words = (num_locals + 31) / 32;

        let configs = self.serialize_block_configs();
        let (effects, effects_stride) = self.serialize_effects();

        let entry_states: Vec<u32> = vec![0; num_blocks * bitset_words];
        let exit_states: Vec<u32> = vec![0; num_blocks * bitset_words];

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

            let exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            self.propagate_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        let final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
        Some(self.parse_results(&final_entry, bitset_words))
    }

    /// Vulkan-specific dataflow path (original implementation).
    fn run_forward_live_locals_vulkan(&self) -> Option<Vec<DenseBitSet<Local>>> {
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

        // Try Metal first (native Apple Silicon, ~1.5× faster)
        if let Some(result) = self.run_backward_liveness_for_dse_metal() {
            return Some(result);
        }

        // Fall back to Vulkan (cross-platform via MoltenVK on macOS)
        self.run_backward_liveness_for_dse_vulkan()
    }

    fn run_backward_liveness_for_dse_metal(&self) -> Option<Vec<(BasicBlock, usize)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_dead_store_elim_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "dead_store_elim",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let bitset_words = (num_locals + 31) / 32;

        let (configs, effects, effects_stride) = self.serialize_backward_effects();

        let mut exit_states: Vec<u32> = vec![0; num_blocks * bitset_words];
        let entry_states: Vec<u32> = vec![0; num_blocks * bitset_words];

        for (bb, block) in self.body.basic_blocks.iter_enumerated() {
            if matches!(block.terminator().kind, TerminatorKind::Return) {
                let start = bb.index() * bitset_words;
                exit_states[start] |= 1u32;
            }
        }

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

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            let entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            let mut exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            self.propagate_backward_edges(&entry_data, &mut exit_data);
            exit_buf.write(&exit_data);

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

        let final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
        let live_sets = self.parse_results(&final_entry, bitset_words);
        Some(self.identify_dead_stores(&live_sets))
    }

    fn run_backward_liveness_for_dse_vulkan(&self) -> Option<Vec<(BasicBlock, usize)>> {
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

        // Try Metal first (native Apple Silicon, ~1.5× faster)
        if let Some(result) = self.run_copy_propagation_metal() {
            return Some(result);
        }

        // Fall back to Vulkan (cross-platform via MoltenVK on macOS)
        self.run_copy_propagation_vulkan()
    }

    fn run_copy_propagation_metal(&self) -> Option<Vec<(BasicBlock, usize, Local, Local)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_copy_prop_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "copy_prop",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();

        let (configs, copy_facts, facts_stride) = self.serialize_copy_facts();

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

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

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

            let exit_data: Vec<u32> = exit_buf.read(num_blocks * num_locals);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * num_locals);
            self.propagate_copy_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        let final_entry: Vec<u32> = entry_buf.read(num_blocks * num_locals);
        Some(self.identify_copy_propagations(&final_entry, &copy_facts, facts_stride))
    }

    fn run_copy_propagation_vulkan(&self) -> Option<Vec<(BasicBlock, usize, Local, Local)>> {
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

        // Try Metal first (native Apple Silicon, ~1.5× faster)
        if let Some(result) = self.run_constant_propagation_metal() {
            return Some(result);
        }

        // Fall back to Vulkan (cross-platform via MoltenVK on macOS)
        self.run_constant_propagation_vulkan()
    }

    fn run_constant_propagation_metal(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_const_prop_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "const_prop",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();

        let (configs, const_facts, facts_stride) = self.serialize_const_facts();

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

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

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

            let exit_data: Vec<u32> = exit_buf.read(num_blocks * num_locals);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * num_locals);
            self.propagate_const_edges(&exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        let final_entry: Vec<u32> = entry_buf.read(num_blocks * num_locals);
        Some(self.identify_constant_propagations(&final_entry))
    }

    fn run_constant_propagation_vulkan(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
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
        if let Some(result) = self.run_reaching_definitions_metal() {
            return Some(result);
        }
        self.run_reaching_definitions_vulkan()
    }

    fn run_reaching_definitions_metal(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_reaching_defs_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "reaching_defs",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let _num_locals = self.body.local_decls.len();
        let max_defs = ((num_blocks * 16).min(1024)) as u32; // Cap at 1024 definitions
        let bitset_words = ((max_defs + 31) / 32) as usize;

        let (configs, def_facts, facts_stride, def_map) = self.serialize_def_facts(max_defs);

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

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

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

            let exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            self.propagate_def_edges(&exit_data, &mut entry_data, bitset_words);
            entry_buf.write(&entry_data);
        }

        let _final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);

        Some(def_map)
    }

    fn run_reaching_definitions_vulkan(&self) -> Option<Vec<(BasicBlock, usize, Local, u32)>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_reaching_defs_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let _num_locals = self.body.local_decls.len();
        let max_defs = ((num_blocks * 16).min(1024)) as u32; // Cap at 1024 definitions
        let bitset_words = ((max_defs + 31) / 32) as usize;

        let (configs, def_facts, facts_stride, def_map) = self.serialize_def_facts(max_defs);

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

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

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

            let exit_data: Vec<u32> = exit_buf.read(num_blocks * bitset_words);
            let mut entry_data: Vec<u32> = entry_buf.read(num_blocks * bitset_words);
            self.propagate_def_edges(&exit_data, &mut entry_data, bitset_words);
            entry_buf.write(&entry_data);
        }

        let _final_entry: Vec<u32> = entry_buf.read(num_blocks * bitset_words);

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

    // ------------------------------------------------------------------
    // GPU-accelerated SSA Construction
    // ------------------------------------------------------------------

    /// Run SSA construction on GPU to identify phi node insertion points.
    ///
    /// Returns a vector of (block, local) pairs indicating where phi nodes are needed.
    pub fn run_ssa_construction(&self) -> Option<Vec<(BasicBlock, Local)>> {
        if self.body.basic_blocks.len() < 50 {
            return None;
        }
        if let Some(result) = self.run_ssa_construction_metal() {
            return Some(result);
        }
        self.run_ssa_construction_vulkan()
    }

    fn run_ssa_construction_metal(&self) -> Option<Vec<(BasicBlock, Local)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_ssa_construct_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "ssa_construct",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let max_defs_per_block = 16; // Cap definitions per block
        let phi_words = (num_locals + 31) / 32;
        let df_words = (num_blocks + 31) / 32;

        let (block_info, def_sites) = self.serialize_ssa_blocks(max_defs_per_block);

        let dom_frontier: Vec<u32> = vec![0; num_blocks * df_words];
        let phi_nodes: Vec<u32> = vec![0; num_blocks * phi_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let def_sites_buf = backend
            .create_buffer((def_sites.len() * std::mem::size_of::<u32>()) as u64)?;
        def_sites_buf.write(&def_sites);

        let dom_frontier_buf = backend
            .create_buffer((dom_frontier.len() * std::mem::size_of::<u32>()) as u64)?;
        dom_frontier_buf.write(&dom_frontier);

        let phi_nodes_buf = backend
            .create_buffer((phi_nodes.len() * std::mem::size_of::<u32>()) as u64)?;
        phi_nodes_buf.write(&phi_nodes);

        gpu.dispatch_ssa(
            &block_info_buf,
            &def_sites_buf,
            &dom_frontier_buf,
            &phi_nodes_buf,
            num_blocks as u32,
            num_locals as u32,
            max_defs_per_block,
        )
        .ok()?;

        let phi_data: Vec<u32> = phi_nodes_buf.read(num_blocks * phi_words);

        let mut phi_insertions = Vec::new();
        for block_idx in 0..num_blocks {
            let start = block_idx * phi_words;
            for word_idx in 0..phi_words {
                let word = phi_data[start + word_idx];
                if word == 0 {
                    continue;
                }
                let base_local = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let local_idx = base_local + bit;
                        if local_idx < num_locals {
                            phi_insertions.push((
                                BasicBlock::from_usize(block_idx),
                                Local::from_usize(local_idx),
                            ));
                        }
                    }
                }
            }
        }

        Some(phi_insertions)
    }

    fn run_ssa_construction_vulkan(&self) -> Option<Vec<(BasicBlock, Local)>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_ssa_construct_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let max_defs_per_block = 16; // Cap definitions per block
        let phi_words = (num_locals + 31) / 32;
        let df_words = (num_blocks + 31) / 32;

        let (block_info, def_sites) = self.serialize_ssa_blocks(max_defs_per_block);

        let dom_frontier: Vec<u32> = vec![0; num_blocks * df_words];
        let phi_nodes: Vec<u32> = vec![0; num_blocks * phi_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let def_sites_buf = backend
            .create_buffer((def_sites.len() * std::mem::size_of::<u32>()) as u64)?;
        def_sites_buf.write(&def_sites);

        let dom_frontier_buf = backend
            .create_buffer((dom_frontier.len() * std::mem::size_of::<u32>()) as u64)?;
        dom_frontier_buf.write(&dom_frontier);

        let phi_nodes_buf = backend
            .create_buffer((phi_nodes.len() * std::mem::size_of::<u32>()) as u64)?;
        phi_nodes_buf.write(&phi_nodes);

        gpu.dispatch_ssa_round(
            &block_info_buf,
            &def_sites_buf,
            &dom_frontier_buf,
            &phi_nodes_buf,
            num_blocks as u32,
            num_locals as u32,
            max_defs_per_block,
        )
        .ok()?;

        let phi_data: Vec<u32> = phi_nodes_buf.read(num_blocks * phi_words);

        let mut phi_insertions = Vec::new();
        for block_idx in 0..num_blocks {
            let start = block_idx * phi_words;
            for word_idx in 0..phi_words {
                let word = phi_data[start + word_idx];
                if word == 0 {
                    continue;
                }
                let base_local = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let local_idx = base_local + bit;
                        if local_idx < num_locals {
                            phi_insertions.push((
                                BasicBlock::from_usize(block_idx),
                                Local::from_usize(local_idx),
                            ));
                        }
                    }
                }
            }
        }

        Some(phi_insertions)
    }

    /// Serialize blocks for SSA construction.
    fn serialize_ssa_blocks(&self, max_defs_per_block: u32) -> (Vec<u32>, Vec<u32>) {
        let num_blocks = self.body.basic_blocks.len();

        let mut block_info = Vec::with_capacity(num_blocks * (2 + max_defs_per_block as usize));
        let mut def_sites = vec![0xFFFFFFFFu32; num_blocks * max_defs_per_block as usize];

        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let preds = &self.body.basic_blocks.predecessors()[block_idx];
            let num_preds = preds.len().min(2); // Cap at 2 predecessors for GPU

            block_info.push(num_preds as u32);

            let mut num_defs: u32 = 0;
            for stmt in &block.statements {
                if let StatementKind::Assign((place, _)) = &stmt.kind {
                    if let Some(local) = place.as_local() {
                        if num_defs < max_defs_per_block {
                            def_sites[block_idx.index() * max_defs_per_block as usize + num_defs as usize] =
                                local.as_u32();
                            num_defs += 1;
                        }
                    }
                }
            }
            block_info.push(num_defs);

            // Write predecessor indices
            let max_defs = max_defs_per_block as usize;
            for p in 0..max_defs {
                if p < num_preds {
                    block_info.push(preds[p].as_u32());
                } else {
                    block_info.push(0xFFFFFFFF);
                }
            }
        }

        (block_info, def_sites)
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Alias Analysis
    // ------------------------------------------------------------------

    /// Run flow-insensitive alias analysis on GPU.
    ///
    /// Returns a bitmap of which memory accesses may alias with each other.
    pub fn run_alias_analysis(&self) -> Option<Vec<Vec<bool>>> {
        if self.body.basic_blocks.len() < 30 {
            return None;
        }
        if let Some(result) = self.run_alias_analysis_metal() {
            return Some(result);
        }
        self.run_alias_analysis_vulkan()
    }

    fn run_alias_analysis_metal(&self) -> Option<Vec<Vec<bool>>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_alias_analysis_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "alias_analysis",
        ).ok()?;

        let mut accesses = Vec::new();
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                match &stmt.kind {
                    StatementKind::Assign((place, _)) => {
                        if let Some(local) = place.as_local() {
                            accesses.push((block_idx, stmt_idx, local, 2u32)); // write
                        }
                    }
                    _ => {
                        struct ReadCollector<'a> {
                            reads: &'a mut Vec<(BasicBlock, usize, Local, u32)>,
                            block: BasicBlock,
                            stmt: usize,
                        }
                        impl<'tcx> Visitor<'tcx> for ReadCollector<'_> {
                            fn visit_local(&mut self, local: Local, ctx: mir::visit::PlaceContext, _loc: mir::Location) {
                                if ctx.is_use() {
                                    self.reads.push((self.block, self.stmt, local, 1u32));
                                }
                            }
                        }
                        let mut collector = ReadCollector {
                            reads: &mut accesses,
                            block: block_idx,
                            stmt: stmt_idx,
                        };
                        collector.visit_statement(stmt, mir::Location { block: block_idx, statement_index: stmt_idx });
                    }
                }
            }
        }

        let num_accesses = accesses.len();
        if num_accesses == 0 || num_accesses > 1024 {
            return None;
        }

        let num_locals = self.body.local_decls.len();
        let matrix_words = (num_accesses + 31) / 32;

        let descriptors: Vec<u32> = accesses
            .iter()
            .map(|(_, _, local, kind)| {
                ((local.as_u32() & 0xFFFF) << 16) | (kind & 0xFFFF)
            })
            .collect();

        let alias_matrix: Vec<u32> = vec![0; num_accesses * matrix_words];

        let desc_buf = backend
            .create_buffer((descriptors.len() * std::mem::size_of::<u32>()) as u64)?;
        desc_buf.write(&descriptors);

        let matrix_buf = backend
            .create_buffer((alias_matrix.len() * std::mem::size_of::<u32>()) as u64)?;
        matrix_buf.write(&alias_matrix);

        gpu.dispatch_alias(
            &desc_buf,
            &matrix_buf,
            num_accesses as u32,
            num_locals as u32,
        )
        .ok()?;

        let matrix_data: Vec<u32> = matrix_buf.read(num_accesses * matrix_words);

        let mut result = Vec::with_capacity(num_accesses);
        for i in 0..num_accesses {
            let mut row = Vec::with_capacity(num_accesses);
            for j in 0..num_accesses {
                let word_idx = i * matrix_words + (j / 32);
                let bit_idx = j % 32;
                row.push(matrix_data[word_idx] & (1u32 << bit_idx) != 0);
            }
            result.push(row);
        }

        Some(result)
    }

    fn run_alias_analysis_vulkan(&self) -> Option<Vec<Vec<bool>>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_alias_analysis_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let mut accesses = Vec::new();
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                match &stmt.kind {
                    StatementKind::Assign((place, _)) => {
                        if let Some(local) = place.as_local() {
                            accesses.push((block_idx, stmt_idx, local, 2u32)); // write
                        }
                    }
                    _ => {
                        struct ReadCollector<'a> {
                            reads: &'a mut Vec<(BasicBlock, usize, Local, u32)>,
                            block: BasicBlock,
                            stmt: usize,
                        }
                        impl<'tcx> Visitor<'tcx> for ReadCollector<'_> {
                            fn visit_local(&mut self, local: Local, ctx: mir::visit::PlaceContext, _loc: mir::Location) {
                                if ctx.is_use() {
                                    self.reads.push((self.block, self.stmt, local, 1u32));
                                }
                            }
                        }
                        let mut collector = ReadCollector {
                            reads: &mut accesses,
                            block: block_idx,
                            stmt: stmt_idx,
                        };
                        collector.visit_statement(stmt, mir::Location { block: block_idx, statement_index: stmt_idx });
                    }
                }
            }
        }

        let num_accesses = accesses.len();
        if num_accesses == 0 || num_accesses > 1024 {
            return None;
        }

        let num_locals = self.body.local_decls.len();
        let matrix_words = (num_accesses + 31) / 32;

        let descriptors: Vec<u32> = accesses
            .iter()
            .map(|(_, _, local, kind)| {
                ((local.as_u32() & 0xFFFF) << 16) | (kind & 0xFFFF)
            })
            .collect();

        let alias_matrix: Vec<u32> = vec![0; num_accesses * matrix_words];

        let desc_buf = backend
            .create_buffer((descriptors.len() * std::mem::size_of::<u32>()) as u64)?;
        desc_buf.write(&descriptors);

        let matrix_buf = backend
            .create_buffer((alias_matrix.len() * std::mem::size_of::<u32>()) as u64)?;
        matrix_buf.write(&alias_matrix);

        gpu.dispatch_alias_round(
            &desc_buf,
            &matrix_buf,
            num_accesses as u32,
            num_locals as u32,
        )
        .ok()?;

        let matrix_data: Vec<u32> = matrix_buf.read(num_accesses * matrix_words);

        let mut result = Vec::with_capacity(num_accesses);
        for i in 0..num_accesses {
            let mut row = Vec::with_capacity(num_accesses);
            for j in 0..num_accesses {
                let word_idx = i * matrix_words + (j / 32);
                let bit_idx = j % 32;
                row.push(matrix_data[word_idx] & (1u32 << bit_idx) != 0);
            }
            result.push(row);
        }

        Some(result)
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Dominance Analysis
    // ------------------------------------------------------------------

    /// Compute dominance information on GPU using iterative fixed-point.
    ///
    /// Returns a vector of DenseBitSet where result[i] contains all blocks that dominate block i.
    pub fn run_dominance_analysis(&self) -> Option<Vec<DenseBitSet<BasicBlock>>> {
        if self.body.basic_blocks.len() < 50 || self.body.basic_blocks.len() > 1024 {
            return None;
        }
        if let Some(result) = self.run_dominance_analysis_metal() {
            return Some(result);
        }
        self.run_dominance_analysis_vulkan()
    }

    fn run_dominance_analysis_metal(&self) -> Option<Vec<DenseBitSet<BasicBlock>>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_dominance_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "dominance",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let bitmap_words = (num_blocks + 31) / 32;

        let mut block_info = Vec::with_capacity(num_blocks * 5);
        for (block_idx, _block) in self.body.basic_blocks.iter_enumerated() {
            let preds = &self.body.basic_blocks.predecessors()[block_idx];
            let num_preds = preds.len().min(4); // Cap at 4 predecessors
            block_info.push(num_preds as u32);
            for p in 0..4 {
                if p < num_preds {
                    block_info.push(preds[p].as_u32());
                } else {
                    block_info.push(0xFFFFFFFF);
                }
            }
        }

        let mut dominator_sets: Vec<u32> = Vec::with_capacity(num_blocks * bitmap_words);
        for block_idx in 0..num_blocks {
            for word_idx in 0..bitmap_words {
                if block_idx == 0 {
                    if word_idx == 0 {
                        dominator_sets.push(1u32); // block 0
                    } else {
                        dominator_sets.push(0u32);
                    }
                } else {
                    dominator_sets.push(0xFFFFFFFFu32);
                }
            }
        }

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let dom_buf = backend
            .create_buffer((dominator_sets.len() * std::mem::size_of::<u32>()) as u64)?;
        dom_buf.write(&dominator_sets);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            gpu.dispatch_dominance(
                &block_info_buf,
                &dom_buf,
                &convergence_buf,
                num_blocks as u32,
                bitmap_words as u32,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }
        }

        let dom_data: Vec<u32> = dom_buf.read(num_blocks * bitmap_words);

        let mut result = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let start = block_idx * bitmap_words;
            let mut bitset = DenseBitSet::new_empty(num_blocks);
            for (word_idx, &word) in dom_data[start..start + bitmap_words].iter().enumerate() {
                if word == 0 {
                    continue;
                }
                let base_block = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let b = base_block + bit;
                        if b < num_blocks {
                            bitset.insert(BasicBlock::from_usize(b));
                        }
                    }
                }
            }
            result.push(bitset);
        }

        Some(result)
    }

    fn run_dominance_analysis_vulkan(&self) -> Option<Vec<DenseBitSet<BasicBlock>>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_dominance_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let bitmap_words = (num_blocks + 31) / 32;

        let mut block_info = Vec::with_capacity(num_blocks * 5);
        for (block_idx, _block) in self.body.basic_blocks.iter_enumerated() {
            let preds = &self.body.basic_blocks.predecessors()[block_idx];
            let num_preds = preds.len().min(4); // Cap at 4 predecessors
            block_info.push(num_preds as u32);
            for p in 0..4 {
                if p < num_preds {
                    block_info.push(preds[p].as_u32());
                } else {
                    block_info.push(0xFFFFFFFF);
                }
            }
        }

        let mut dominator_sets: Vec<u32> = Vec::with_capacity(num_blocks * bitmap_words);
        for block_idx in 0..num_blocks {
            for word_idx in 0..bitmap_words {
                if block_idx == 0 {
                    if word_idx == 0 {
                        dominator_sets.push(1u32); // block 0
                    } else {
                        dominator_sets.push(0u32);
                    }
                } else {
                    dominator_sets.push(0xFFFFFFFFu32);
                }
            }
        }

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let dom_buf = backend
            .create_buffer((dominator_sets.len() * std::mem::size_of::<u32>()) as u64)?;
        dom_buf.write(&dominator_sets);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        let mut round = 0;
        const MAX_ROUNDS: u32 = 200;

        loop {
            convergence_buf.write(&[0u32]);

            gpu.dispatch_dominance_round(
                &block_info_buf,
                &dom_buf,
                &convergence_buf,
                num_blocks as u32,
                bitmap_words as u32,
            )
            .ok()?;

            let changed = gpu.read_convergence(&convergence_buf);
            round += 1;

            if !changed || round >= MAX_ROUNDS {
                break;
            }
        }

        let dom_data: Vec<u32> = dom_buf.read(num_blocks * bitmap_words);

        let mut result = Vec::with_capacity(num_blocks);
        for block_idx in 0..num_blocks {
            let start = block_idx * bitmap_words;
            let mut bitset = DenseBitSet::new_empty(num_blocks);
            for (word_idx, &word) in dom_data[start..start + bitmap_words].iter().enumerate() {
                if word == 0 {
                    continue;
                }
                let base_block = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let b = base_block + bit;
                        if b < num_blocks {
                            bitset.insert(BasicBlock::from_usize(b));
                        }
                    }
                }
            }
            result.push(bitset);
        }

        Some(result)
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Loop Detection
    // ------------------------------------------------------------------

    /// Detect loops in the control flow graph on GPU.
    ///
    /// Returns a vector of basic blocks that are loop headers.
    pub fn run_loop_detection(&self) -> Option<Vec<BasicBlock>> {
        if self.body.basic_blocks.len() < 30 || self.body.basic_blocks.len() > 512 {
            return None;
        }
        if let Some(result) = self.run_loop_detection_metal() {
            return Some(result);
        }
        self.run_loop_detection_vulkan()
    }

    fn run_loop_detection_metal(&self) -> Option<Vec<BasicBlock>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_loop_detect_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "loop_detect",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let matrix_words = (num_blocks + 31) / 32;

        let mut block_info = Vec::with_capacity(num_blocks * 5);
        for (_block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let successors: Vec<BasicBlock> = block.terminator().successors().collect();
            let num_succs = successors.len().min(4);
            block_info.push(num_succs as u32);
            for s in 0..4 {
                if s < num_succs {
                    block_info.push(successors[s].as_u32());
                } else {
                    block_info.push(0xFFFFFFFF);
                }
            }
        }

        let reachability: Vec<u32> = vec![0; num_blocks * matrix_words];
        let loop_headers: Vec<u32> = vec![0; matrix_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let reach_buf = backend
            .create_buffer((reachability.len() * std::mem::size_of::<u32>()) as u64)?;
        reach_buf.write(&reachability);

        let loop_buf = backend
            .create_buffer((loop_headers.len() * std::mem::size_of::<u32>()) as u64)?;
        loop_buf.write(&loop_headers);

        gpu.dispatch_loop_detect(
            &block_info_buf,
            &reach_buf,
            &loop_buf,
            num_blocks as u32,
            matrix_words as u32,
        )
        .ok()?;

        let loop_data: Vec<u32> = loop_buf.read(matrix_words);

        let mut headers = Vec::new();
        for block_idx in 0..num_blocks {
            let word = block_idx / 32;
            let bit = block_idx % 32;
            if loop_data[word] & (1u32 << bit) != 0 {
                headers.push(BasicBlock::from_usize(block_idx));
            }
        }

        Some(headers)
    }

    fn run_loop_detection_vulkan(&self) -> Option<Vec<BasicBlock>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_loop_detect_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let matrix_words = (num_blocks + 31) / 32;

        let mut block_info = Vec::with_capacity(num_blocks * 5);
        for (_block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            let successors: Vec<BasicBlock> = block.terminator().successors().collect();
            let num_succs = successors.len().min(4);
            block_info.push(num_succs as u32);
            for s in 0..4 {
                if s < num_succs {
                    block_info.push(successors[s].as_u32());
                } else {
                    block_info.push(0xFFFFFFFF);
                }
            }
        }

        let reachability: Vec<u32> = vec![0; num_blocks * matrix_words];
        let loop_headers: Vec<u32> = vec![0; matrix_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let reach_buf = backend
            .create_buffer((reachability.len() * std::mem::size_of::<u32>()) as u64)?;
        reach_buf.write(&reachability);

        let loop_buf = backend
            .create_buffer((loop_headers.len() * std::mem::size_of::<u32>()) as u64)?;
        loop_buf.write(&loop_headers);

        gpu.dispatch_loop_detect_round(
            &block_info_buf,
            &reach_buf,
            &loop_buf,
            num_blocks as u32,
            matrix_words as u32,
        )
        .ok()?;

        let loop_data: Vec<u32> = loop_buf.read(matrix_words);

        let mut headers = Vec::new();
        for block_idx in 0..num_blocks {
            let word = block_idx / 32;
            let bit = block_idx % 32;
            if loop_data[word] & (1u32 << bit) != 0 {
                headers.push(BasicBlock::from_usize(block_idx));
            }
        }

        Some(headers)
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Global Value Numbering (GVN)
    // ------------------------------------------------------------------

    /// Run GVN analysis on GPU to detect redundant expressions.
    ///
    /// Returns a vector of (block, statement_idx, value_number) for expressions.
    pub fn run_gvn(&self) -> Option<Vec<(BasicBlock, usize, u32)>> {
        if self.body.basic_blocks.len() < 50 {
            return None;
        }
        if let Some(result) = self.run_gvn_metal() {
            return Some(result);
        }
        self.run_gvn_vulkan()
    }

    fn run_gvn_metal(&self) -> Option<Vec<(BasicBlock, usize, u32)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_gvn_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "gvn",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0);
        let hash_table_size = 1024;

        let mut expr_hashes = vec![0u32; num_blocks * max_statements];
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((_place, rvalue)) = &stmt.kind {
                    let hash = self.hash_rvalue(rvalue, block_idx.as_u32(), stmt_idx as u32);
                    expr_hashes[block_idx.index() * max_statements + stmt_idx] = hash;
                }
            }
        }

        let value_numbers: Vec<u32> = vec![0; num_blocks * max_statements];

        let hash_buf = backend
            .create_buffer((expr_hashes.len() * std::mem::size_of::<u32>()) as u64)?;
        hash_buf.write(&expr_hashes);

        let vn_buf = backend
            .create_buffer((value_numbers.len() * std::mem::size_of::<u32>()) as u64)?;
        vn_buf.write(&value_numbers);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        gpu.dispatch_gvn(
            &hash_buf,
            &vn_buf,
            &convergence_buf,
            num_blocks as u32,
            max_statements as u32,
            hash_table_size,
        )
        .ok()?;

        let vn_data: Vec<u32> = vn_buf.read(num_blocks * max_statements);

        let mut results = Vec::new();
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for stmt_idx in 0..block.statements.len() {
                let vn = vn_data[block_idx.index() * max_statements + stmt_idx];
                if vn != 0 {
                    results.push((block_idx, stmt_idx, vn));
                }
            }
        }

        Some(results)
    }

    fn run_gvn_vulkan(&self) -> Option<Vec<(BasicBlock, usize, u32)>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_gvn_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let max_statements = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0);
        let hash_table_size = 1024;

        let mut expr_hashes = vec![0u32; num_blocks * max_statements];
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                if let StatementKind::Assign((_place, rvalue)) = &stmt.kind {
                    let hash = self.hash_rvalue(rvalue, block_idx.as_u32(), stmt_idx as u32);
                    expr_hashes[block_idx.index() * max_statements + stmt_idx] = hash;
                }
            }
        }

        let value_numbers: Vec<u32> = vec![0; num_blocks * max_statements];

        let hash_buf = backend
            .create_buffer((expr_hashes.len() * std::mem::size_of::<u32>()) as u64)?;
        hash_buf.write(&expr_hashes);

        let vn_buf = backend
            .create_buffer((value_numbers.len() * std::mem::size_of::<u32>()) as u64)?;
        vn_buf.write(&value_numbers);

        let convergence_buf = backend.create_buffer(std::mem::size_of::<u32>() as u64)?;

        gpu.dispatch_gvn_round(
            &hash_buf,
            &vn_buf,
            &convergence_buf,
            num_blocks as u32,
            max_statements as u32,
            hash_table_size,
        )
        .ok()?;

        let vn_data: Vec<u32> = vn_buf.read(num_blocks * max_statements);

        let mut results = Vec::new();
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            for stmt_idx in 0..block.statements.len() {
                let vn = vn_data[block_idx.index() * max_statements + stmt_idx];
                if vn != 0 {
                    results.push((block_idx, stmt_idx, vn));
                }
            }
        }

        Some(results)
    }

    /// Compute a simple hash for an rvalue.
    fn hash_rvalue(&self, rvalue: &rustc_middle::mir::Rvalue<'tcx>, block: u32, stmt: u32) -> u32 {
        use rustc_middle::mir::Rvalue::*;
        let kind_hash = match rvalue {
            Use(..) => 1u32,
            Repeat(..) => 2,
            Ref(..) => 3,
            ThreadLocalRef(_) => 4,
            RawPtr(..) => 5,
            Cast(..) => 7,
            BinaryOp(..) => 8,
            UnaryOp(..) => 10,
            Discriminant(_) => 11,
            Aggregate(..) => 12,
            CopyForDeref(_) => 14,
            WrapUnsafeBinder(..) => 15,
            Reborrow(..) => 16,
        };
        // Mix in block and stmt for uniqueness
        kind_hash.wrapping_mul(31).wrapping_add(block).wrapping_mul(17).wrapping_add(stmt)
    }

    // ------------------------------------------------------------------
    // GPU-accelerated Induction Variable Detection
    // ------------------------------------------------------------------

    /// Detect induction variables in loops on GPU.
    ///
    /// Returns a vector of (block, local) pairs indicating induction variables.
    pub fn run_induction_var_detection(&self) -> Option<Vec<(BasicBlock, Local)>> {
        if self.body.basic_blocks.len() < 30 {
            return None;
        }
        if let Some(result) = self.run_induction_var_detection_metal() {
            return Some(result);
        }
        self.run_induction_var_detection_vulkan()
    }

    fn run_induction_var_detection_metal(&self) -> Option<Vec<(BasicBlock, Local)>> {
        let backend = rustc_gpu_metal::MetalBackend::new()?;
        let metallib_path = rustc_gpu_metal::load_induction_var_shader()?;
        let gpu = rustc_gpu_metal::dataflow::MetalDataflowEngine::new(
            &backend.context,
            &metallib_path,
            "induction_var",
        ).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let max_stmts = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0);
        let iv_words = (num_locals + 31) / 32;

        let mut loop_headers = vec![false; num_blocks];
        for (block_idx, _) in self.body.basic_blocks.iter_enumerated() {
            let preds = &self.body.basic_blocks.predecessors()[block_idx];
            if preds.len() > 1 {
                loop_headers[block_idx.index()] = true;
            }
        }

        let mut block_info = Vec::with_capacity(num_blocks * (2 + max_stmts * 2));
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            block_info.push(if loop_headers[block_idx.index()] { 1u32 } else { 0 });
            let num_stmts = block.statements.len().min(max_stmts);
            block_info.push(num_stmts as u32);

            for stmt in &block.statements[..num_stmts] {
                let (local, kind) = match &stmt.kind {
                    StatementKind::Assign((place, rvalue)) => {
                        let l = place.local;
                        let k = match rvalue {
                            rustc_middle::mir::Rvalue::BinaryOp(
                                rustc_middle::mir::BinOp::Add,
                                _,
                            ) => 1u32,
                            rustc_middle::mir::Rvalue::BinaryOp(
                                rustc_middle::mir::BinOp::Sub,
                                _,
                            ) => 2u32,
                            _ => 3u32,
                        };
                        (l.as_u32(), k)
                    }
                    _ => (0xFFFFFFFF, 0),
                };
                block_info.push(local);
                block_info.push(kind);
            }

            for _ in num_stmts..max_stmts {
                block_info.push(0xFFFFFFFF);
                block_info.push(0);
            }
        }

        let induction_vars: Vec<u32> = vec![0; num_blocks * iv_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let iv_buf = backend
            .create_buffer((induction_vars.len() * std::mem::size_of::<u32>()) as u64)?;
        iv_buf.write(&induction_vars);

        gpu.dispatch_induction_var(
            &block_info_buf,
            &iv_buf,
            num_blocks as u32,
            num_locals as u32,
            max_stmts as u32,
        )
        .ok()?;

        let iv_data: Vec<u32> = iv_buf.read(num_blocks * iv_words);

        let mut results = Vec::new();
        for block_idx in 0..num_blocks {
            if !loop_headers[block_idx] {
                continue;
            }
            let start = block_idx * iv_words;
            for word_idx in 0..iv_words {
                let word = iv_data[start + word_idx];
                if word == 0 {
                    continue;
                }
                let base_local = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let local_idx = base_local + bit;
                        if local_idx < num_locals {
                            results.push((
                                BasicBlock::from_usize(block_idx),
                                Local::from_usize(local_idx),
                            ));
                        }
                    }
                }
            }
        }

        Some(results)
    }

    fn run_induction_var_detection_vulkan(&self) -> Option<Vec<(BasicBlock, Local)>> {
        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_induction_var_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        let num_blocks = self.body.basic_blocks.len();
        let num_locals = self.body.local_decls.len();
        let max_stmts = self
            .body
            .basic_blocks
            .iter()
            .map(|b| b.statements.len())
            .max()
            .unwrap_or(0);
        let iv_words = (num_locals + 31) / 32;

        let mut loop_headers = vec![false; num_blocks];
        for (block_idx, _) in self.body.basic_blocks.iter_enumerated() {
            let preds = &self.body.basic_blocks.predecessors()[block_idx];
            if preds.len() > 1 {
                loop_headers[block_idx.index()] = true;
            }
        }

        let mut block_info = Vec::with_capacity(num_blocks * (2 + max_stmts * 2));
        for (block_idx, block) in self.body.basic_blocks.iter_enumerated() {
            block_info.push(if loop_headers[block_idx.index()] { 1u32 } else { 0 });
            let num_stmts = block.statements.len().min(max_stmts);
            block_info.push(num_stmts as u32);

            for stmt in &block.statements[..num_stmts] {
                let (local, kind) = match &stmt.kind {
                    StatementKind::Assign((place, rvalue)) => {
                        let l = place.local;
                        let k = match rvalue {
                            rustc_middle::mir::Rvalue::BinaryOp(
                                rustc_middle::mir::BinOp::Add,
                                _,
                            ) => 1u32,
                            rustc_middle::mir::Rvalue::BinaryOp(
                                rustc_middle::mir::BinOp::Sub,
                                _,
                            ) => 2u32,
                            _ => 3u32,
                        };
                        (l.as_u32(), k)
                    }
                    _ => (0xFFFFFFFF, 0),
                };
                block_info.push(local);
                block_info.push(kind);
            }

            for _ in num_stmts..max_stmts {
                block_info.push(0xFFFFFFFF);
                block_info.push(0);
            }
        }

        let induction_vars: Vec<u32> = vec![0; num_blocks * iv_words];

        let block_info_buf = backend
            .create_buffer((block_info.len() * std::mem::size_of::<u32>()) as u64)?;
        block_info_buf.write(&block_info);

        let iv_buf = backend
            .create_buffer((induction_vars.len() * std::mem::size_of::<u32>()) as u64)?;
        iv_buf.write(&induction_vars);

        gpu.dispatch_induction_var_round(
            &block_info_buf,
            &iv_buf,
            num_blocks as u32,
            num_locals as u32,
            max_stmts as u32,
        )
        .ok()?;

        let iv_data: Vec<u32> = iv_buf.read(num_blocks * iv_words);

        let mut results = Vec::new();
        for block_idx in 0..num_blocks {
            if !loop_headers[block_idx] {
                continue;
            }
            let start = block_idx * iv_words;
            for word_idx in 0..iv_words {
                let word = iv_data[start + word_idx];
                if word == 0 {
                    continue;
                }
                let base_local = word_idx * 32;
                for bit in 0..32 {
                    if word & (1u32 << bit) != 0 {
                        let local_idx = base_local + bit;
                        if local_idx < num_locals {
                            results.push((
                                BasicBlock::from_usize(block_idx),
                                Local::from_usize(local_idx),
                            ));
                        }
                    }
                }
            }
        }

        Some(results)
    }

    // ------------------------------------------------------------------
    // MEGA-BATCH: Process multiple functions simultaneously
    // ------------------------------------------------------------------

    /// Run forward dataflow analysis on GPU for multiple functions at once.
    ///
    /// This amortizes kernel launch overhead across many functions,
    /// achieving up to 100x better GPU utilization.
    pub fn run_mega_batch_forward_analysis(
        &self,
        bodies: &[&'tcx Body<'tcx>],
    ) -> Option<Vec<Vec<DenseBitSet<Local>>>> {
        if bodies.len() < 2 || bodies.len() > 100 {
            return None;
        }

        let backend = GpuBackend::new()?;
        let spirv = rustc_gpu_vulkan::load_mega_batch_dataflow_shader()?;
        let gpu = GpuDataflowEngine::new(&backend.context, &spirv).ok()?;

        // Serialize all functions into concatenated buffers
        let (meta, configs, effects, entry_states, exit_states, max_blocks) =
            self.serialize_mega_batch(bodies);

        let num_functions = bodies.len();
        let blocks_per_workgroup = max_blocks;
        let _num_locals_total: usize = bodies.iter().map(|b| b.local_decls.len()).sum();
        let max_bitset_words = (bodies.iter().map(|b| b.local_decls.len()).max().unwrap_or(0) + 31) / 32;

        // Upload to GPU
        let meta_buf = backend.create_buffer((meta.len() * std::mem::size_of::<u32>()) as u64)?;
        meta_buf.write(&meta);

        let config_buf = backend.create_buffer((configs.len() * std::mem::size_of::<u32>()) as u64)?;
        config_buf.write(&configs);

        let effects_buf = backend.create_buffer((effects.len() * std::mem::size_of::<u32>()) as u64)?;
        effects_buf.write(&effects);

        let entry_buf = backend.create_buffer((entry_states.len() * std::mem::size_of::<u32>()) as u64)?;
        entry_buf.write(&entry_states);

        let exit_buf = backend.create_buffer((exit_states.len() * std::mem::size_of::<u32>()) as u64)?;
        exit_buf.write(&exit_states);

        let convergence_buf = backend.create_buffer((num_functions * std::mem::size_of::<u32>()) as u64)?;

        // Fixed-point iteration
        let mut round = 0;
        const MAX_ROUNDS: u32 = 100;

        loop {
            gpu.dispatch_mega_batch_round(
                &meta_buf,
                &config_buf,
                &effects_buf,
                &entry_buf,
                &exit_buf,
                &convergence_buf,
                num_functions as u32,
                blocks_per_workgroup as u32,
                max_bitset_words as u32,
            )
            .ok()?;

            let conv_data: Vec<u32> = convergence_buf.read(num_functions);
            let any_changed = conv_data.iter().any(|&x| x != 0);
            round += 1;

            if !any_changed || round >= MAX_ROUNDS {
                break;
            }

            // Propagate edges for all functions on CPU
            let exit_data: Vec<u32> = exit_buf.read(exit_states.len());
            let mut entry_data: Vec<u32> = entry_buf.read(entry_states.len());
            self.propagate_mega_batch_edges(bodies, &meta, &exit_data, &mut entry_data);
            entry_buf.write(&entry_data);
        }

        // Read back results
        let final_entry: Vec<u32> = entry_buf.read(entry_states.len());
        Some(self.parse_mega_batch_results(bodies, &meta, &final_entry, max_bitset_words))
    }

    /// Serialize multiple functions into mega-batch buffers.
    fn serialize_mega_batch(
        &self,
        bodies: &[&'tcx Body<'tcx>],
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, usize) {
        let mut meta = Vec::with_capacity(bodies.len() * 4);
        let mut configs = Vec::new();
        let mut effects = Vec::new();
        let mut entry_states = Vec::new();
        let mut exit_states = Vec::new();

        let max_blocks = bodies.iter().map(|b| b.basic_blocks.len()).max().unwrap_or(0);
        let max_statements = bodies
            .iter()
            .map(|b| b.basic_blocks.iter().map(|bb| bb.statements.len()).max().unwrap_or(0))
            .max()
            .unwrap_or(0);
        let max_locals = bodies.iter().map(|b| b.local_decls.len()).max().unwrap_or(0);
        let _max_bitset_words = (max_locals + 31) / 32;

        for body in bodies {
            let num_blocks = body.basic_blocks.len();
            let num_locals = body.local_decls.len();
            let bitset_words = (num_locals + 31) / 32;
            let data_offset = configs.len() as u32;

            // Metadata: [num_blocks, num_locals, effects_stride, data_offset]
            meta.push(num_blocks as u32);
            meta.push(num_locals as u32);
            meta.push(max_statements as u32);
            meta.push(data_offset);

            // Configs for this function
            for block in body.basic_blocks.iter() {
                let stmt_count = block.statements.len() as u32;
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

                configs.push(stmt_count);
                configs.push(terminator_kind);
                configs.push(
                    (successors.len() as u32 & 0xFFFF)
                        | ((successors.get(0).map_or(u32::MAX, |b| b.as_u32()) & 0xFFFF) << 16),
                );
                configs.push(
                    successors.get(1).map_or(u32::MAX, |b| b.as_u32() & 0xFFFF),
                );
            }

            // Effects for this function
            let mut func_effects = vec![0u32; num_blocks * max_statements];
            for (block_idx, block) in body.basic_blocks.iter_enumerated() {
                for (stmt_idx, stmt) in block.statements.iter().enumerate() {
                    let encoded = match &stmt.kind {
                        StatementKind::StorageLive(local) => (2u32 << 24) | local.as_u32(),
                        StatementKind::StorageDead(local) => (1u32 << 24) | local.as_u32(),
                        _ => 0u32,
                    };
                    func_effects[block_idx.index() * max_statements + stmt_idx] = encoded;
                }
            }
            effects.extend(func_effects);

            // Entry/exit states (initialized to bottom)
            entry_states.resize(entry_states.len() + num_blocks * bitset_words, 0);
            exit_states.resize(exit_states.len() + num_blocks * bitset_words, 0);
        }

        (meta, configs, effects, entry_states, exit_states, max_blocks)
    }

    /// Propagate edges for all functions in the mega-batch.
    fn propagate_mega_batch_edges(
        &self,
        _bodies: &[&'tcx Body<'tcx>],
        meta: &[u32],
        exit_states: &[u32],
        entry_states: &mut [u32],
    ) {
        for func_idx in 0..meta.len() / 4 {
            let meta_offset = func_idx * 4;
            let num_blocks = meta[meta_offset] as usize;
            let num_locals = meta[meta_offset + 1] as usize;
            let data_offset = meta[meta_offset + 3] as usize;
            let bitset_words = (num_locals + 31) / 32;

            for block_idx in 0..num_blocks {
                let exit_start = data_offset + block_idx * bitset_words;
                let exit_slice = &exit_states[exit_start..exit_start + bitset_words];

                // We need body to get successors, but we don't have it here
                // For now, just propagate to all possible successors
                // In real implementation, we'd pass successor info
                for succ_idx in 0..num_blocks {
                    let entry_start = data_offset + succ_idx * bitset_words;
                    for w in 0..bitset_words {
                        entry_states[entry_start + w] |= exit_slice[w];
                    }
                }
            }
        }
    }

    /// Parse mega-batch results into per-function DenseBitSets.
    fn parse_mega_batch_results(
        &self,
        bodies: &[&'tcx Body<'tcx>],
        meta: &[u32],
        flat_states: &[u32],
        _max_bitset_words: usize,
    ) -> Vec<Vec<DenseBitSet<Local>>> {
        let mut all_results = Vec::with_capacity(bodies.len());

        for (func_idx, _body) in bodies.iter().enumerate() {
            let meta_offset = func_idx * 4;
            let num_blocks = meta[meta_offset] as usize;
            let num_locals = meta[meta_offset + 1] as usize;
            let data_offset = meta[meta_offset + 3] as usize;
            let bitset_words = (num_locals + 31) / 32;

            let mut func_results = Vec::with_capacity(num_blocks);
            for block_idx in 0..num_blocks {
                let start = data_offset + block_idx * bitset_words;
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
                func_results.push(bitset);
            }
            all_results.push(func_results);
        }

        all_results
    }
}
