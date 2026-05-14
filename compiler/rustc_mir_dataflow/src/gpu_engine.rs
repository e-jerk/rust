use rustc_gpu_vulkan::dataflow::{GpuDataflowConfig, GpuDataflowEngine};
use rustc_gpu_vulkan::{GpuBackend, load_dataflow_shader};
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{BasicBlock, Body, Local, StatementKind, TerminatorKind};
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
}
