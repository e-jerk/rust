use metal::ComputePipelineState;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use std::os::raw::c_void;
use std::sync::Arc;

/// Metal dataflow engine for fixed-point iteration.
///
/// Mirrors `GpuDataflowEngine` from `rustc_gpu_vulkan` but uses native Metal APIs.
/// Key simplifications vs Vulkan:
/// - No descriptor pools/sets (buffers bound by index)
/// - No memory barriers (command buffer boundaries handle sync)
/// - No fences (wait_until_completed is the primitive)
/// - No shader module creation at runtime (shaders pre-compiled to .metallib)
pub struct MetalDataflowEngine {
    _device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: Arc<ComputePipelineState>,
}

impl MetalDataflowEngine {
    pub fn new(
        context: &MetalContext,
        metallib_path: &str,
        function_name: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let library = context.device.new_library_with_file(metallib_path)
            .map_err(|e| format!("Failed to load metallib: {}", e))?;
        
        let function = library.get_function(function_name, None)
            .map_err(|e| format!("Failed to get function: {}", e))?;
        
        let pipeline = context.device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline creation failed: {:?}", e))?;
        
        Ok(MetalDataflowEngine {
            _device: context.device.clone(),
            queue: context.queue.clone(),
            pipeline: Arc::new(pipeline),
        })
    }
    
    /// Create from a cached pipeline (avoids recompiling the shader).
    pub fn from_pipeline(
        context: &MetalContext,
        pipeline: Arc<metal::ComputePipelineState>,
    ) -> Self {
        MetalDataflowEngine {
            _device: context.device.clone(),
            queue: context.queue.clone(),
            pipeline,
        }
    }
    
    /// Dispatch a batch of dataflow rounds in a single command buffer.
    ///
    /// Each tuple is (config, effects, entry, exit, convergence, num_blocks, bitset_words, effects_stride).
    /// All dispatches share the same pipeline and are encoded back-to-back.
    /// Amortizes command buffer creation + commit + wait overhead.
    pub fn dispatch_round_batch(
        &self,
        dispatches: &[(
            &MetalBuffer, &MetalBuffer, &MetalBuffer, &MetalBuffer, &MetalBuffer,
            u32, u32, u32,
        )],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        for (config_buf, effects_buf, entry_buf, exit_buf, convergence_buf,
             num_blocks, bitset_words, effects_stride) in dispatches {
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pipeline);
            encoder.set_buffer(0, Some(&config_buf.buffer), 0);
            encoder.set_buffer(1, Some(&effects_buf.buffer), 0);
            encoder.set_buffer(2, Some(&entry_buf.buffer), 0);
            encoder.set_buffer(3, Some(&exit_buf.buffer), 0);
            encoder.set_buffer(4, Some(&convergence_buf.buffer), 0);
            let push_constants = [*num_blocks, *bitset_words, *effects_stride];
            encoder.set_bytes(
                5,
                std::mem::size_of_val(&push_constants) as u64,
                &push_constants as *const _ as *const c_void,
            );
            let threadgroup_size = metal::MTLSize::new(128, 1, 1);
            if *num_blocks % 128 == 0 {
                let threadgroups = metal::MTLSize::new((*num_blocks / 128) as u64, 1, 1);
                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            } else {
                let grid_size = metal::MTLSize::new(*num_blocks as u64, 1, 1);
                encoder.dispatch_threads(grid_size, threadgroup_size);
            }
            encoder.end_encoding();
        }
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        Ok(())
    }

    /// Dispatch one dataflow round.
    pub fn dispatch_round(
        &self,
        config_buf: &MetalBuffer,
        effects_buf: &MetalBuffer,
        entry_buf: &MetalBuffer,
        exit_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_blocks: u32,
        bitset_words: u32,
        effects_stride: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&config_buf.buffer), 0);
        encoder.set_buffer(1, Some(&effects_buf.buffer), 0);
        encoder.set_buffer(2, Some(&entry_buf.buffer), 0);
        encoder.set_buffer(3, Some(&exit_buf.buffer), 0);
        encoder.set_buffer(4, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_blocks, bitset_words, effects_stride];
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        if num_blocks % 256 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch fused MIR optimization round (4 analyses in 1 dispatch).
    pub fn dispatch_fused_mir_opt(
        &self,
        config_buf: &MetalBuffer,
        effects_buf: &MetalBuffer,
        entry_buf: &MetalBuffer,
        exit_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_blocks: u32,
        num_locals: u32,
        bitset_words: u32,
        effects_stride: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&config_buf.buffer), 0);
        encoder.set_buffer(1, Some(&effects_buf.buffer), 0);
        encoder.set_buffer(2, Some(&entry_buf.buffer), 0);
        encoder.set_buffer(3, Some(&exit_buf.buffer), 0);
        encoder.set_buffer(4, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_blocks, num_locals, bitset_words, effects_stride];
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        // Use dispatchThreadgroups when grid aligns to threadgroup for lower driver overhead
        let threadgroup_size = metal::MTLSize::new(128, 1, 1);
        if num_blocks % 128 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 128) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch fused MIR optimization batch: N dispatches in 1 command buffer.
    ///
    /// Amortizes command buffer creation + commit + wait overhead across N dispatches.
    /// Useful for independent work items (e.g., monomorphization batches, multiple
    /// functions in a crate) where CPU synchronization is not needed between dispatches.
    pub fn dispatch_fused_mir_opt_batch(
        &self,
        dispatches: &[(
            &MetalBuffer, // config
            &MetalBuffer, // effects
            &MetalBuffer, // entry
            &MetalBuffer, // exit
            &MetalBuffer, // convergence
            u32, // num_blocks
            u32, // num_locals
            u32, // bitset_words
            u32, // effects_stride
        )],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        
        for (config_buf, effects_buf, entry_buf, exit_buf, convergence_buf,
             num_blocks, num_locals, bitset_words, effects_stride) in dispatches {
            let encoder = cmd_buf.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(&self.pipeline);
            
            encoder.set_buffer(0, Some(&config_buf.buffer), 0);
            encoder.set_buffer(1, Some(&effects_buf.buffer), 0);
            encoder.set_buffer(2, Some(&entry_buf.buffer), 0);
            encoder.set_buffer(3, Some(&exit_buf.buffer), 0);
            encoder.set_buffer(4, Some(&convergence_buf.buffer), 0);
            
            let push_constants = [*num_blocks, *num_locals, *bitset_words, *effects_stride];
            encoder.set_bytes(
                5,
                std::mem::size_of_val(&push_constants) as u64,
                &push_constants as *const _ as *const c_void,
            );
            
            let threadgroup_size = metal::MTLSize::new(128, 1, 1);
            if *num_blocks % 128 == 0 {
                let threadgroups = metal::MTLSize::new((*num_blocks / 128) as u64, 1, 1);
                encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
            } else {
                let grid_size = metal::MTLSize::new(*num_blocks as u64, 1, 1);
                encoder.dispatch_threads(grid_size, threadgroup_size);
            }
            
            encoder.end_encoding();
        }
        
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch alias analysis.
    pub fn dispatch_alias(
        &self,
        desc_buf: &MetalBuffer,
        matrix_buf: &MetalBuffer,
        num_accesses: u32,
        num_locals: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&desc_buf.buffer), 0);
        encoder.set_buffer(1, Some(&matrix_buf.buffer), 0);
        
        let push_constants = [num_accesses, num_locals];
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_accesses % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_accesses / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_accesses as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch dominance analysis.
    pub fn dispatch_dominance(
        &self,
        block_info_buf: &MetalBuffer,
        dom_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_blocks: u32,
        block_bitmap_words: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&block_info_buf.buffer), 0);
        encoder.set_buffer(1, Some(&dom_buf.buffer), 0);
        encoder.set_buffer(2, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_blocks, block_bitmap_words];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_blocks % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch loop detection.
    pub fn dispatch_loop_detect(
        &self,
        block_info_buf: &MetalBuffer,
        reach_buf: &MetalBuffer,
        loop_buf: &MetalBuffer,
        num_blocks: u32,
        matrix_words: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&block_info_buf.buffer), 0);
        encoder.set_buffer(1, Some(&reach_buf.buffer), 0);
        encoder.set_buffer(2, Some(&loop_buf.buffer), 0);
        
        let push_constants = [num_blocks, matrix_words];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_blocks % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch GVN (global value numbering).
    pub fn dispatch_gvn(
        &self,
        hash_buf: &MetalBuffer,
        vn_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_blocks: u32,
        statements_per_block: u32,
        hash_table_size: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&hash_buf.buffer), 0);
        encoder.set_buffer(1, Some(&vn_buf.buffer), 0);
        encoder.set_buffer(2, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_blocks, statements_per_block, hash_table_size];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_blocks % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch induction variable detection.
    pub fn dispatch_induction_var(
        &self,
        block_info_buf: &MetalBuffer,
        iv_buf: &MetalBuffer,
        num_blocks: u32,
        num_locals: u32,
        max_stmts_per_block: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&block_info_buf.buffer), 0);
        encoder.set_buffer(1, Some(&iv_buf.buffer), 0);
        
        let push_constants = [num_blocks, num_locals, max_stmts_per_block];
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_blocks % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch mega-batch dataflow.
    pub fn dispatch_mega_batch(
        &self,
        meta_buf: &MetalBuffer,
        config_buf: &MetalBuffer,
        effects_buf: &MetalBuffer,
        entry_buf: &MetalBuffer,
        exit_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_functions: u32,
        blocks_per_workgroup: u32,
        max_bitset_words: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&meta_buf.buffer), 0);
        encoder.set_buffer(1, Some(&config_buf.buffer), 0);
        encoder.set_buffer(2, Some(&effects_buf.buffer), 0);
        encoder.set_buffer(3, Some(&entry_buf.buffer), 0);
        encoder.set_buffer(4, Some(&exit_buf.buffer), 0);
        encoder.set_buffer(5, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_functions, blocks_per_workgroup, max_bitset_words];
        encoder.set_bytes(
            6,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let total_blocks = num_functions * blocks_per_workgroup;
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        if total_blocks % 256 == 0 {
            let threadgroups = metal::MTLSize::new((total_blocks / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(total_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch graph partitioning.
    pub fn dispatch_partition(
        &self,
        edge_list_buf: &MetalBuffer,
        edge_offsets_buf: &MetalBuffer,
        labels_buf: &MetalBuffer,
        label_counts_buf: &MetalBuffer,
        convergence_buf: &MetalBuffer,
        num_nodes: u32,
        max_label: u32,
        max_size: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&edge_list_buf.buffer), 0);
        encoder.set_buffer(1, Some(&edge_offsets_buf.buffer), 0);
        encoder.set_buffer(2, Some(&labels_buf.buffer), 0);
        encoder.set_buffer(3, Some(&label_counts_buf.buffer), 0);
        encoder.set_buffer(4, Some(&convergence_buf.buffer), 0);
        
        let push_constants = [num_nodes, max_label, max_size];
        encoder.set_bytes(
            5,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(256, 1, 1);
        if num_nodes % 256 == 0 {
            let threadgroups = metal::MTLSize::new((num_nodes / 256) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_nodes as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Dispatch SSA construction (phi-node insertion).
    pub fn dispatch_ssa(
        &self,
        block_info_buf: &MetalBuffer,
        def_sites_buf: &MetalBuffer,
        dom_frontier_buf: &MetalBuffer,
        phi_nodes_buf: &MetalBuffer,
        num_blocks: u32,
        num_locals: u32,
        max_defs_per_block: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&block_info_buf.buffer), 0);
        encoder.set_buffer(1, Some(&def_sites_buf.buffer), 0);
        encoder.set_buffer(2, Some(&dom_frontier_buf.buffer), 0);
        encoder.set_buffer(3, Some(&phi_nodes_buf.buffer), 0);
        
        let push_constants = [num_blocks, num_locals, max_defs_per_block];
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&push_constants) as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        if num_blocks % 64 == 0 {
            let threadgroups = metal::MTLSize::new((num_blocks / 64) as u64, 1, 1);
            encoder.dispatch_thread_groups(threadgroups, threadgroup_size);
        } else {
            let grid_size = metal::MTLSize::new(num_blocks as u64, 1, 1);
            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
    
    /// Read back the convergence flag.
    pub fn read_convergence(&self, convergence_buf: &MetalBuffer) -> bool {
        let data: Vec<u32> = convergence_buf.read(1);
        data[0] != 0
    }
}
