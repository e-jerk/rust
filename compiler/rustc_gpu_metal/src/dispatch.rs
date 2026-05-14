use metal::ComputePipelineState;
use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use std::os::raw::c_void;

/// Metal dispatch engine for monomorphization and collection.
///
/// Mirrors `GpuDispatch` from `rustc_gpu_vulkan` but uses native Metal APIs.
/// No persistent descriptor pools needed — Metal binds buffers directly by index.
pub struct MetalDispatch {
    _device: metal::Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
}

impl MetalDispatch {
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
        
        Ok(MetalDispatch {
            _device: context.device.clone(),
            queue: context.queue.clone(),
            pipeline,
        })
    }
    
    pub fn dispatch(
        &self,
        actions_buf: &MetalBuffer,
        offsets_buf: &MetalBuffer,
        edges_buf: &MetalBuffer,
        num_bodies: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.dispatch_with_counter(actions_buf, offsets_buf, edges_buf, None, num_bodies)
    }
    
    pub fn dispatch_with_counter(
        &self,
        actions_buf: &MetalBuffer,
        offsets_buf: &MetalBuffer,
        edges_buf: &MetalBuffer,
        counter_buf: Option<&MetalBuffer>,
        num_bodies: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cmd_buf = self.queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.pipeline);
        
        encoder.set_buffer(0, Some(&actions_buf.buffer), 0);
        encoder.set_buffer(1, Some(&offsets_buf.buffer), 0);
        encoder.set_buffer(2, Some(&edges_buf.buffer), 0);
        
        if let Some(counter) = counter_buf {
            encoder.set_buffer(3, Some(&counter.buffer), 0);
        }
        
        let push_constants = num_bodies;
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &push_constants as *const _ as *const c_void,
        );
        
        let grid_size = metal::MTLSize::new(num_bodies as u64, 1, 1);
        let threadgroup_size = metal::MTLSize::new(64, 1, 1);
        encoder.dispatch_threads(grid_size, threadgroup_size);
        
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        
        Ok(())
    }
}
