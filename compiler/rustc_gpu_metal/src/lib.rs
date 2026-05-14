pub mod buffer;
pub mod context;
pub mod dataflow;
pub mod dispatch;

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

thread_local! {
    static METAL_BACKEND: RefCell<Option<MetalBackend>> = RefCell::new(None);
}

/// Feature-gated Metal GPU backend. Returns None if Metal unavailable.
/// Uses thread-local caching to avoid recreating the device/context per call.
/// Also caches compute pipelines to avoid repeated compilation.
pub struct MetalBackend {
    pub context: Arc<context::MetalContext>,
    pipeline_cache: RefCell<HashMap<(String, String), Arc<metal::ComputePipelineState>>>,
}

impl Clone for MetalBackend {
    fn clone(&self) -> Self {
        MetalBackend {
            context: Arc::clone(&self.context),
            pipeline_cache: RefCell::new(HashMap::new()),
        }
    }
}

impl MetalBackend {
    pub fn new() -> Option<Self> {
        METAL_BACKEND.with(|b| {
            if b.borrow().is_none() {
                *b.borrow_mut() = Self::create();
            }
            b.borrow().clone()
        })
    }

    fn create() -> Option<Self> {
        let context = context::MetalContext::new().ok()?;
        Some(MetalBackend {
            context: Arc::new(context),
            pipeline_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn create_buffer(&self, size: u64) -> Option<buffer::MetalBuffer> {
        buffer::MetalBuffer::new(&self.context.device, size)
    }

    /// Get or create a cached compute pipeline for the given shader.
    pub fn get_pipeline(
        &self,
        metallib_path: &str,
        function_name: &str,
    ) -> Option<Arc<metal::ComputePipelineState>> {
        let key = (metallib_path.to_string(), function_name.to_string());
        {
            let cache = self.pipeline_cache.borrow();
            if let Some(pipeline) = cache.get(&key) {
                return Some(Arc::clone(pipeline));
            }
        }
        let library = self.context.device.new_library_with_file(metallib_path).ok()?;
        let function = library.get_function(function_name, None).ok()?;
        let pipeline = self.context.device
            .new_compute_pipeline_state_with_function(&function)
            .ok()?;
        let pipeline = Arc::new(pipeline);
        self.pipeline_cache.borrow_mut().insert(key, Arc::clone(&pipeline));
        Some(pipeline)
    }
}

// Shader loader functions — each reads the path from the env var set by build.rs

pub fn load_mono_collect_shader() -> Option<String> {
    std::env::var("MONO_COLLECT_METALLIB").ok()
}

pub fn load_dataflow_shader() -> Option<String> {
    std::env::var("DATAFLOW_METALLIB").ok()
}

pub fn load_dead_store_elim_shader() -> Option<String> {
    std::env::var("DEAD_STORE_ELIM_METALLIB").ok()
}

pub fn load_copy_prop_shader() -> Option<String> {
    std::env::var("COPY_PROP_METALLIB").ok()
}

pub fn load_const_prop_shader() -> Option<String> {
    std::env::var("CONST_PROP_METALLIB").ok()
}

pub fn load_reaching_defs_shader() -> Option<String> {
    std::env::var("REACHING_DEFS_METALLIB").ok()
}

pub fn load_ssa_construct_shader() -> Option<String> {
    std::env::var("SSA_CONSTRUCT_METALLIB").ok()
}

pub fn load_alias_analysis_shader() -> Option<String> {
    std::env::var("ALIAS_ANALYSIS_METALLIB").ok()
}

pub fn load_dominance_shader() -> Option<String> {
    std::env::var("DOMINANCE_METALLIB").ok()
}

pub fn load_loop_detect_shader() -> Option<String> {
    std::env::var("LOOP_DETECT_METALLIB").ok()
}

pub fn load_gvn_shader() -> Option<String> {
    std::env::var("GVN_METALLIB").ok()
}

pub fn load_induction_var_shader() -> Option<String> {
    std::env::var("INDUCTION_VAR_METALLIB").ok()
}

pub fn load_mega_batch_dataflow_shader() -> Option<String> {
    std::env::var("MEGA_BATCH_DATAFLOW_METALLIB").ok()
}

pub fn load_macro_expand_shader() -> Option<String> {
    std::env::var("MACRO_EXPAND_METALLIB").ok()
}

pub fn load_borrow_check_shader() -> Option<String> {
    std::env::var("BORROW_CHECK_METALLIB").ok()
}

pub fn load_partition_shader() -> Option<String> {
    std::env::var("PARTITION_METALLIB").ok()
}

/// Load the fused MIR optimization shader (4 analyses in 1 dispatch).
pub fn load_fused_mir_opt_shader() -> Option<String> {
    std::env::var("FUSED_MIR_OPT_METALLIB").ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_buffer_size_calculation_u32() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5];
        let expected = data.len() * std::mem::size_of::<u32>();
        assert_eq!(expected, 20);
    }

    #[test]
    fn test_buffer_size_calculation_u8() {
        let data: Vec<u8> = vec![0; 64];
        let expected = data.len() * std::mem::size_of::<u8>();
        assert_eq!(expected, 64);
    }

    #[test]
    fn test_buffer_size_calculation_empty() {
        let data: Vec<u32> = vec![];
        let expected = data.len() * std::mem::size_of::<u32>();
        assert_eq!(expected, 0);
    }
}
