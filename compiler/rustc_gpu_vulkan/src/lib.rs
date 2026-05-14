#![allow(internal_features)]

pub mod buffer;
pub mod context;
pub mod dataflow;
pub mod dispatch;
pub mod shader;

use gpu_alloc as _;
use gpu_alloc_ash as _;
use tracing as _;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

thread_local! {
    static GPU_BACKEND: RefCell<Option<GpuBackend>> = RefCell::new(None);
    static SPV_CACHE: RefCell<HashMap<String, Vec<u8>>> = RefCell::new(HashMap::new());
}

/// Feature-gated GPU backend. Returns None if Vulkan unavailable.
/// Uses thread-local caching to avoid recreating the instance/device per call.
pub struct GpuBackend {
    pub context: Arc<context::GpuContext>,
}

impl Clone for GpuBackend {
    fn clone(&self) -> Self {
        GpuBackend {
            context: Arc::clone(&self.context),
        }
    }
}

impl GpuBackend {
    pub fn new() -> Option<Self> {
        GPU_BACKEND.with(|b| {
            if b.borrow().is_none() {
                *b.borrow_mut() = Self::create();
            }
            b.borrow().clone()
        })
    }

    fn create() -> Option<Self> {
        let context = context::GpuContext::new().ok()?;
        Some(GpuBackend { context: Arc::new(context) })
    }

    pub fn create_buffer(&self, size: u64) -> Option<buffer::GpuBuffer> {
        buffer::GpuBuffer::new_host_visible(
            &self.context.device,
            self.context.physical_device,
            &self.context.instance,
            size,
        )
        .ok()
    }
}

fn load_spv_cached(env_var: &str) -> Option<Vec<u8>> {
    SPV_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(data) = cache.get(env_var) {
            return Some(data.clone());
        }
        let data = std::env::var(env_var).ok().and_then(|p| std::fs::read(p).ok())?;
        cache.insert(env_var.to_string(), data.clone());
        Some(data)
    })
}

pub fn load_mono_collect_shader() -> Option<Vec<u8>> {
    load_spv_cached("MONO_COLLECT_SPV")
}

pub fn load_dataflow_shader() -> Option<Vec<u8>> {
    load_spv_cached("DATAFLOW_SPV")
}

pub fn load_dead_store_elim_shader() -> Option<Vec<u8>> {
    load_spv_cached("DEAD_STORE_ELIM_SPV")
}

pub fn load_copy_prop_shader() -> Option<Vec<u8>> {
    load_spv_cached("COPY_PROP_SPV")
}

pub fn load_const_prop_shader() -> Option<Vec<u8>> {
    load_spv_cached("CONST_PROP_SPV")
}

pub fn load_reaching_defs_shader() -> Option<Vec<u8>> {
    load_spv_cached("REACHING_DEFS_SPV")
}

pub fn load_ssa_construct_shader() -> Option<Vec<u8>> {
    load_spv_cached("SSA_CONSTRUCT_SPV")
}

pub fn load_alias_analysis_shader() -> Option<Vec<u8>> {
    load_spv_cached("ALIAS_ANALYSIS_SPV")
}

pub fn load_dominance_shader() -> Option<Vec<u8>> {
    load_spv_cached("DOMINANCE_SPV")
}

pub fn load_loop_detect_shader() -> Option<Vec<u8>> {
    load_spv_cached("LOOP_DETECT_SPV")
}

pub fn load_gvn_shader() -> Option<Vec<u8>> {
    load_spv_cached("GVN_SPV")
}

pub fn load_induction_var_shader() -> Option<Vec<u8>> {
    load_spv_cached("INDUCTION_VAR_SPV")
}

pub fn load_mega_batch_dataflow_shader() -> Option<Vec<u8>> {
    load_spv_cached("MEGA_BATCH_DATAFLOW_SPV")
}

pub fn load_macro_expand_shader() -> Option<Vec<u8>> {
    load_spv_cached("MACRO_EXPAND_SPV")
}

pub fn load_partition_shader() -> Option<Vec<u8>> {
    load_spv_cached("PARTITION_SPV")
}

pub fn load_fused_mir_opt_shader() -> Option<Vec<u8>> {
    load_spv_cached("FUSED_MIR_OPT_SPV")
}
}

pub fn load_dataflow_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("DATAFLOW_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_dead_store_elim_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("DEAD_STORE_ELIM_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_copy_prop_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("COPY_PROP_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_const_prop_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("CONST_PROP_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_reaching_defs_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("REACHING_DEFS_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_ssa_construct_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("SSA_CONSTRUCT_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_alias_analysis_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("ALIAS_ANALYSIS_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_dominance_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("DOMINANCE_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_loop_detect_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("LOOP_DETECT_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_gvn_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("GVN_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_induction_var_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("INDUCTION_VAR_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_mega_batch_dataflow_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("MEGA_BATCH_DATAFLOW_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_macro_expand_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("MACRO_EXPAND_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_borrow_check_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("BORROW_CHECK_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

pub fn load_partition_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("PARTITION_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}

/// Load the fused MIR optimization shader (4 analyses in 1 dispatch).
pub fn load_fused_mir_opt_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("FUSED_MIR_OPT_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
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
