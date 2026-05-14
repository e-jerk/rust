#![allow(internal_features)]

pub mod buffer;
pub mod context;
pub mod dataflow;
pub mod dispatch;
pub mod shader;

use gpu_alloc as _;
use gpu_alloc_ash as _;
use tracing as _;
use std::sync::Arc;

/// Feature-gated GPU backend. Returns None if Vulkan unavailable.
pub struct GpuBackend {
    pub context: Arc<context::GpuContext>,
}

impl GpuBackend {
    pub fn new() -> Option<Self> {
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

pub fn load_mono_collect_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("MONO_COLLECT_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
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
