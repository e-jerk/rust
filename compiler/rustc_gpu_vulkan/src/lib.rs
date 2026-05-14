#![allow(internal_features)]

pub mod buffer;
pub mod context;
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
}

pub fn load_mono_collect_shader() -> Option<Vec<u8>> {
    if let Ok(spv_path) = std::env::var("MONO_COLLECT_SPV") {
        std::fs::read(spv_path).ok()
    } else {
        None
    }
}
