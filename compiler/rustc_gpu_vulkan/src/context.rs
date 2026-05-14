use ash as _;
use gpu_alloc as _;
use gpu_alloc_ash as _;
use tracing as _;

pub struct GpuContext;

impl GpuContext {
    pub fn new() -> Result<Self, ()> {
        Ok(GpuContext)
    }
}
