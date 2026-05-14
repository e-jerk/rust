use metal::{Device, Buffer, MTLResourceOptions};

/// Metal buffer wrapper with unified memory (StorageModeShared).
///
/// On Apple Silicon, StorageModeShared gives zero-copy access — the GPU and CPU
/// share the same physical memory. No staging buffers or explicit transfers needed.
pub struct MetalBuffer {
    pub buffer: Buffer,
    pub size: u64,
}

impl MetalBuffer {
    pub fn new(device: &Device, size: u64) -> Option<Self> {
        let buffer = device.new_buffer(
            size,
            MTLResourceOptions::StorageModeShared,
        );
        Some(MetalBuffer { buffer, size })
    }
    
    pub fn write<T>(&self, data: &[T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer.contents() as *mut u8,
                data.len() * std::mem::size_of::<T>(),
            );
        }
    }
    
    pub fn read<T: Clone>(&self, count: usize) -> Vec<T> {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents() as *const T,
                count,
            ).to_vec()
        }
    }
}
