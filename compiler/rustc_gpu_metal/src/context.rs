use metal::{Device, CommandQueue};

/// Metal GPU context: device + command queue.
///
/// Much simpler than Vulkan: no instance, physical device, or queue family enumeration.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
}

impl MetalContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::system_default()
            .ok_or("No Metal device found. This requires macOS with Metal support.")?;
        
        let queue = device.new_command_queue();
        
        Ok(MetalContext { device, queue })
    }
}
