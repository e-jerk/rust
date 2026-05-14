use ash::vk;

pub struct GpuContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
}

impl GpuContext {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let entry = unsafe { ash::Entry::load()? };
        
        let app_info = vk::ApplicationInfo::default()
            .api_version(vk::make_api_version(0, 1, 2, 0));
        
        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info);
        
        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };
        
        // Pick first discrete GPU, fallback to integrated
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let physical_device = physical_devices.iter().find(|&&pd| {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
        }).copied().or_else(|| physical_devices.first().copied())
            .ok_or("No Vulkan physical device found")?;
        
        let queue_family_index = 0u32; // Simplified: assume compute-capable queue at index 0
        
        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info));
        
        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        
        let pool_create_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None)? };
        
        Ok(GpuContext {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
        })
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
