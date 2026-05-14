use ash::vk;

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub mapped: *mut u8,
}

impl GpuBuffer {
    pub fn new_host_visible(
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        instance: &ash::Instance,
        size: vk::DeviceSize,
    ) -> Result<Self, vk::Result> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        
        let memory_type_index = (0..mem_properties.memory_type_count)
            .find(|&i| {
                let mem_type = mem_properties.memory_types[i as usize];
                (mem_requirements.memory_type_bits & (1 << i)) != 0
                    && mem_type.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .ok_or(vk::Result::ERROR_UNKNOWN)?;
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index as u32);
        
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };
        
        let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8 };
        
        Ok(GpuBuffer { buffer, memory, size, mapped })
    }
    
    pub fn write<T: Copy>(&self, data: &[T]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<T>())
        };
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.mapped, bytes.len());
        }
    }
    
    pub fn read<T: Copy>(&self, count: usize) -> Vec<T> {
        let byte_count = count * std::mem::size_of::<T>();
        let mut result = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(self.mapped, result.as_mut_ptr() as *mut u8, byte_count);
            result.set_len(count);
        }
        result
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Safe cleanup requires device reference — handled by context Drop order
    }
}
