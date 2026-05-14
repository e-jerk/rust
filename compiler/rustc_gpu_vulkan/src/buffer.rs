use ash::vk;
use crate::context::GpuContext;

pub struct PersistentGpuBuffers {
    pub actions_buf: GpuBuffer,
    pub offsets_buf: GpuBuffer,
    pub edges_buf: GpuBuffer,
    pub counter_buf: GpuBuffer, // atomic counter for edge count
    pub max_actions: usize,
    pub max_edges: usize,
}

impl PersistentGpuBuffers {
    pub fn new(
        context: &GpuContext,
        physical_device: vk::PhysicalDevice,
        max_actions: usize,
        max_edges: usize,
    ) -> Result<Self, vk::Result> {
        let device = &context.device;
        let instance = &context.instance;
        
        let actions_buf = GpuBuffer::new_host_visible(
            device, physical_device, instance,
            (max_actions * std::mem::size_of::<u8>()) as u64,
        )?;
        
        let offsets_buf = GpuBuffer::new_host_visible(
            device, physical_device, instance,
            (max_actions * std::mem::size_of::<u32>()) as u64,
        )?;
        
        let edges_buf = GpuBuffer::new_host_visible(
            device, physical_device, instance,
            (max_edges * std::mem::size_of::<u8>()) as u64,
        )?;
        
        let counter_buf = GpuBuffer::new_host_visible(
            device, physical_device, instance,
            std::mem::size_of::<u32>() as u64,
        )?;
        
        Ok(PersistentGpuBuffers {
            actions_buf,
            offsets_buf,
            edges_buf,
            counter_buf,
            max_actions,
            max_edges,
        })
    }
    
    pub fn reset_counter(&self) {
        let zero: u32 = 0;
        self.counter_buf.write(&[zero]);
    }
}

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub mapped: *mut u8,
    device: ash::Device,
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
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        
        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        
        let memory_type_index = mem_properties.memory_types
            .iter()
            .enumerate()
            .position(|(i, t)| {
                (mem_requirements.memory_type_bits & (1 << i)) != 0
                    && t.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            })
            .unwrap_or(0) as u32;
        
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        
        unsafe { device.bind_buffer_memory(buffer, memory, 0)?; }
        
        let mapped = unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8 };
        
        Ok(GpuBuffer {
            buffer,
            memory,
            size,
            mapped,
            device: device.clone(),
        })
    }
    
    pub fn write<T>(&self, data: &[T]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.mapped,
                data.len() * std::mem::size_of::<T>(),
            );
        }
    }
    
    pub fn read<T>(&self, count: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mapped,
                result.as_mut_ptr() as *mut u8,
                count * std::mem::size_of::<T>(),
            );
            result.set_len(count);
        }
        result
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.unmap_memory(self.memory);
            self.device.free_memory(self.memory, None);
            self.device.destroy_buffer(self.buffer, None);
        }
    }
}
