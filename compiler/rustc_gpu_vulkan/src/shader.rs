use ash::vk;
use std::ffi::CStr;

pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl ComputePipeline {
    pub fn from_spirv(
        device: &ash::Device,
        spirv_code: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shader_module = unsafe {
            let code = std::slice::from_raw_parts(spirv_code.as_ptr() as *const u32, spirv_code.len() / 4);
            let create_info = vk::ShaderModuleCreateInfo::default().code(code);
            device.create_shader_module(&create_info, None)?
        };
        
        let descriptor_set_layout = unsafe {
            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            device.create_descriptor_set_layout(&create_info, None)?
        };
        
        let pipeline_layout = unsafe {
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout));
            device.create_pipeline_layout(&create_info, None)?
        };
        
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
        
        let create_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        
        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&create_info), None)
                .map_err(|e| e.1)?[0]
        };
        
        unsafe { device.destroy_shader_module(shader_module, None); }
        
        Ok(ComputePipeline { pipeline, layout: pipeline_layout, descriptor_set_layout })
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // Cleanup requires device reference
    }
}