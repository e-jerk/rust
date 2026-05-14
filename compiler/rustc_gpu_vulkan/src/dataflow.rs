use ash::vk;
use std::ffi::CStr;
use crate::buffer::GpuBuffer;
use crate::context::GpuContext;

/// Per-basic-block configuration, matching the GLSL `BlockConfig` struct.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuDataflowConfig {
    pub statement_count: u32,
    pub terminator_kind: u32,
    pub successor_count: u32,
    pub successor_0: u32,
    pub successor_1: u32,
}

/// Push constants for the dataflow compute shader.
#[repr(C)]
struct DataflowPushConstants {
    num_blocks: u32,
    bitset_words: u32,
    effects_stride: u32,
}

/// GPU dataflow engine that manages the compute pipeline for fixed-point iteration.
pub struct GpuDataflowEngine<'ctx> {
    context: &'ctx GpuContext,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

impl<'ctx> GpuDataflowEngine<'ctx> {
    /// Create a new dataflow engine from SPIR-V shader code.
    pub fn new(
        context: &'ctx GpuContext,
        spirv_code: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = &context.device;

        let shader_module = unsafe {
            let code = std::slice::from_raw_parts(
                spirv_code.as_ptr() as *const u32,
                spirv_code.len() / 4,
            );
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
                vk::DescriptorSetLayoutBinding::default()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            device.create_descriptor_set_layout(&create_info, None)?
        };

        let pipeline_layout = unsafe {
            let push_constant_range = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<DataflowPushConstants>() as u32);
            let create_info = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(std::slice::from_ref(&descriptor_set_layout))
                .push_constant_ranges(std::slice::from_ref(&push_constant_range));
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
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&create_info),
                    None,
                )
                .map_err(|e| e.1)?[0]
        };

        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(GpuDataflowEngine {
            context,
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
        })
    }

    /// Dispatch one dataflow round.
    ///
    /// Clears the convergence flag to 0 before dispatch so that `read_convergence`
    /// returns true only if at least one block changed during this round.
    pub fn dispatch_round(
        &self,
        config_buf: &GpuBuffer,
        effects_buf: &GpuBuffer,
        entry_buf: &GpuBuffer,
        exit_buf: &GpuBuffer,
        convergence_buf: &GpuBuffer,
        num_blocks: u32,
        bitset_words: u32,
        effects_stride: u32,
    ) -> Result<(), vk::Result> {
        let device = &self.context.device;

        // Clear convergence flag before dispatch
        unsafe {
            let ptr = convergence_buf.mapped as *mut u32;
            *ptr = 0;
        }

        // Allocate descriptor pool
        let descriptor_pool = unsafe {
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(5);
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(1);
            device.create_descriptor_pool(&create_info, None)?
        };

        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // Write descriptor set
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(config_buf.buffer)
                .range(config_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(effects_buf.buffer)
                .range(effects_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(entry_buf.buffer)
                .range(entry_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(exit_buf.buffer)
                .range(exit_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(convergence_buf.buffer)
                .range(convergence_buf.size),
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[3])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[4])),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // Record command buffer
        let cmd_buf = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.context.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            device.allocate_command_buffers(&alloc_info)?[0]
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            // Push constants
            let push_constants = DataflowPushConstants {
                num_blocks,
                bitset_words,
                effects_stride,
            };
            let push_bytes = std::slice::from_raw_parts(
                &push_constants as *const _ as *const u8,
                std::mem::size_of::<DataflowPushConstants>(),
            );
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );

            // Dispatch: local_size_x = 256
            let workgroup_count = (num_blocks + 255) / 256;
            device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);

            // Memory barrier for convergence and exit buffers
            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .buffer(exit_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .buffer(convergence_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
            ];
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );

            device.end_command_buffer(cmd_buf)?;
        }

        // Submit
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        unsafe {
            device.queue_submit(self.context.queue, std::slice::from_ref(&submit_info), fence)?;
        }
        unsafe {
            device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
        }

        // Cleanup
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.context.command_pool, std::slice::from_ref(&cmd_buf));
            device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(())
    }

    /// Read back the convergence flag.
    ///
    /// Returns `true` if at least one block's exit state changed during the last round.
    pub fn read_convergence(&self, convergence_buf: &GpuBuffer) -> bool {
        let data: Vec<u32> = convergence_buf.read(1);
        data[0] != 0
    }

    /// Dispatch one SSA construction round.
    ///
    /// This is a single-pass (non-iterative) dispatch for phi node insertion.
    pub fn dispatch_ssa_round(
        &self,
        block_info_buf: &GpuBuffer,
        def_sites_buf: &GpuBuffer,
        dom_frontier_buf: &GpuBuffer,
        phi_nodes_buf: &GpuBuffer,
        num_blocks: u32,
        num_locals: u32,
        max_defs_per_block: u32,
    ) -> Result<(), vk::Result> {
        let device = &self.context.device;

        // Allocate descriptor pool
        let descriptor_pool = unsafe {
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4);
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(1);
            device.create_descriptor_pool(&create_info, None)?
        };

        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // Write descriptor set
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(block_info_buf.buffer)
                .range(block_info_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(def_sites_buf.buffer)
                .range(def_sites_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(dom_frontier_buf.buffer)
                .range(dom_frontier_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(phi_nodes_buf.buffer)
                .range(phi_nodes_buf.size),
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[2])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[3])),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // Record command buffer
        let cmd_buf = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.context.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            device.allocate_command_buffers(&alloc_info)?[0]
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            // Push constants for SSA
            let push_constants = [num_blocks, num_locals, max_defs_per_block];
            let push_bytes = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * std::mem::size_of::<u32>(),
            );
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );

            // Dispatch: local_size_x = 64
            let workgroup_count = (num_blocks + 63) / 64;
            device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);

            // Memory barrier for output buffers
            let barriers = [
                vk::BufferMemoryBarrier::default()
                    .buffer(dom_frontier_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
                vk::BufferMemoryBarrier::default()
                    .buffer(phi_nodes_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
            ];
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                &barriers,
                &[],
            );

            device.end_command_buffer(cmd_buf)?;
        }

        // Submit
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        unsafe {
            device.queue_submit(self.context.queue, std::slice::from_ref(&submit_info), fence)?;
        }
        unsafe {
            device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
        }

        // Cleanup
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.context.command_pool, std::slice::from_ref(&cmd_buf));
            device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(())
    }

    /// Dispatch alias analysis kernel.
    ///
    /// Single-pass dispatch that computes alias relations between memory accesses.
    pub fn dispatch_alias_round(
        &self,
        desc_buf: &GpuBuffer,
        matrix_buf: &GpuBuffer,
        num_accesses: u32,
        num_locals: u32,
    ) -> Result<(), vk::Result> {
        let device = &self.context.device;

        // Allocate descriptor pool
        let descriptor_pool = unsafe {
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2);
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(1);
            device.create_descriptor_pool(&create_info, None)?
        };

        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(std::slice::from_ref(&self.descriptor_set_layout));
            device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // Write descriptor set
        let buffer_infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(desc_buf.buffer)
                .range(desc_buf.size),
            vk::DescriptorBufferInfo::default()
                .buffer(matrix_buf.buffer)
                .range(matrix_buf.size),
        ];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&buffer_infos[1])),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        // Record command buffer
        let cmd_buf = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.context.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            device.allocate_command_buffers(&alloc_info)?[0]
        };

        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            device.cmd_bind_pipeline(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            // Push constants for alias analysis
            let push_constants = [num_accesses, num_locals];
            let push_bytes = std::slice::from_raw_parts(
                push_constants.as_ptr() as *const u8,
                push_constants.len() * std::mem::size_of::<u32>(),
            );
            device.cmd_push_constants(
                cmd_buf,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_bytes,
            );

            // Dispatch: local_size_x = 64
            let workgroup_count = (num_accesses + 63) / 64;
            device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);

            // Memory barrier for output buffer
            let barrier = vk::BufferMemoryBarrier::default()
                .buffer(matrix_buf.buffer)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&barrier),
                &[],
            );

            device.end_command_buffer(cmd_buf)?;
        }

        // Submit
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        unsafe {
            device.queue_submit(self.context.queue, std::slice::from_ref(&submit_info), fence)?;
        }
        unsafe {
            device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
        }

        // Cleanup
        unsafe {
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.context.command_pool, std::slice::from_ref(&cmd_buf));
            device.destroy_descriptor_pool(descriptor_pool, None);
        }

        Ok(())
    }
}

impl<'ctx> Drop for GpuDataflowEngine<'ctx> {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_pipeline(self.pipeline, None);
            self.context
                .device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.context
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
