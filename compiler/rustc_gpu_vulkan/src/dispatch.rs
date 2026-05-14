use ash::vk;
use crate::buffer::GpuBuffer;
use crate::shader::ComputePipeline;
use crate::context::GpuContext;

/// GPU dispatch engine for monomorphization and collection.
/// 
/// Uses persistent descriptor pools, command buffers, and fences to amortize
/// allocation overhead across many dispatches.
pub struct GpuDispatch<'ctx> {
    context: &'ctx GpuContext,
    // Persistent resources
    descriptor_pool: vk::DescriptorPool,
    cmd_pool: vk::CommandPool,
    _descriptor_set: vk::DescriptorSet,
    cmd_buf: vk::CommandBuffer,
    fence: vk::Fence,
}

impl<'ctx> GpuDispatch<'ctx> {
    pub fn new(context: &'ctx GpuContext) -> Result<Self, vk::Result> {
        let device = &context.device;

        // Create persistent descriptor pool (max 4 storage buffers for counter variant)
        let descriptor_pool = unsafe {
            let pool_size = vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4);
            let create_info = vk::DescriptorPoolCreateInfo::default()
                .pool_sizes(std::slice::from_ref(&pool_size))
                .max_sets(1)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
            device.create_descriptor_pool(&create_info, None)?
        };

        // Pre-allocate descriptor set (layout will be set per-dispatch)
        // Note: descriptor_set_layout is per-pipeline, so we'll allocate per-pipeline
        // For simplicity, we'll keep a single descriptor pool and reallocate sets
        let cmd_pool = unsafe {
            let create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(context.queue_family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            device.create_command_pool(&create_info, None)?
        };

        let cmd_buf = unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            device.allocate_command_buffers(&alloc_info)?[0]
        };

        let fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        // Dummy descriptor set (will be reallocated per-pipeline)
        let _descriptor_set = vk::DescriptorSet::null();

        Ok(GpuDispatch {
            context,
            descriptor_pool,
            cmd_pool,
            _descriptor_set,
            cmd_buf,
            fence,
        })
    }

    fn reset(&self) -> Result<(), vk::Result> {
        let device = &self.context.device;
        unsafe {
            device.reset_fences(std::slice::from_ref(&self.fence))?;
            device.reset_command_pool(
                self.cmd_pool,
                vk::CommandPoolResetFlags::empty(),
            )?;
        }
        Ok(())
    }

    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        actions_buf: &GpuBuffer,
        offsets_buf: &GpuBuffer,
        edges_buf: &GpuBuffer,
        num_bodies: u32,
    ) -> Result<(), vk::Result> {
        self.dispatch_with_counter(pipeline, actions_buf, offsets_buf, edges_buf, None, num_bodies)
    }

    pub fn dispatch_with_counter(
        &self,
        pipeline: &ComputePipeline,
        actions_buf: &GpuBuffer,
        offsets_buf: &GpuBuffer,
        edges_buf: &GpuBuffer,
        counter_buf: Option<&GpuBuffer>,
        num_bodies: u32,
    ) -> Result<(), vk::Result> {
        let device = &self.context.device;

        // Reset persistent resources
        self.reset()?;

        // Allocate descriptor set for this pipeline's layout
        let descriptor_set = unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(std::slice::from_ref(&pipeline.descriptor_set_layout));
            device.allocate_descriptor_sets(&alloc_info)?[0]
        };

        // Write descriptor set
        let mut buffer_infos = vec![
            vk::DescriptorBufferInfo::default().buffer(actions_buf.buffer).range(actions_buf.size),
            vk::DescriptorBufferInfo::default().buffer(offsets_buf.buffer).range(offsets_buf.size),
            vk::DescriptorBufferInfo::default().buffer(edges_buf.buffer).range(edges_buf.size),
        ];

        if let Some(counter) = counter_buf {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default().buffer(counter.buffer).range(counter.size),
            );
        }

        let mut writes = vec![
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
        ];

        if counter_buf.is_some() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buffer_infos[3])),
            );
        }

        unsafe { device.update_descriptor_sets(&writes, &[]); }

        // Record command buffer (reuse persistent one)
        let cmd_buf = self.cmd_buf;
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd_buf, &begin_info)?;

            device.cmd_bind_pipeline(cmd_buf, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                std::slice::from_ref(&descriptor_set),
                &[],
            );

            // Push constants for num_bodies
            let push_constants = num_bodies.to_ne_bytes();
            device.cmd_push_constants(
                cmd_buf,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push_constants,
            );

            let workgroup_count = (num_bodies + 63) / 64;
            device.cmd_dispatch(cmd_buf, workgroup_count, 1, 1);

            // Memory barrier for edges_buf and optionally counter_buf
            let mut barriers = vec![
                vk::BufferMemoryBarrier::default()
                    .buffer(edges_buf.buffer)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)
                    .size(vk::WHOLE_SIZE),
            ];

            if let Some(counter) = counter_buf {
                barriers.push(
                    vk::BufferMemoryBarrier::default()
                        .buffer(counter.buffer)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ)
                        .size(vk::WHOLE_SIZE),
                );
            }

            let barrier_slice = &barriers[..];

            device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                barrier_slice,
                &[],
            );

            device.end_command_buffer(cmd_buf)?;
        }

        // Submit with persistent fence
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
        unsafe {
            device.queue_submit(self.context.queue, std::slice::from_ref(&submit_info), self.fence)?;
            device.wait_for_fences(std::slice::from_ref(&self.fence), true, u64::MAX)?;
        }

        // Free descriptor set back to pool for reuse
        unsafe {
            device.free_descriptor_sets(self.descriptor_pool, std::slice::from_ref(&descriptor_set))?;
        }

        Ok(())
    }
}

impl<'ctx> Drop for GpuDispatch<'ctx> {
    fn drop(&mut self) {
        unsafe {
            let device = &self.context.device;
            device.destroy_fence(self.fence, None);
            device.free_command_buffers(self.cmd_pool, std::slice::from_ref(&self.cmd_buf));
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
