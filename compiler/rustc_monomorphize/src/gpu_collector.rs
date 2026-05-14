#![allow(unused_imports, dead_code, unreachable_pub)]

use std::collections::VecDeque;

use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::unord::UnordSet;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{self, visit::Visitor, CastKind, Rvalue, TerminatorKind};
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, GenericArgsRef, Instance, Ty, TyCtxt};

use crate::collector::UsageMap;

const GPU_BATCH_SIZE: usize = 65536; // Mega-batch: process up to 64K items at once

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuMonoAction {
    pub kind: u32,
    pub def_id_index: u32,
    pub args_idx: u32,
    pub def_id_krate: u32,
}

pub const ACTION_CALL: u32 = 1;
pub const ACTION_DROP: u32 = 2;
pub const ACTION_CAST: u32 = 3;
pub const ACTION_CONST: u32 = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuEdge {
    pub def_id_index: u32,
    pub args_idx: u32,
    pub source_idx: u32, // which body in the batch emitted this edge
    pub def_id_krate: u32,
}

pub struct SerializedBatch<'tcx> {
    pub actions: Vec<GpuMonoAction>,
    pub body_offsets: Vec<u32>,
    pub generic_args_table: Vec<rustc_middle::ty::GenericArgsRef<'tcx>>,
    pub instances: Vec<Instance<'tcx>>, // source instances for this batch
}

struct GpuMirSerializer<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    body: &'a mir::Body<'tcx>,
    actions: &'a mut Vec<GpuMonoAction>,
    args_table: &'a mut Vec<GenericArgsRef<'tcx>>,
    args_index: &'a mut FxHashMap<GenericArgsRef<'tcx>, u32>,
    instance: Instance<'tcx>,
}

impl<'tcx, 'a> Visitor<'tcx> for GpuMirSerializer<'tcx, 'a> {
    fn visit_terminator(&mut self, terminator: &mir::Terminator<'tcx>, _location: mir::Location) {
        match &terminator.kind {
            TerminatorKind::Call { func, .. } | TerminatorKind::TailCall { func, .. } => {
                if let Some((def_id, args)) = self.resolve_callee(func) {
                    let args_idx = self.intern_args(args);
                    self.actions.push(GpuMonoAction {
                        kind: ACTION_CALL,
                        def_id_index: def_id.index.as_u32(),
                        args_idx,
                        def_id_krate: def_id.krate.as_u32(),
                    });
                }
            }
            TerminatorKind::Drop { place, .. } => {
                let ty = place.ty(self.body, self.tcx).ty;
                let ty = self.monomorphize(ty);
                if let Some((def_id, args)) = self.drop_glue_instance(ty) {
                    let args_idx = self.intern_args(args);
                    self.actions.push(GpuMonoAction {
                        kind: ACTION_DROP,
                        def_id_index: def_id.index.as_u32(),
                        args_idx,
                        def_id_krate: def_id.krate.as_u32(),
                    });
                }
            }
            _ => {}
        }
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: mir::Location) {
        match rvalue {
            Rvalue::Cast(
                CastKind::PointerCoercion(PointerCoercion::Unsize, ..),
                operand,
                _ty,
            ) => {
                let source_ty = operand.ty(self.body, self.tcx);
                let _source_ty_mono = self.monomorphize(source_ty);
                let _target_ty_mono = self.monomorphize(*_ty);
                // For unsizing casts, collect vtable methods.
                // This is complex; for MVP, we skip detailed vtable serialization.
            }
            _ => {}
        }
        self.super_rvalue(rvalue, location);
    }
}

impl<'tcx, 'a> GpuMirSerializer<'tcx, 'a> {
    fn monomorphize<T>(&self, value: T) -> T
    where
        T: ty::TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(value),
        )
    }

    fn intern_args(&mut self, args: GenericArgsRef<'tcx>) -> u32 {
        if let Some(&idx) = self.args_index.get(&args) {
            return idx;
        }
        let idx = self.args_table.len() as u32;
        self.args_table.push(args);
        self.args_index.insert(args, idx);
        idx
    }

    fn resolve_callee(&self, func: &mir::Operand<'tcx>) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let func_ty = func.ty(self.body, self.tcx);
        let func_ty = self.monomorphize(func_ty);
        match *func_ty.kind() {
            ty::FnDef(def_id, args) => Some((def_id, args)),
            _ => None,
        }
    }

    fn drop_glue_instance(&self, ty: Ty<'tcx>) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        // Simplified MVP: return the ADT's own def_id if it needs drop.
        // A full implementation would use Instance::resolve_drop_glue.
        match *ty.kind() {
            ty::Adt(adt, args) => {
                if ty.needs_drop(self.tcx, ty::TypingEnv::fully_monomorphized()) {
                    Some((adt.did(), args))
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

pub fn serialize_batch<'tcx>(
    tcx: TyCtxt<'tcx>,
    items: &[MonoItem<'tcx>],
) -> SerializedBatch<'tcx> {
    let mut actions = Vec::new();
    let mut body_offsets = Vec::new();
    let mut args_table = Vec::new();
    let mut args_index = FxHashMap::default();
    let mut instances = Vec::new();

    body_offsets.push(0);

    for item in items {
        let instance = match item {
            MonoItem::Fn(instance) => *instance,
            _ => continue, // Statics and global asm handled separately
        };

        let body = tcx.instance_mir(instance.def);
        let _start = actions.len() as u32;

        let mut serializer = GpuMirSerializer {
            tcx,
            actions: &mut actions,
            args_table: &mut args_table,
            args_index: &mut args_index,
            instance,
            body,
        };
        serializer.visit_body(body);

        body_offsets.push(actions.len() as u32);
        instances.push(instance);
    }

    SerializedBatch {
        actions,
        body_offsets,
        generic_args_table: args_table,
        instances,
    }
}

pub fn resolve_edge<'tcx>(
    tcx: TyCtxt<'tcx>,
    edge: GpuEdge,
    batch: &SerializedBatch<'tcx>,
) -> Option<Instance<'tcx>> {
    let def_id = DefId {
        krate: rustc_hir::def_id::CrateNum::from_u32(edge.def_id_krate),
        index: rustc_hir::def_id::DefIndex::from_u32(edge.def_id_index),
    };
    let args = batch.generic_args_table.get(edge.args_idx as usize)?;

    Instance::try_resolve(tcx, ty::TypingEnv::fully_monomorphized(), def_id, *args).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_mono_action_size() {
        assert_eq!(std::mem::size_of::<GpuMonoAction>(), 16);
        assert_eq!(std::mem::align_of::<GpuMonoAction>(), 4);
    }

    #[test]
    fn test_gpu_edge_size() {
        assert_eq!(std::mem::size_of::<GpuEdge>(), 16);
        assert_eq!(std::mem::align_of::<GpuEdge>(), 4);
    }

    #[test]
    fn test_action_kinds() {
        assert_eq!(ACTION_CALL, 1);
        assert_eq!(ACTION_DROP, 2);
        assert_eq!(ACTION_CAST, 3);
        assert_eq!(ACTION_CONST, 4);
    }

    #[test]
    fn test_action_roundtrip() {
        let action = GpuMonoAction {
            kind: ACTION_CALL,
            def_id_index: 42,
            args_idx: 7,
            def_id_krate: 1,
        };

        // Write to bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &action as *const _ as *const u8,
                std::mem::size_of::<GpuMonoAction>()
            )
        };

        // Read back
        let read_back = unsafe {
            std::ptr::read(bytes.as_ptr() as *const GpuMonoAction)
        };

        assert_eq!(read_back.kind, ACTION_CALL);
        assert_eq!(read_back.def_id_index, 42);
        assert_eq!(read_back.args_idx, 7);
        assert_eq!(read_back.def_id_krate, 1);
    }

    #[test]
    fn test_batch_structure() {
        let batch = SerializedBatch {
            actions: vec![
                GpuMonoAction { kind: ACTION_CALL, def_id_index: 1, args_idx: 0, def_id_krate: 0 },
                GpuMonoAction { kind: ACTION_DROP, def_id_index: 2, args_idx: 0, def_id_krate: 0 },
            ],
            body_offsets: vec![0, 1, 2],
            generic_args_table: vec![], // would have real args in production
            instances: vec![],
        };

        assert_eq!(batch.actions.len(), 2);
        assert_eq!(batch.body_offsets.len(), 3);
        assert_eq!(batch.body_offsets[0], 0);
        assert_eq!(batch.body_offsets[1], 1);
        assert_eq!(batch.body_offsets[2], 2);
    }
}

pub fn gpu_collect_mono_items<'tcx>(
    tcx: TyCtxt<'tcx>,
    roots: Vec<MonoItem<'tcx>>,
) -> Option<(Vec<MonoItem<'tcx>>, UsageMap<'tcx>)> {
    let backend = rustc_gpu_vulkan::GpuBackend::new()?;
    let pipeline = {
        let spirv = rustc_gpu_vulkan::load_mono_collect_shader()?;
        rustc_gpu_vulkan::shader::ComputePipeline::from_spirv(
            &backend.context.device,
            &spirv,
        ).ok()?
    };

    let mut visited = UnordSet::default();
    let mut queue = VecDeque::from(roots);
    let mut usage_map = UsageMap::new();

    while !queue.is_empty() {
        let batch_size = GPU_BATCH_SIZE.min(queue.len());
        let batch: Vec<_> = queue.drain(..batch_size).collect();

        // Serialize
        let serialized = serialize_batch(tcx, &batch);

        // Allocate GPU buffers
        let device = &backend.context.device;
        let physical_device = unsafe {
            backend.context.instance.enumerate_physical_devices().ok()?[0]
        };

        let actions_buf = rustc_gpu_vulkan::buffer::GpuBuffer::new_host_visible(
            device, physical_device, &backend.context.instance,
            (serialized.actions.len() * std::mem::size_of::<GpuMonoAction>()) as u64,
        ).ok()?;
        actions_buf.write(&serialized.actions);

        let offsets_buf = rustc_gpu_vulkan::buffer::GpuBuffer::new_host_visible(
            device, physical_device, &backend.context.instance,
            (serialized.body_offsets.len() * std::mem::size_of::<u32>()) as u64,
        ).ok()?;
        offsets_buf.write(&serialized.body_offsets);

        let max_edges = serialized.actions.len(); // worst case
        let edges_buf = rustc_gpu_vulkan::buffer::GpuBuffer::new_host_visible(
            device, physical_device, &backend.context.instance,
            (max_edges * std::mem::size_of::<GpuEdge>()) as u64,
        ).ok()?;

        // Dispatch
        let dispatch = rustc_gpu_vulkan::dispatch::GpuDispatch::new(&backend.context);
        dispatch.dispatch(
            &pipeline,
            &actions_buf,
            &offsets_buf,
            &edges_buf,
            serialized.instances.len() as u32,
        ).ok()?;

        // Read back
        let edges = edges_buf.read::<GpuEdge>(max_edges);
        // Note: we need to read the atomic counter too, but for MVP we can use edges.len()
        // In practice, the GPU writes edges sequentially and we need to know how many were written
        // For now, we'll read all and filter by def_id != 0

        // Resolve edges on CPU
        for edge in &edges {
            if edge.def_id_krate == 0 && edge.def_id_index == 0 {
                continue; // skip empty slots
            }
            if let Some(instance) = resolve_edge(tcx, *edge, &serialized) {
                let mono_item = MonoItem::Fn(instance);
                let source_idx = edge.source_idx as usize;
                if source_idx < batch.len() {
                    let source_item = batch[source_idx];
                    usage_map.record_usage(source_item, mono_item);
                }

                if visited.insert(mono_item) {
                    queue.push_back(mono_item);
                }
            }
        }
    }

    let mono_items = tcx.with_stable_hashing_context(|mut hcx| {
        visited.into_sorted(&mut hcx, true)
    });
    Some((mono_items, usage_map))
}


