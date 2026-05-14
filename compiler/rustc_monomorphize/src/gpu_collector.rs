#![allow(unused_imports, dead_code, unreachable_pub)]

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::{self, visit::Visitor, CastKind, Rvalue, TerminatorKind};
use rustc_middle::mono::MonoItem;
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{self, GenericArgsRef, Instance, Ty, TyCtxt};

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
