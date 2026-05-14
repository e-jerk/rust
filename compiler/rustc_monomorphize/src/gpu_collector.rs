#![allow(unused_imports, dead_code, unreachable_pub)]

use rustc_middle::mono::MonoItem;
use rustc_middle::ty::Instance;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuMonoAction {
    pub kind: u32,
    pub def_id: u32,
    pub args_idx: u32,
    pub _padding: u32,
}

pub const ACTION_CALL: u32 = 1;
pub const ACTION_DROP: u32 = 2;
pub const ACTION_CAST: u32 = 3;
pub const ACTION_CONST: u32 = 4;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GpuEdge {
    pub def_id: u32,
    pub args_idx: u32,
    pub source_idx: u32, // which body in the batch emitted this edge
    pub _padding: u32,
}

pub struct SerializedBatch<'tcx> {
    pub actions: Vec<GpuMonoAction>,
    pub body_offsets: Vec<u32>,
    pub generic_args_table: Vec<rustc_middle::ty::GenericArgsRef<'tcx>>,
    pub instances: Vec<Instance<'tcx>>, // source instances for this batch
}
