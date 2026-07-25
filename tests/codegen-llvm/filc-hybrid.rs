// Rust's compile-time checks already cover most memory accesses, so `-Zfil-c` leaves
// those alone and only checks the ones it has to: raw pointer dereferences, atomics,
// and untyped copies. `-Zfilc-instrument-all` opts back in to checking everything.

//@ revisions: HYBRID ALL
//@ compile-flags: -Zfil-c -Copt-level=1
//@ [ALL] compile-flags: -Zfilc-instrument-all

#![crate_type = "lib"]

// Indexing a slice is bounds-checked by the panic branch above the load, and the
// reference itself was proven live by the borrow checker.
//
// CHECK-LABEL: @safe_index
#[no_mangle]
pub fn safe_index(xs: &[u32], i: usize) -> u32 {
    // HYBRID-NOT: filc_check_read
    // ALL: call void @filc_check_read
    xs[i]
}

// CHECK-LABEL: @safe_field_write
#[no_mangle]
pub fn safe_field_write(p: &mut (u32, u32), v: u32) {
    // HYBRID-NOT: filc_check_write
    // ALL: call void @filc_check_write
    p.1 = v;
}

// Nothing static rules out `p` dangling or being out of bounds, so this is where
// Fil-C earns its keep.
//
// CHECK-LABEL: @raw_read
#[no_mangle]
pub unsafe fn raw_read(p: *const u32) -> u32 {
    // CHECK: call void @filc_check_read
    *p
}

// CHECK-LABEL: @raw_write
#[no_mangle]
pub unsafe fn raw_write(p: *mut u32, v: u32) {
    // CHECK: call void @filc_check_write
    *p = v;
}

// A field projected out of a raw dereference is still raw.
//
// CHECK-LABEL: @raw_field_read
#[no_mangle]
pub unsafe fn raw_field_read(p: *const (u32, u32)) -> u32 {
    // CHECK: call void @filc_check_read
    (*p).1
}

// `copy_nonoverlapping` lowers to a `memcpy`, which needs both ends checked.
//
// CHECK-LABEL: @raw_copy
#[no_mangle]
pub unsafe fn raw_copy(dst: *mut u8, src: *const u8, n: usize) {
    // CHECK: call void @filc_check_write
    // CHECK: call void @filc_check_read
    std::ptr::copy_nonoverlapping(src, dst, n);
}
