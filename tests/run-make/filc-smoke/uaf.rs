use std::alloc::{Layout, alloc, dealloc};

fn main() {
    unsafe {
        let layout = Layout::from_size_align(16, 8).unwrap();
        let p = alloc(layout);
        assert!(!p.is_null());
        dealloc(p, layout);
        // Use-after-free: should panic under Fil-C when the allocation was tracked.
        std::ptr::write_volatile(p, 42u8);
    }
}