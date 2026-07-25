//! Compiles the Fil-C memory safety runtime.

use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS was not set");
    let target_env = env::var("CARGO_CFG_TARGET_ENV").expect("CARGO_CFG_TARGET_ENV was not set");

    let mut cfg = cc::Build::new();
    cfg.file("cbits/filc_runtime.c");
    cfg.warnings(false);

    if target_env == "msvc" {
        cfg.flag("/Zl");
    } else {
        cfg.flag("-fno-builtin");
        if target_os != "windows" {
            cfg.flag("-fvisibility=default");
        }
    }

    // Needed for dlsym(RTLD_NEXT) interposition on Unix.
    if env::var_os("CARGO_CFG_UNIX").is_some() {
        println!("cargo::rustc-link-lib=dl");
        println!("cargo::rustc-link-lib=pthread");
    }

    println!("cargo::rerun-if-changed=cbits/filc_runtime.c");
    cfg.compile("filc-rt");
}
