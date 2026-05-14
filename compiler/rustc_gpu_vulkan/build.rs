use std::process::Command;

fn main() {
    let shader_path = "src/shaders/mono_collect.comp";
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let spv_path = format!("{}/mono_collect.spv", out_dir);
    
    let output = Command::new("glslangValidator")
        .args(["-V", shader_path, "-o", &spv_path, "--target-env", "vulkan1.2"])
        .output();
    
    match output {
        Ok(out) if out.status.success() => {
            println!("cargo:rerun-if-changed={}", shader_path);
            println!("cargo:rustc-env=MONO_COLLECT_SPV={}", spv_path);
        }
        Ok(out) => {
            eprintln!("glslangValidator stderr: {}", String::from_utf8_lossy(&out.stderr));
            panic!("Failed to compile shader");
        }
        Err(e) => {
            eprintln!("Warning: glslangValidator not found: {}. SPIR-V will not be compiled.", e);
        }
    }
}
