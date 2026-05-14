use std::process::Command;

fn compile_shader(shader_path: &str, out_dir: &str, env_name: &str) {
    let shader_name = std::path::Path::new(shader_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap();
    let spv_path = format!("{}/{}.spv", out_dir, shader_name);

    let output = Command::new("glslangValidator")
        .args(["-V", shader_path, "-o", &spv_path, "--target-env", "vulkan1.2"])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            println!("cargo:rerun-if-changed={}", shader_path);
            println!("cargo:rustc-env={}={}", env_name, spv_path);
        }
        Ok(out) => {
            eprintln!(
                "glslangValidator stderr: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            panic!("Failed to compile shader {}", shader_path);
        }
        Err(e) => {
            eprintln!(
                "Warning: glslangValidator not found: {}. SPIR-V will not be compiled.",
                e
            );
        }
    }
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();

    compile_shader(
        "src/shaders/mono_collect.comp",
        &out_dir,
        "MONO_COLLECT_SPV",
    );
    compile_shader(
        "src/shaders/dataflow.comp",
        &out_dir,
        "DATAFLOW_SPV",
    );
    compile_shader(
        "src/shaders/dead_store_elim.comp",
        &out_dir,
        "DEAD_STORE_ELIM_SPV",
    );
    compile_shader(
        "src/shaders/copy_prop.comp",
        &out_dir,
        "COPY_PROP_SPV",
    );
    compile_shader(
        "src/shaders/const_prop.comp",
        &out_dir,
        "CONST_PROP_SPV",
    );
    compile_shader(
        "src/shaders/reaching_defs.comp",
        &out_dir,
        "REACHING_DEFS_SPV",
    );
    compile_shader(
        "src/shaders/ssa_construct.comp",
        &out_dir,
        "SSA_CONSTRUCT_SPV",
    );
    compile_shader(
        "src/shaders/alias_analysis.comp",
        &out_dir,
        "ALIAS_ANALYSIS_SPV",
    );
    compile_shader(
        "src/shaders/dominance.comp",
        &out_dir,
        "DOMINANCE_SPV",
    );
    compile_shader(
        "src/shaders/loop_detect.comp",
        &out_dir,
        "LOOP_DETECT_SPV",
    );
    compile_shader(
        "src/shaders/gvn.comp",
        &out_dir,
        "GVN_SPV",
    );
    compile_shader(
        "src/shaders/induction_var.comp",
        &out_dir,
        "INDUCTION_VAR_SPV",
    );
    compile_shader(
        "src/shaders/mega_batch_dataflow.comp",
        &out_dir,
        "MEGA_BATCH_DATAFLOW_SPV",
    );
    compile_shader(
        "src/shaders/macro_expand.comp",
        &out_dir,
        "MACRO_EXPAND_SPV",
    );
    compile_shader(
        "src/shaders/borrow_check.comp",
        &out_dir,
        "BORROW_CHECK_SPV",
    );
}
