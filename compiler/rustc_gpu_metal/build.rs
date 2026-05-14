use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Get OUT_DIR from cargo
    let out_dir = env::var("OUT_DIR")
        .expect("OUT_DIR environment variable not set by cargo");
    let out_path = PathBuf::from(out_dir);

    // Check if Metal toolchain is available
    let metal_available = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if !metal_available {
        println!("cargo:warning=Metal toolchain not available. Skipping .metal shader compilation.");
        println!("cargo:warning=Install with: xcodebuild -downloadComponent MetalToolchain");
        // Still emit rerun-if-changed so cargo rebuilds when shaders change
        let shaders_dir = PathBuf::from("src/shaders");
        if let Ok(entries) = fs::read_dir(&shaders_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("metal") {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
        return;
    }

    // Scan src/shaders/ for .metal files
    let shaders_dir = PathBuf::from("src/shaders");
    
    if !shaders_dir.exists() {
        return;
    }

    let entries = match fs::read_dir(&shaders_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Error reading shaders directory {}: {}", shaders_dir.display(), err);
            return;
        }
    };

    let mut metal_files = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(err) => {
                eprintln!("Error reading directory entry: {}", err);
                continue;
            }
        };
        
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "metal" {
                metal_files.push(path);
            }
        }
    }

    if metal_files.is_empty() {
        return;
    }

    // Compile each .metal file to .air and then to .metallib
    for metal_file in metal_files {
        let stem = metal_file.file_stem()
            .expect("Failed to get file stem")
            .to_string_lossy();
        
        let air_file = out_path.join(format!("{}.air", stem));
        let metallib_file = out_path.join(format!("{}.metallib", stem));

        // Compile .metal to .air
        let metal_output = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metal",
                "-c",
                metal_file.to_str().unwrap(),
                "-o",
                air_file.to_str().unwrap(),
            ])
            .output();

        match metal_output {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("Error compiling {} to .air: {}", metal_file.display(), stderr);
                    continue; // Skip this shader, don't panic
                }
            }
            Err(err) => {
                eprintln!("Error running xcrun metal command for {}: {}", metal_file.display(), err);
                continue;
            }
        }

        // Link .air to .metallib
        let metallib_output = Command::new("xcrun")
            .args([
                "-sdk", "macosx",
                "metallib",
                air_file.to_str().unwrap(),
                "-o",
                metallib_file.to_str().unwrap(),
            ])
            .output();

        match metallib_output {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("Error linking {} to .metallib: {}", air_file.display(), stderr);
                    continue;
                }
            }
            Err(err) => {
                eprintln!("Error running xcrun metallib command for {}: {}", air_file.display(), err);
                continue;
            }
        }

        // Set environment variable for runtime loading
        let env_name = format!("{}_METALLIB", stem.to_uppercase());
        let metallib_path = metallib_file.canonicalize()
            .unwrap_or_else(|_| metallib_file.clone());
        println!("cargo:rustc-env={}={}", env_name, metallib_path.display());
        
        // Set rerun-if-changed
        println!("cargo:rerun-if-changed={}", metal_file.display());
    }
}
