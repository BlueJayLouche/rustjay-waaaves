//! Build script for RustJay Waaaves
//!
//! Links the local Syphon framework for inter-app video on macOS.
//! This uses the framework from ../crates/syphon/syphon-lib/ rather than
//! the system framework, avoiding install name issues.

use std::path::PathBuf;

fn main() {
    #[cfg(target_os = "macos")]
    link_local_syphon_framework();
}

#[cfg(target_os = "macos")]
fn link_local_syphon_framework() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Path to the local Syphon framework (same one used by rusty-404)
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let framework_path = manifest_dir
        .parent().unwrap()  // .. from rustjay_waaaves
        .join("crates")
        .join("syphon")
        .join("syphon-lib");
    
    let framework_full = if framework_path.exists() {
        framework_path.canonicalize().unwrap_or_else(|_| framework_path.clone())
    } else {
        println!("cargo:warning=Local Syphon.framework not found at {:?}", framework_path);
        println!("cargo:warning=Attempting to fall back to /Library/Frameworks/");
        PathBuf::from("/Library/Frameworks")
    };
    
    let framework_binary = framework_path.join("Syphon.framework").join("Syphon");
    
    if framework_binary.exists() {
        // The local framework uses @rpath which is the correct modern approach
        // We just need to add our local path to the rpath search list
        
        // Tell cargo where to find the framework
        println!("cargo:rustc-link-search=framework={}", framework_full.display());
        
        // Add rpath so the binary can find the framework at runtime
        // This embeds the path in the binary
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", framework_full.display());
        
        // Also add an rpath that works from the target directory (for cargo run)
        println!("cargo:rustc-link-arg=-Wl,-rpath,../crates/syphon/syphon-lib");
        
        println!("cargo:warning=✅ Using local Syphon.framework from: {}", framework_full.display());
        
        // Verify the install name uses @rpath (which allows flexible loading)
        match std::process::Command::new("otool")
            .args(&["-D", &framework_binary.to_string_lossy()])
            .output() 
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if stdout.contains("@rpath") {
                    println!("cargo:warning=   Framework uses @rpath (flexible loading)");
                } else if stdout.contains("@loader_path") {
                    println!("cargo:warning=   ⚠️  Framework uses @loader_path (may need fix)");
                }
            }
            Err(_) => {}
        }
    } else {
        println!("cargo:warning=⚠️  Syphon.framework not found!");
        println!("cargo:warning=   Checked: {:?}", framework_path);
        println!("cargo:warning=   ");
        println!("cargo:warning=   The app may fail to run with 'Library not loaded' error.");
        println!("cargo:warning=   ");
        println!("cargo:warning=   To fix, ensure the framework exists at:");
        println!("cargo:warning=     crates/syphon/syphon-lib/Syphon.framework/");
    }
    
    // Link required frameworks (Metal, IOSurface, etc. are system frameworks)
    println!("cargo:rustc-link-lib=framework=IOSurface");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=CoreGraphics");
    println!("cargo:rustc-link-lib=framework=Foundation");
    
    // For the Syphon framework itself, we use the local copy
    // Note: cargo-bundle will handle embedding this for distribution
    println!("cargo:rustc-link-lib=framework=Syphon");
    
    // Add NDI library rpath — check both known install locations
    let ndi_lib_paths = [
        "/usr/local/lib",
        "/Library/NDI SDK for Apple/lib/macOS",
    ];
    for path in &ndi_lib_paths {
        if std::path::Path::new(path).join("libndi.dylib").exists() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
            println!("cargo:warning=   Added NDI rpath: {}", path);
        }
    }

    // Standard relative rpaths for cargo run and app bundles
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
}
