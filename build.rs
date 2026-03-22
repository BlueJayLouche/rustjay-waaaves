fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=SYPHON_FRAMEWORK_DIR");

        let syphon_framework_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| {
                let candidate = p.join("syphon-rs/syphon-lib");
                if candidate.join("Syphon.framework").exists() {
                    Some(candidate)
                } else {
                    None
                }
            })
            .or_else(|| {
                std::env::var("SYPHON_FRAMEWORK_DIR")
                    .ok()
                    .map(std::path::PathBuf::from)
                    .filter(|path| path.join("Syphon.framework").exists())
            })
            .expect(
                "Syphon.framework not found. Set SYPHON_FRAMEWORK_DIR to the directory \
                 containing Syphon.framework, or place it at <workspace>/../syphon-rs/syphon-lib/",
            );
        let syphon_framework_dir = syphon_framework_dir.to_string_lossy().into_owned();

        println!("cargo:rustc-link-arg=-F{}", syphon_framework_dir);
        println!("cargo:rustc-link-arg=-framework");
        println!("cargo:rustc-link-arg=Syphon");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", syphon_framework_dir);

        let ndi_lib_paths = ["/usr/local/lib", "/Library/NDI SDK for Apple/lib/macOS"];
        for path in &ndi_lib_paths {
            if std::path::Path::new(path).exists() {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path);
            }
        }

        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../Frameworks");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
    }
}
