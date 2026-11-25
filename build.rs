fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // The OpenCV libraries are in the vendored install, which the dynamic loader doesn't search. Recording
    // that directory in the binaries lets them start without LD_LIBRARY_PATH. Without the feature `reader`
    // there is no OpenCV and nothing to record.
    if std::env::var_os("CARGO_FEATURE_READER").is_some() {
        let lib_dir =
            std::env::var("DEP_OPENCV_VENDORED_LIB").expect("opencv-vendored exports its library directory");
        println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
