fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    // The OpenCV libraries are in the vendored install, which the dynamic loader doesn't search. Recording
    // that directory in the binaries lets them start without LD_LIBRARY_PATH.
    let lib_dir =
        std::env::var("DEP_OPENCV_VENDORED_LIB").expect("opencv-vendored exports its library directory");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{lib_dir}");
}
