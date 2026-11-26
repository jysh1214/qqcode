//! Builds `third_party/opencv` and installs it under `third_party/_install/opencv`: the few modules that
//! `qqcode` takes frames with, without CUDA and without the neural networks.
//!
//! The build tree lives in `third_party/_build` rather than in `OUT_DIR` on purpose: `OUT_DIR` is per
//! profile and changes with every toolchain update, and recompiling OpenCV takes minutes. CMake keeps a
//! tree incremental, and a stamp file lets an up-to-date tree skip CMake entirely.
//!
//! Environment variables (all optional):
//! * `CMAKE`: cmake executable, default `cmake` from `PATH`
//! * `CMAKE_PREFIX_PATH`: extra prefixes to search, e.g. for an FFmpeg that is not installed system-wide
//! * `QQCODE_OPENCV_CMAKE_ARGS`: extra arguments appended to the CMake command line (whitespace separated)

use std::env;
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// OpenCV modules the application uses. `BUILD_LIST` adds whatever these depend on.
const MODULES: &[&str] = &["core", "imgproc", "imgcodecs", "videoio"];

/// Variables that change the result of the build; cargo re-runs this script when one of them changes.
const ENV_VARS: &[&str] = &[
    "CMAKE",
    "QQCODE_OPENCV_CMAKE_ARGS",
    "CMAKE_PREFIX_PATH",
    "CC",
    "CXX",
];

const STAMP_NAME: &str = "qqcode-build-stamp";

/// Extension of the unversioned shared-library symlink CMake installs, which differs by platform
/// (`libopencv_core.so` vs. `libopencv_core.dylib`).
#[cfg(target_os = "macos")]
const SHARED_LIB_EXT: &str = "dylib";
#[cfg(not(target_os = "macos"))]
const SHARED_LIB_EXT: &str = "so";

/// What the dynamic loader replaces with the directory of the library that is being loaded.
#[cfg(target_os = "macos")]
const RPATH_ORIGIN: &str = "@loader_path";
#[cfg(not(target_os = "macos"))]
const RPATH_ORIGIN: &str = "$ORIGIN";

fn main() {
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let third_party = manifest_dir
        .join("../../third_party")
        .canonicalize()
        .expect("third_party/ directory next to crates/");
    let opencv_src = third_party.join("opencv");
    let build_dir = third_party.join("_build/opencv");
    let prefix = third_party.join("_install/opencv");
    let stamp = prefix.join(STAMP_NAME);

    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", stamp.display());
    for var in ENV_VARS {
        println!("cargo::rerun-if-env-changed={var}");
    }
    let revision = source_revision(&opencv_src);

    let args = cmake_args(&opencv_src, &prefix);
    let wanted_stamp = format!("{revision}\n{}\n", args.join("\n"));
    let up_to_date = fs::read_to_string(&stamp).is_ok_and(|have| have == wanted_stamp);
    if !up_to_date {
        // a half-finished build must not look current to the next run
        let _ = fs::remove_file(&stamp);
        build(&build_dir, &args);
        fs::write(&stamp, wanted_stamp).expect("can't write the build stamp");
    }

    let lib_dir = prefix.join("lib");
    let libs = installed_libs(&lib_dir);
    assert!(
        !libs.is_empty(),
        "no libopencv_*.{SHARED_LIB_EXT} in {}",
        lib_dir.display()
    );
    println!("cargo::metadata=root={}", prefix.display());
    println!(
        "cargo::metadata=include={}",
        prefix.join("include/opencv4").display()
    );
    println!("cargo::metadata=lib={}", lib_dir.display());
    println!("cargo::metadata=libs={}", libs.join(","));
}

/// Commit of a source checkout, and a re-run trigger for when a different commit gets checked out.
fn source_revision(dir: &Path) -> String {
    if !dir.join("CMakeLists.txt").exists() && !dir.join("modules").exists() {
        panic!(
            "{} is empty, fetch the submodule first: git submodule update --init --depth 1",
            dir.display()
        );
    }
    let git = |args: &[&str]| {
        let out = Command::new("git").arg("-C").arg(dir).args(args).output().ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
    };
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        println!("cargo::rerun-if-changed={git_dir}/HEAD");
    }
    git(&["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown revision".to_string())
}

/// The CMake command line.
fn cmake_args(opencv_src: &Path, prefix: &Path) -> Vec<String> {
    let mut args = vec![
        "-S".to_string(),
        opencv_src.display().to_string(),
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()),
        "-DCMAKE_INSTALL_LIBDIR=lib".to_string(),
        // the installed libraries find each other, and keep finding an FFmpeg in a non-default location
        format!("-DCMAKE_INSTALL_RPATH={RPATH_ORIGIN}"),
        "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON".to_string(),
        format!("-DBUILD_LIST={}", MODULES.join(",")),
    ];
    if which("ninja") {
        args.push("-GNinja".to_string());
    }
    let common = [
        "-DBUILD_SHARED_LIBS=ON",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_DOCS=OFF",
        "-DBUILD_opencv_apps=OFF",
        "-DBUILD_JAVA=OFF",
        "-DBUILD_opencv_python2=OFF",
        "-DBUILD_opencv_python3=OFF",
        "-DOPENCV_GENERATE_PKGCONFIG=OFF",
    ];
    let rest = [
        // video files and cameras
        "-DWITH_FFMPEG=ON",
        "-DWITH_V4L=ON",
        "-DWITH_GSTREAMER=OFF",
        "-DWITH_OBSENSOR=OFF",
        // nothing below is used; IPP and ADE would also download archives at configure time
        "-DWITH_CUDA=OFF",
        "-DWITH_PROTOBUF=OFF",
        "-DWITH_FLATBUFFERS=OFF",
        "-DWITH_IPP=OFF",
        "-DWITH_ADE=OFF",
        "-DWITH_ITT=OFF",
        "-DWITH_OPENCL=OFF",
        "-DWITH_VA=OFF",
        "-DWITH_EIGEN=OFF",
        "-DWITH_LAPACK=OFF",
        "-DWITH_GTK=OFF",
        "-DWITH_QT=OFF",
        "-DWITH_VTK=OFF",
        "-DWITH_TIFF=OFF",
        "-DWITH_WEBP=OFF",
        "-DWITH_OPENEXR=OFF",
        "-DWITH_JASPER=OFF",
        "-DWITH_OPENJPEG=OFF",
        "-DWITH_AVIF=OFF",
    ];
    args.extend(common.iter().chain(&rest).map(|s| s.to_string()));
    if let Ok(extra) = env::var("QQCODE_OPENCV_CMAKE_ARGS") {
        args.extend(extra.split_whitespace().map(str::to_string));
    }
    args
}

fn build(build_dir: &Path, args: &[String]) {
    fs::create_dir_all(build_dir).expect("can't create the OpenCV build directory");
    let cmake = env::var_os("CMAKE").unwrap_or_else(|| "cmake".into());
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "1".to_string());

    let configure_log = build_dir.join("qqcode-configure.log");
    let mut configure = Command::new(&cmake);
    configure.arg("-B").arg(build_dir).args(args);
    run(configure, &configure_log, "configuring OpenCV");
    let summary = fs::read_to_string(&configure_log).unwrap_or_default();
    // OpenCV only notes a missing FFmpeg in its summary, without it no video file can be opened
    if !summary_says_yes(&summary, "FFMPEG:") {
        panic!(
            "OpenCV was configured without FFmpeg (development files for libavcodec/libavformat/libswscale are needed), see {}",
            configure_log.display()
        );
    }

    let build_log = build_dir.join("qqcode-build.log");
    let mut build = Command::new(&cmake);
    build
        .arg("--build")
        .arg(build_dir)
        .args(["--target", "install", "--parallel", &jobs]);
    run(build, &build_log, "building OpenCV");
}

/// Runs `cmd` with its output going to `log`; a failure panics with the end of that log.
fn run(mut cmd: Command, log: &Path, what: &str) {
    let out = File::create(log).unwrap_or_else(|e| panic!("can't create {}: {e}", log.display()));
    let err = out.try_clone().expect("can't duplicate the log handle");
    let status = cmd
        .stdin(Stdio::null())
        .stdout(out)
        .stderr(err)
        .status()
        .unwrap_or_else(|e| match e.kind() {
            ErrorKind::NotFound => panic!(
                "{:?} was not found, put cmake on PATH or point the CMAKE environment variable at it",
                cmd.get_program()
            ),
            _ => panic!("can't run {:?}: {e}", cmd.get_program()),
        });
    if status.success() {
        return;
    }
    let text = fs::read_to_string(log).unwrap_or_default();
    let lines = text.lines().collect::<Vec<_>>();
    let tail = lines[lines.len().saturating_sub(60)..].join("\n");
    panic!("{what} failed ({status}), full log: {}\n{tail}", log.display());
}

/// Whether OpenCV's configure summary has a `-- <key> YES ...` line.
fn summary_says_yes(summary: &str, key: &str) -> bool {
    summary.lines().any(|line| {
        line.split_once(key)
            .is_some_and(|(before, value)| before.trim() == "--" && value.trim_start().starts_with("YES"))
    })
}

/// Names (without `lib` / the shared-library extension) of the installed OpenCV libraries.
fn installed_libs(lib_dir: &Path) -> Vec<String> {
    let suffix = format!(".{SHARED_LIB_EXT}");
    let mut libs = fs::read_dir(lib_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            Some(name.strip_prefix("lib")?.strip_suffix(&suffix)?.to_string())
        })
        // macOS has the version in front of the extension (`libopencv_core.4.15.0.dylib`), which would list
        // every library three times
        .filter(|name| name.starts_with("opencv_") && !name.contains('.'))
        .collect::<Vec<_>>();
    libs.sort();
    libs
}

fn which(program: &str) -> bool {
    env::var_os("PATH").is_some_and(|paths| env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
}
