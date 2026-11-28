# third_party

| Path | What | Version |
| --- | --- | --- |
| `opencv/` | OpenCV, git submodule | branch `4.x` at `a495c9b` (2026-11-23), reports itself as 4.15.0-dev |
| `opencv-rust/` | the [`opencv`](https://crates.io/crates/opencv) crate (Rust bindings), copied in and patched | 0.102.0 |
| `opencv-rust.patch` | the changes made to `opencv-rust/` | |
| `_build/`, `_install/` | OpenCV build tree and install prefix, generated, not in git | |

The submodule is not on a release tag because the latest 4.x release, 4.14.0, does not compile against
FFmpeg 9 (`AVCodec::pix_fmts` and `AVCodec::supported_framerates` are gone); the fix is on the `4.x` branch
only. Move it to the 4.15.0 tag once that exists.

## How the build fits together

`crates/opencv-vendored/build.rs` runs CMake on `opencv/` and installs the result into `_install/opencv`.
The modules to build and all CMake options are listed in that file: `core`, `imgproc`, `imgcodecs` and
`videoio`, without CUDA and without the neural networks, which is what it takes to get frames from a camera
or a file and to scale them for the preview. The finder patterns of the matrix are found by a Rust crate,
not by OpenCV.

The `opencv` crate generates its bindings from the installed headers, so OpenCV has to be built before that
crate's build script runs. Cargo only orders build scripts along dependency edges, and the published crate has
no dependency that could build OpenCV for it. That is why a copy of the crate lives here with a small patch:

- `Cargo.toml`: a dependency on `opencv-vendored`, which gives cargo the ordering and passes the install
  location on as `DEP_OPENCV_VENDORED_*`; three lints switched off that fire with few modules enabled; and
  the `[[example]]` and `[[test]]` sections taken out, see below;
- `build/library.rs`: use that location, and only that one, instead of probing the system for an OpenCV;
- `rustfmt.toml`: keeps `cargo fmt --all` from reformatting the copy.

What the build does not take of the published crate is left out: `docs/` (pre-generated bindings for docs.rs,
16 MB), `examples/`, `tests/`, and `AGENTS.md`, `CHANGES.md`, `INSTALL.md` and `TROUBLESHOOTING.md`. Cargo
builds neither the examples nor the tests of a dependency, but `cargo fmt --all` fails on a target whose file
is missing, which is why their sections are out of `Cargo.toml` as well.

## Updating

OpenCV: check out the new tag in the submodule. The next `cargo build` notices the new revision and rebuilds.

`opencv` crate: replace `opencv-rust/` with the new release from crates.io, without what is left out, and
re-apply the patch from inside it:

```sh
patch -p1 < ../opencv-rust.patch
```

A release with other examples or tests than 0.102.0 has other sections in `Cargo.toml`, and that part of the
patch no longer applies: every `[[example]]` and `[[test]]` section is taken out by hand then, and the patch
written anew.
