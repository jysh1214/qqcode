# qqcode

Carries a file from one machine to another through a screen and a camera. `qqcode-show` writes a web page
that puts the file on the screen, `qqcode` reads it from the frames of a camera and writes it. Written in
Rust; the frames come through OpenCV.

1. Where the file is, write the page with `qqcode-show` and open it in a browser: it begins on a start
   screen.
2. Where the camera is, point it at that browser and start `qqcode`.
3. When `qqcode` says that it has the start screen in view, press `enter` on the page.
4. `qqcode` ends by itself when the file is whole. Both programs print the SHA-256 and the MD5 of the file,
   to be compared.

## Building

Needed on the machine: FFmpeg development files, `cmake` (and `ninja` if available), and `clang` with its
libclang.

```sh
git submodule update --init --depth 1
cargo build --release
```

OpenCV is not taken from the system: the first build compiles the checkout in `third_party/opencv`, which
takes a minute or two, see [third_party/README.md](third_party/README.md).

### Environment variables

Environment variables read by the OpenCV build:

| Variable | Meaning |
| --- | --- |
| `CMAKE` | cmake executable, default `cmake` |
| `CMAKE_PREFIX_PATH` | extra prefixes to search, e.g. an FFmpeg outside the system directories |
| `QQCODE_OPENCV_CMAKE_ARGS` | extra arguments for OpenCV's CMake configuration |

## Usage

The examples show and read the two files in `assets/`: `sample.txt`, 3758 bytes of text, and `sample.bin`,
65536 bytes.

```sh
qqcode > sample.txt                                      # camera 0
qqcode 2 -o sample.bin --preview                         # camera 2, with a view of what it sees
qqcode -o sample.bin --check sample.bin.checksums.txt    # held against the checksums from qqcode-show
qqcode --threads 4 --buffer 64 > sample.bin              # on a machine with little memory
qqcode --camera-size 1280x720 > sample.bin               # smaller frames
qqcode recording.mp4 > sample.bin                        # a video of the show instead of a camera
qqcode 'frame_%04d.png' > sample.bin                     # or pictures of it
```

- The file goes to the standard output, or to the file that `-o` names. Nothing of it is written before it
  is whole and its checksum is right.
- With `-o sample.bin` the file is put together on disk as it arrives, in `sample.bin.part`, which a file
  that is larger than the memory needs. A run that is stopped leaves that where it is, and the same command
  goes on with it.
- `--check sample.bin.checksums.txt` holds the file against the checksums that `qqcode-show` writes next to
  its page, megabyte by megabyte.
- `--preview` shows what the camera sees, for pointing it, with the pixels that a tile gets and how fast
  the file arrives; `q` quits.

`qqcode --help` lists every option.

## The web page

```sh
qqcode-show assets/sample.txt --html show.html                       # 176x99 tiles, 15 frames a second
qqcode-show assets/sample.bin --html show.html --scale 16 --rate 12  # more tiles, for a sharp camera
qqcode-show assets/sample.bin --html show.html --shape classic       # for a camera with frames of 4:3
qqcode-show sample.bin.checksums.txt --html list.html                # the checksums, to go ahead of the file
```

The page shows a matrix of coloured tiles, frame after frame without end: 10 kB of the file in a frame and
151 kB a second by default. Any frames that are enough make the file, so one that the camera misses costs
nothing. The matrix is best shown filling the frames of the camera; where frames are not read, a smaller
`--scale` or a camera nearer to the screen is the first thing to try.

Keys on the page: `enter` or `space` start and pause, `+` and `-` change the frames a second, `g` goes back
to the start screen, `f` takes the whole screen, `h` hides the notes.

Next to the page `qqcode-show` writes the checksums of the file, `sample.bin.checksums.txt` for
`assets/sample.bin`, for `qqcode --check`. That list has to be where the camera is before the file, and it is
a file like any other, so it can go the same way first. A page holds the file that it shows up to 64 MB; the
page of a longer file asks for the file when it is opened.

`qqcode-show --help` lists every option.

## Tests

```
cargo test
```

It needs neither a camera nor a browser. One test runs the script of the page in `deno`, and passes without
it.
