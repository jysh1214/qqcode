# qqcode

Carries a file from one machine to another through a screen and a camera. `qqcode-show` shows the file on
a terminal as a sequence of QR codes, `qqcode` reads the codes from the frames of a camera and writes the
file. Written in Rust; the frames come through OpenCV.

```sh
qqcode-show notes.tar.gz     # where the file is, in a terminal that the camera sees
qqcode > notes.tar.gz        # where the camera is
```

`qqcode-show` cuts the file into pieces and puts each into a code, together with the checksum and the length
of the file and the place of the piece in it. `qqcode`, per frame:

1. finds the QR codes and decodes them (the QR reader of `rxing`);
2. takes the one that is a code of `qqcode-show` and puts its piece where it belongs in the file;
3. once every byte has arrived, checks the checksum, writes the file and ends.

Nothing in that is a neural network, so neither program has a use for a GPU.

## Building

OpenCV is not taken from the system. `cargo build` compiles the checkout in `third_party/opencv`, the four
modules of it that are used, and builds the Rust bindings against it, see
[third_party/README.md](third_party/README.md).

Needed on the machine:

- FFmpeg development files, for reading video files
- `cmake` on `PATH`, and `ninja` if available
- `clang` on `PATH` plus its libclang, for generating the bindings

```sh
git submodule update --init --depth 1
cargo build --release
```

The first build compiles OpenCV, which takes a minute or two; its log is
`third_party/_build/opencv/qqcode-build.log`. Later builds reuse `third_party/_build` and
`third_party/_install` (`cargo clean` leaves them alone, delete them to rebuild OpenCV from scratch).

The build does not notice a system update that replaces FFmpeg. If `qqcode` stops starting after one
(`error while loading shared libraries: libavformat.so...`), delete those two directories and build again.

### Environment variables

Environment variables read by the OpenCV build:

| Variable | Meaning |
| --- | --- |
| `CMAKE` | cmake executable, default `cmake` |
| `CMAKE_PREFIX_PATH` | extra prefixes to search, e.g. an FFmpeg outside the system directories |
| `QQCODE_OPENCV_CMAKE_ARGS` | extra arguments for OpenCV's CMake configuration |

## Usage

```sh
qqcode > notes.tar.gz                                  # camera 0
qqcode 2 -o notes.tar.gz --preview                     # camera 2, with a view of what it sees
qqcode --camera-size 1280x720 > notes.tar.gz           # smaller frames
qqcode recording.mp4 > notes.tar.gz                    # a video of the show instead of a camera
qqcode 'frame_%04d.png' > notes.tar.gz                 # or pictures of it
```

| Option | Default | |
| --- | --- | --- |
| input | `0` | camera index, or a video file, an image or an image sequence |
| `-o`, `--output` | the standard output | file to write |
| `--camera-size` | `1920x1080` | frame size to ask a camera for, `WIDTHxHEIGHT`; see below |
| `--preview` | off | show what the camera sees on the terminal, with the codes that are found; see below |

The file is written when it is whole and its checksum is right, in one piece, and nothing of it before that:
a run that is stopped, or whose video ends too early, leaves an empty file behind `>` and none at all behind
`-o`, and ends with an error. The file goes to the standard output, so `qqcode` refuses to start with that
being the terminal. What it has to say while it runs goes to the standard error:

```
camera 0: 1920x1080 frames, 30 a second
frame 31: start screen in view: notes.tar.gz, 3000 bytes, waiting for the first code
frame 58: 842 of 3000 bytes
frame 64: 1684 of 3000 bytes
frame 70: 2526 of 3000 bytes
frame 76: the file is whole
76 frames read, 3000 bytes written to the standard output
SHA-256 03172bf1dc18c2c972ff3b2cca2fcf4e800c6dd590474210b53c193aaeaf8bc9
MD5 7850bf91f76b3d7e90d3b7b4eacec425
```

The last two lines are the SHA-256 and the MD5 of the file as it was written. `qqcode-show` has those of
the file it shows on its end screen and prints them when it quits, and `sha256sum` and `md5sum` say the
same of either file: they are there to be compared, by whoever wants more than the checksum in the codes,
which `qqcode` has checked by then.

### Codes that are missed

A code that no frame shows sharply enough is a hole in the file. `qqcode` says so when the next code comes
(`frame 64: 842 bytes went by unread`), and when the end screen comes it names what is missing:

```
frame 90: end screen in view, 842 of 3000 bytes are missing: 842 to 1683; g in qqcode-show shows the codes again
```

It goes on reading then. `g` in `qqcode-show` begins the show again, and `qqcode` takes from the second
round what it lacks and ends as soon as it has everything; codes it has read before cost it nothing. The
rounds need not be alike: a show with larger modules, slower, or in a window of another size fills the same
file, as every code says where its bytes go rather than which code it is.

Should all pieces arrive and the checksum be wrong, a code was misread in a way that its own error
correction did not notice. `qqcode` says so, drops what it has and collects the file anew.

### Cameras

A camera delivers the size it starts up with, which is 640x480 for nearly all of them and too little for
codes of any size, so `qqcode` asks for 1920x1080 unless `--camera-size` names another size. It says which
frames it gets when it starts; if the camera does not have the size asked for, those are of the nearest one
it has, and `v4l2-ctl --list-formats-ext` lists them all. USB cameras tend to have their large sizes as
Motion JPEG only, or uncompressed at a few frames a second, so `qqcode` switches to Motion JPEG where that
comes nearer to the size or brings more frames a second: a code that is on show for half a second is easily
missed at five frames a second.

To point the camera, `--preview` draws what it sees on the terminal that `qqcode` runs in, a frame at a
time, as coarse as the terminal has character cells (a smaller font gives a finer picture). In the picture
a code of `qqcode-show` is outlined in green, any other QR code in magenta, and a QR code that is found but
can't be decoded in red. The lines below it say what the code in view is, how many modules it has to a side
and how many pixels of the frame a module gets, what to do about it, and how much of the file has arrived.
Reading goes on as without the preview; what `qqcode` would otherwise write to the terminal while it runs
is printed when it ends. `q` ends it. The preview is drawn through the standard error, which leaves the
standard output to the file.

## `qqcode-show`

```sh
qqcode-show notes.tar.gz                              # a new code every half second
qqcode-show notes.tar.gz --interval 0.2 --level l     # faster, and more in a code
qqcode-show notes.tar.gz --interval 0                 # the codes turn by key
```

`qqcode-show` begins on a start screen and waits there. Its code is of the size that the codes of the file
will have, and says what the file is called and how long it is. `qqcode` says when it has that screen in
view, which is the moment to press `enter`: the codes follow, every `--interval` seconds or by key, and
after the last one an end screen. `qqcode` ends by itself with the code that completes the file. If it
names missing parts at the end screen instead, `g` shows the codes again.

The code is drawn with half block characters, a module of it as a square of half a character cell, and it
is as large as the terminal has room for: the larger the window and the smaller the font, the more modules,
and the fewer codes for a file. Beside the code, or below it in a narrow window, there is a panel that says
what is on show. The file is cut anew whenever the window, the font or one of the settings below changes,
and the code that is on show then is the one with the byte that the code before began with.

| Terminal | Code | Bytes of the file in a code at level `m` |
| --- | --- | --- |
| 80x24 cells | version 5, 37 modules | 69 |
| 120x40 | version 13, 69 modules | 316 |
| 200x60 | version 23, 109 modules | 842 |
| 213x93 and more | version 40, 177 modules | 2316 |

At a code every half second that is 1.7 kB a second on a terminal of 200x60 cells, 100 kB a minute.

The rest is set with options, and with keys while the codes are on show:

| Option | Keys | Default | |
| --- | --- | --- | --- |
| `--module` | `+` `-` | `1` | size of a module: this many columns wide and half as many rows high |
| `--margin` | `}` `{` | `4` | blank space on every side of the code, in modules; four is what the standard asks for |
| `--max-version` | `.` `,` | `40` | the largest version a code may have where there is room for more: version 1 has 21 modules to a side, every further one 4 more |
| `--level` | `e` | `m` | error correction: `l`, `m`, `q` or `h` put right a code of which some 7, 15, 25 or 30 % is misread, and hold less the more they put right |
| `--interval` | `f` `s` | `0.5` | seconds a code stays; `0` leaves the turning to the keys |
| `--no-notes` | `h` | shown | what is on show, the measurements and the keys in the panel |
| `--ink`, `--paper` | | `000000`, `ffffff` | the dark and the light colour as `RRGGBB`; a grey paper dims a screen that glares |

`enter`, `space` and `n` go to the next code, `p` to the one before, `g` and `G` to the first and the last.
`q` quits and prints the options for the layout as it was then, along with what the terminal measured, so
that a setup that read well can be noted down and shown again, and the SHA-256 and the MD5 of the file, to
be compared with those that `qqcode` prints:

```
qqcode-show notes.tar.gz --module 1 --margin 4 --level m --interval 0.2
terminal 200x60 cells, cell 9x18 px, version 23, code 109x54.5 cells = 981x981 px, 842 bytes a code, 4 codes
SHA-256 03172bf1dc18c2c972ff3b2cca2fcf4e800c6dd590474210b53c193aaeaf8bc9
MD5 7850bf91f76b3d7e90d3b7b4eacec425
```

### How large the modules have to be

What counts is how many pixels of the camera's frames a module gets, which `qqcode --preview` shows for
the code in view (`109 modules | 7.9 px a module`). Measured on synthetic frames of 1920x1080, with codes
of the versions 10 to 40 at the levels `l` and `m`, eight frames each; no camera took part:

| Frame | Every code is read from |
| --- | --- |
| sharp, seen straight on | 2 px a module |
| seen at a slight angle, Gaussian blur of 1 px, some noise | 3 px |
| the same with a blur of 1.5 px | 5 px |
| the same with a blur of 2 px | 6 px |

The version made no difference to that and the level hardly any. A camera adds things that these frames do
not have, such as moiré between its pixels and those of the screen, glare, and the distortion of its lens,
which a code of 177 modules forgives less than one of 57. So the first thing to do with a new setup is to
look at the start screen through `--preview`: if its code is outlined in red, or the pixels a module come
to less than 4 or so, press `+` for larger modules or `,` for smaller codes until it is read, and move the
camera nearer if there is room. A frame of 1920x1080 takes `qqcode` about 5 ms to search, so the camera
sets the pace: at 30 frames a second a code that stays for 0.2 s is in six frames, of which the one or two
taken while the screen changes show two codes at once and are read as none.

### What a code says

Every code holds one message, in the byte mode of QR codes:

```
start   "qq" 1 checksum length name...
piece   "qq" 2 checksum length offset bytes...
end     "qq" 3 checksum length
```

The checksum is the CRC-32 of the whole file and names the file together with its length; both are
numbers of 32 bits, most significant byte first, as is the offset, so a file can have 4 GiB less a byte. A
QR code that does not begin like this is none of `qqcode-show` and is passed over, and codes of another
file than the one being read make `qqcode` start over with that file. The SHA-256 and the MD5 that both
programs print are not in the codes.

## Tests

`cargo test` needs neither a camera nor a terminal: unit tests for the pure logic, the tests of
`qqcode-show`, which have its screens drawn as images, photographed at an angle and out of focus and read
back as the file, and `tests/pipeline.rs`, which does the same with single codes and runs the program
`qqcode` on pictures.
