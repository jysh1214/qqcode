# qqcode

Carries a file from one machine to another through a screen and a camera. `qqcode-show` puts the file on
the screen, `qqcode` reads it from the frames of a camera and writes it. Written in Rust; the frames come
through OpenCV.

```sh
qqcode-show notes.tar.gz --html show.html     # where the file is: a page for a browser the camera sees
qqcode > notes.tar.gz                         # where the camera is
```

There are two ways to show a file, and `qqcode` reads either:

| | on show | a frame holds | a second holds |
| --- | --- | --- | --- |
| [web page](#the-web-page), `--html` | a matrix of coloured tiles, frame after frame without end | 10 kB by default, up to some 21 kB for a camera of 1920x1080 | 151 kB at 15 frames |
| [terminal](#qqcode-show-on-a-terminal) | QR codes drawn with half block characters, one after the other | 0.8 kB on a terminal of 200x60 cells, 2.9 kB at most | 1.7 kB at 2 codes |

The terminal needs nothing but a terminal. The page is what to use where a browser is to be had: it is some
ninety times faster, and a frame that the camera misses costs it nothing, as any frames that are enough
make the file.

`qqcode`, per frame:

1. finds the finder patterns in it (those of a QR code, which the matrix has in its corners as well);
2. reads the matrix between four of them, tile by tile, or decodes the QR code (the QR reader of `rxing`);
3. takes what the frame says into the file: a packet of the page, or the piece that a QR code holds;
4. once the file is whole, checks its checksum, writes it and ends.

The frames go from the camera into a buffer and are read by as many threads as the machine has processors,
so the camera never waits for the reading. Nothing in that is a neural network, so neither program has a
use for a GPU.

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
qqcode --threads 4 --buffer 64 > notes.tar.gz          # on a machine with little memory
qqcode --camera-size 1280x720 > notes.tar.gz           # smaller frames
qqcode recording.mp4 > notes.tar.gz                    # a video of the show instead of a camera
qqcode 'frame_%04d.png' > notes.tar.gz                 # or pictures of it
```

| Option | Default | |
| --- | --- | --- |
| input | `0` | camera index, or a video file, an image or an image sequence |
| `-o`, `--output` | the standard output | file to write; a file from a web page is put together on disk as it arrives, see [long files](#long-files) |
| `--camera-size` | `1920x1080` | frame size to ask a camera for, `WIDTHxHEIGHT`; see below |
| `--preview` | off | show what the camera sees on the terminal, with the codes that are found; see below |
| `--threads` | one for each processor but one | how many frames are read at a time |
| `--buffer` | `256` | how many frames may wait to be read; see below |

The file is there when it is whole and its checksum is right, and nothing of it before that: a run that is
stopped, or whose video ends too early, leaves an empty file behind `>` and none at all behind `-o`, and
ends with an error. The file goes to the standard output, so `qqcode` refuses to start with that being the
terminal. What it has to say while it runs goes to the standard error:

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

### Frames that wait

A camera goes on taking frames whatever becomes of them, so reading must not hold it up. One thread does
nothing but take the frames and put them into a buffer; the others take them out and read them, and what
they find is put together in the order of the frames. A frame of the web page takes some 30 ms to read and
one that can't be read up to twice that, so two or three threads keep up with a camera, and the rest of
them are there for the moments in which every frame is a hard one.

If the threads do not keep up, the frames wait, as many as `--buffer` says, and are read later: nothing
is lost by reading slowly but time. Only when that many are waiting are the frames of a camera dropped, and
`qqcode` says at the end how many were. A frame of 1920x1080 takes 6 MB while it waits, so the 256 of the
default come to 1.6 GB at worst. A video file or pictures are never dropped: reading them waits for room.

### Codes that are missed

What follows is about the QR codes of a terminal. A page has no codes to miss: it goes on showing packets,
and `qqcode` ends when it has enough of them.

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

## The web page

```sh
qqcode-show notes.tar.gz --html show.html                       # 176x99 tiles, 15 frames a second
qqcode-show notes.tar.gz --html show.html --scale 16 --rate 12  # more tiles, for a sharp camera
qqcode-show notes.tar.gz --html show.html --shape classic       # for a camera with frames of 4:3
```

It says what it wrote, here for `assets/sample.bin`:

```
wrote show.html: sample.bin, 65536 bytes in 7 frames of 10077 bytes, 15 a second: 151 kB a second, 1 s if every frame is read
matrix of 176x99 tiles, 1592x899 px: best shown at that size or a whole multiple of it, and filling the frames of the camera
SHA-256 d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061
MD5 62b80e7f487ffaae9158a421f8defc52
file:///home/me/qqcode/show.html
```

The last line is the page as a browser is told to open it, with the whole path from the root: most
terminals make a link of it, and it can be pasted into the address bar as it is.

The page is one file that loads nothing else and has the file to show inside it, so it is about a third
larger than that; the page of a file of more than 64 MB [asks for the file](#long-files) instead. It can be
copied to whatever has the screen. A browser shows the matrix as large as its
window allows, on a white ground, and begins on a start screen that says what the file is called and how
long it is. `qqcode` says when it has that in view, which is the moment to press `enter`: the frames
follow, `--rate` of them a second, without end. `qqcode` ends by itself when it has the file.

| Key | |
| --- | --- |
| `enter`, `space` | start, pause, go on |
| `+` `-` | more, fewer frames a second |
| `g` | back to the start screen |
| `f` | the whole screen |
| `h` | hide the notes below the matrix |

| Option | Default | |
| --- | --- | --- |
| `--html` | | file to write the page to |
| `--shape` | `wide` | shape of the matrix: `wide` is 16:9, `classic` 4:3, `square` 1:1; best that of the camera's frames |
| `--scale` | `11` | size of the matrix, 4 to 24: it has 9 rows of tiles for each |
| `--rate` | `15` | frames a second, 1 to 60 |
| `--spare` | `10` | frames in a hundred that the first round shows on top of those that make the file, see [long files](#long-files) |
| `--ask` | for files over 64 MB | write a page that asks for the file when it is opened instead of holding it |

### The matrix

A tile is a square of 8 pixels on a dark ground, 9 pixels from the next one: one of sixteen symbols in one
of four colours, which makes six bits. A symbol is made of sub-cells of 2 pixels, lit or dark, and half of
it is always lit, so that every tile shows its colour. The sixteen are as unlike each other as patterns of
four by four sub-cells can be, and none has a feature finer than a sub-cell. In the four corners of the
matrix sit the finder patterns of a QR code, by which `qqcode` finds the matrix and lays it flat.

The bytes of a frame are spread over the tiles in two planes, the symbols and the colours, each with
Reed-Solomon check bytes of its own: 64 in every 255 bytes of symbols, which puts right a frame in which
one symbol in sixteen is misread, and 40 in every 255 bytes of colours, which does the same for one colour
in fifty. The symbols have more of them: in the measurements below it is the symbols that a blurred or
compressed frame gets wrong, far more often than a colour. `qqcode --preview` says how much of a frame it had to put right.

| `--scale`, shape `wide` | Tiles | Pixels | Bytes of the file in a frame | At 15 frames a second |
| --- | --- | --- | --- | --- |
| 8 | 128x72 | 1160x656 | 5265 | 79 kB a second |
| 11 | 176x99 | 1592x899 | 10077 | 151 kB a second |
| 14 | 224x126 | 2024x1142 | 16329 | 245 kB a second |
| 16 | 256x144 | 2312x1304 | 21465 | 322 kB a second |

### Any frames that are enough

A screen can show a camera more frames than the camera catches, and nothing tells the screen which ones
were lost. So the page does not show the file piece by piece and then over again. It cuts the file into
packets of what a frame holds, shows each once, and after that shows sums of them without end: each the
exclusive or of about half the packets of a block of up to 512, picked by numbers that follow from the
frame's place in the sequence. Whoever has as many frames of a block as the block has packets, whichever
they are, can solve for the packets, give or take a frame or two. A camera that misses a third of the
frames takes half as long again, and never has to wait for a particular one to come round.

### Long files

```sh
qqcode-show disk.img --html show.html      # 10 GB: the page is 24 kB and asks for disk.img
qqcode -o disk.img                         # put together on disk as it arrives
```

Nothing in this keeps a long file in memory, on either side, so a file can be as long as the time there
is for it. 10 GB are a million frames at the default size, and `qqcode-show` says what that comes to:

| 10 GB at 15 frames a second | The first round | Each round after it |
| --- | --- | --- |
| default, `--scale 11` | 20 h 14 min | 1 h 52 min |
| `--scale 16` | 9 h 30 min | 53 min |

The page of a file of more than 64 MB does not hold the file, which no browser would open. It holds what
the file is called, how long it is and its checksum, and asks for the file when it is opened: with the
button, or by dropping the file on the page. It reads the file once to see that it is the one it was
written for, 3 GB in a quarter of a minute, and after that only what it is about to show. `--ask` makes
the page of a short file ask as well.

For that the blocks are not all shown by turns. The page takes them in groups of 16, which are 83 MB of
the file at the default size, shows a group until each of its blocks has had its packets and `--spare`
sums on top, 10 in a hundred, and goes on to the next group. That is the first round, and a camera that
misses fewer frames than it has to spare, one in eleven at the default, has the file when it ends. The
rounds after it are for what the first left unfinished: they show sums alone, never the same one twice,
as many as the first round had to spare and a tenth of the packets at least, so a round takes about a
tenth of the time of the first. They go on without end.

`qqcode -o FILE` puts the file together on disk, in `FILE.part`. A packet of the file itself is written
where it belongs the moment it arrives. A sum goes to a second file, `FILE.part.sums`, and waits there
until its block has as many packets and sums as it takes; then the packets that are missing are solved
for and written into their places. Nothing that the camera reads is thrown away, however many rounds a
file takes, and nothing of it is held in memory but one block while it is solved. The price is disk: the
file once over, and as much again as the camera misses. When the last block is whole, `qqcode` reads the
file once more, for its length and its checksum, and only then gives it its name and removes the sums; if
they are not what the frames said, the file stays `FILE.part` and `qqcode` ends with an error.

Without `-o` the file goes to the standard output in one piece when it is whole, and is kept in memory
until then, which is as before and fine for a file that fits there; `qqcode` says so when it sees that a
file of more than 1 GB is coming.

Measured with a file of 1.5 GB at the default size, whose packets were given to the reader as a camera
would, without the pictures in between; the time is that of the show, the first round being 3 h 2 min:

| Frames lost | The file is whole after | Sums on disk |
| --- | --- | --- |
| none | the first round, less its last sums | none |
| 1 in 20 | the first round | 79 MB |
| 3 in 20 | 2 rounds more: 3 h 35 min | 228 MB |
| 3 in 10 | 4 rounds more: 4 h 9 min | 452 MB |

Putting the file together took the reader 13 seconds of all that at most. With frames lost all along, a
larger `--spare` is the better show: the sums come while their group is on show, and the file is whole
in one round.

### What the camera has to manage

Two things: enough pixels for a tile, and frames that are sharp. Measured on synthetic frames of 1920x1080
with the matrix of 176x99 tiles; no camera took part:

| Blur of the frame | The frames are read down to |
| --- | --- |
| none, 0.7 px, 1 px | 0.7 px of the frame to a pixel of the page: 6.3 px a tile |
| 1.25 px | 0.85 px to a pixel: 7.7 px a tile |
| 1.5 px | 1 px to a pixel: 9 px a tile |

That holds for frames that a camera packs as JPEG of quality 70 with colour at half the sharpness, as
cameras do that send Motion JPEG; at these limits up to one symbol in twenty is misread then, which the
check bytes make up for, and hardly a colour. At quality 50 the frames at the limits are lost.

So a camera of 1920x1080 whose frames the page fills from top to bottom has room for a matrix of some
1080 pixels in height at 1 px to a pixel, which is `--scale 13`, and for `--scale 18` if it is sharp enough
for 0.7. The default leaves a wide margin: it is still read where it takes 1114x629 pixels of such a frame,
or fills one of 1280x720. `qqcode --preview` shows the pixels a tile gets and the share of the frame that
had to be put right; a frame is lost when that share goes beyond a few percent, and the first things to
try then are a smaller `--scale` and a camera nearer to the screen.

The page should show a frame for at least two frames of the camera, so that one of them has it whole:
15 a second for a camera that takes 30. A frame of the camera that is taken while the page changes shows
two frames at once and is read as none, which costs nothing but that frame. What these measurements do
not have is what a real camera adds: moiré between its pixels and those of the screen, glare, colours
that are far from those of the page, and a lens that bends the picture. The reading follows a bend of some
pixels, as it finds the tiles area by area rather than where the corners put them, and it takes the
colours as the frame has them rather than as they ought to be, but how far that goes with a given camera
only that camera can tell.

## `qqcode-show` on a terminal

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

Every QR code holds one message, in the byte mode of QR codes:

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

A frame of the web page is always full, and says:

```
start    "qq" 0x10 checksum length 0     0        length of the name, name, zeros     check
packet   "qq" 0x11 checksum length block sequence bytes of the packet                 check
```

The length is a number of 64 bits here; block and sequence say which packet of which block this is, from
which follows what it is the sum of; and the check is the CRC-32 of the frame, which keeps a frame that
was put right wrongly out of the sums, where it would spoil a whole block.

## Files to try it with

`assets/` has two files to show:

| File | Bytes | What | On the web page | On a terminal of 200x60 cells |
| --- | --- | --- | --- | --- |
| `assets/sample.txt` | 3758 | text to look at once it has arrived: prose, every printable ASCII character, two lines of Chinese, numbered lines | 1 frame | 5 codes |
| `assets/sample.bin` | 65536 | every value of a byte in order, then bytes in no order | 7 frames | 78 codes |

```sh
qqcode-show assets/sample.txt                        # in a terminal that the camera sees
qqcode-show assets/sample.bin --html sample.html     # or as a page for a browser
qqcode -o read.bin --preview                         # where the camera is
```

Both programs end by printing the SHA-256 and the MD5 of the file, which for these two are:

```
SHA-256 bc0327a4d16248b69456fb72aef5959d37387daa499bcc4daea72128f934bae2    assets/sample.txt
MD5 1fbf9430aaaca16ce44e1fe6c180f4bc
SHA-256 d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061    assets/sample.bin
MD5 62b80e7f487ffaae9158a421f8defc52
```

Something longer is quickly made: `head -c 1M /dev/urandom > big.bin` is 105 frames of the page, 7 seconds
at 15 a second.

## Tests

`cargo test` needs neither a camera nor a terminal nor a browser: unit tests for the pure logic, the tests
of `qqcode-show`, which have its screens and the frames of its page drawn as images, photographed at an
angle and out of focus and read back as the file, and `tests/pipeline.rs`, which does the same with single
codes and frames and runs the program `qqcode` on pictures.

The page computes in JavaScript what the library computes in Rust, and one test has both make the tiles
of the same frames and compares them. It runs the script of the page in `deno`; without `deno` on `PATH`
it says so and passes.
