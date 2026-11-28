# qqcode

Carries a file from one machine to another through a screen and a camera. `qqcode-show` writes a web page
that puts the file on the screen, `qqcode` reads it from the frames of a camera and writes it. Written in
Rust; the frames come through OpenCV.

```sh
qqcode-show notes.tar.gz --html show.html     # where the file is: a page for a browser the camera sees
qqcode > notes.tar.gz                         # where the camera is
```

The [page](#the-web-page) shows a matrix of coloured tiles, frame after frame without end. A frame holds
10 kB of the file by default and up to some 21 kB for a camera of 1920x1080, which is 151 kB a second at
15 frames. A frame that the camera misses costs nothing, as any frames that are enough make the file.

`qqcode`, per frame:

1. finds the finder patterns in it, which the matrix has in its four corners (they are those of a QR code,
   and found by the QR reader of `rxing`);
2. reads the matrix between them, tile by tile;
3. takes the packet that the frame holds into the file;
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
| `-o`, `--output` | the standard output | file to write; it is put together on disk as it arrives, see [long files](#long-files), and a run that is stopped is [gone on with](#going-on-with-a-file) by the next |
| `--check` | | the checksums of the file that is to arrive, which it is held against; see [checksums ahead of the file](#checksums-ahead-of-the-file) |
| `--camera-size` | `1920x1080` | frame size to ask a camera for, `WIDTHxHEIGHT`; see below |
| `--preview` | off | show what the camera sees on the terminal, with the codes that are found; see below |
| `--threads` | one for each processor but one | how many frames are read at a time |
| `--buffer` | `256` | how many frames may wait to be read; see below |

The file is there when it is whole and its checksum is right, and nothing of it before that: a run that is
stopped, or whose video ends too early, leaves an empty file behind `>` and none at all behind `-o`, and
ends with an error. What had arrived of the file by then is kept next to where `-o` puts it, and the same
command [goes on with it](#going-on-with-a-file). The file goes to the standard output, so `qqcode` refuses
to start with that being the terminal. What it has to say while it runs goes to the standard error, with
a line a second for how far the file has come; here for `assets/sample.bin`:

```
camera 0: 1920x1080 frames, 30 a second
frame 31: start screen in view: sample.bin, 65536 bytes, waiting for the first code
frame 58: 10077 of 65536 bytes, 1 of 7 packets
frame 70: the file is whole
70 frames read, 65536 bytes written to the standard output
SHA-256 d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061
MD5 62b80e7f487ffaae9158a421f8defc52
```

The last two lines are the SHA-256 and the MD5 of the file as it was written. `qqcode-show` prints those
of the file it shows when it writes the page, and `sha256sum` and `md5sum` say the same of either file:
they are there to be compared, by whoever wants more than the checksum in the frames, which `qqcode` has
checked by then.

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

### Cameras

A camera delivers the size it starts up with, which is 640x480 for nearly all of them and too little for
a matrix of any size, so `qqcode` asks for 1920x1080 unless `--camera-size` names another size. It says which
frames it gets when it starts; if the camera does not have the size asked for, those are of the nearest one
it has, and `v4l2-ctl --list-formats-ext` lists them all. USB cameras tend to have their large sizes as
Motion JPEG only, or uncompressed at a few frames a second, so `qqcode` switches to Motion JPEG where that
comes nearer to the size or brings more frames a second: a page that shows fifteen frames a second is not
read at five.

To point the camera, `--preview` draws what it sees on the terminal that `qqcode` runs in, a frame at a
time, as coarse as the terminal has character cells (a smaller font gives a finer picture). In the picture
the matrix of `qqcode-show` is outlined in green where it is read; one that can't be read has no outline.
The lines below it say what is in view, how many tiles the matrix has across, how many pixels of the frame
a tile gets and how much of the frame had to be put right, what to do about it, how fast the file arrives
and how much of it has.

How fast is `27.4 kB a second`, over the last five seconds: the number to try another `--scale`, another
`--rate` or another place for the camera by. A camera takes one frame in every few while the page changes,
which is read as none and costs nothing, so the preview goes on showing what was read before it, and says
that a code can't be read once none was for half a second.

Reading goes on as without the preview; what `qqcode` would otherwise write to the terminal while it runs
is printed when it ends. So is what the libraries under it write there, such as the `Corrupt JPEG data` that
some cameras get for every frame, with the lines that differ in their numbers only told once. `q` ends it.
The preview is drawn through the standard error, which leaves the standard output to the file.

## The web page

```sh
qqcode-show notes.tar.gz --html show.html                       # 176x99 tiles, 15 frames a second
qqcode-show notes.tar.gz --html show.html --scale 16 --rate 12  # more tiles, for a sharp camera
qqcode-show notes.tar.gz --html show.html --shape classic       # for a camera with frames of 4:3
```

It says what it wrote, here for `assets/sample.bin`:

```
wrote show.html: sample.bin, 65536 bytes in 7 frames of 10077 bytes, 15 a second: 151 kB a second, 1 s if every frame is read
the first round is 8 frames, 1 s, with 10 sums to spare in a hundred; the rounds after it are sums alone, 1 frame, 1 s each; the file is in the page
matrix of 176x99 tiles, 1592x899 px: best shown at that size or a whole multiple of it, and filling the frames of the camera
wrote sample.bin.checksums.txt: the checksums of the file, 1 section of 1048576 bytes at most, for qqcode --check; the SHA-256 of this list is 2986d6267f51f6bd52e624863e0ca3fa79bb9e0e336212cbb415767800d6f39b
SHA-256 d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061
MD5 62b80e7f487ffaae9158a421f8defc52
file:///home/me/qqcode/show.html
```

The last line is the page as a browser is told to open it, with the whole path from the root: most
terminals make a link of it, and it can be pasted into the address bar as it is. The line before the
SHA-256 and the MD5 of the file is about a second file that is written next to the page: the
[checksums of the file](#checksums-ahead-of-the-file) megabyte by megabyte, for `qqcode --check`.

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
| `--html` | | file to write the page to, which has to be named; the [checksums of the file](#checksums-ahead-of-the-file) go next to it, as `NAME.checksums.txt` |
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
they are not what the frames said, the file stays `FILE.part` and `qqcode` ends with an error. A run that
ends before the file is whole leaves both where they are, see [going on with a file](#going-on-with-a-file).

Without `-o` the file goes to the standard output in one piece when it is whole, and is kept in memory
until then, which is fine for a file that fits there; `qqcode` says so when it sees that a file of more
than 1 GB is coming. Should all of it arrive and its checksum be wrong, a frame was misread in a way that
its check did not notice: `qqcode` says so, drops what it has and collects the file anew.

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

### Going on with a file

A show of hours gets stopped: by `q` in the preview, by Ctrl-C, by a camera that is pulled out. With
`-o FILE` nothing of what had arrived is lost by that. `FILE.part` and `FILE.part.sums` stay where they
are, and a third file next to them, `FILE.part.index`, says which packets are in the two. The next run
with the same `-o FILE` finds the three, goes on from there, and says so:

```
frame 2: going on with notes.tar.gz.part, which has 96688815 of the 203541436 bytes
```

`qqcode` writes the index when it stops, and every five seconds while packets arrive, each time after
the two files that it is of are on disk. So a run that is cut off with no time to write anything, by
Ctrl-C without `--preview` or by the power, has read its last five seconds at most for nothing, and the
index that it leaves never names a packet that is not there. The index of a file of 200 MB has 13 kB.

The run that goes on needs no start screen, as every frame says what file it is of, and takes the page
as it finds it:

- A page that was paused with `space` goes on where it was, and nothing is missing.
- A page that went on meanwhile has shown frames to nobody. They are made up for as frames that a camera
  misses are: by the sums that the first round has to spare, and after it by the short rounds, which
  bring a tenth of a block each. That is soon done for a minute that went by and slow for an hour, so
  the page is better paused.
- A page that begins again, with `g` or opened anew, shows the file from its first packet. `qqcode` has
  no use for the packets that it has, and the page can't be told to leave them out: the others come
  when their turn comes, which is no sooner than without what was kept.

The file has to be shown as it was begun: the same file, with the same matrix. `--rate` and `--spare`
may be other ones, since they change which frames come when and not what is in them. `--scale` and
`--shape` may not: a frame of another size holds a packet of another size, and what arrived is no part
of those. `qqcode` does not throw away what took hours to read for a page that shows something else, and
ends with an error at the first frame of it:

```
Error: frame 2: notes.tar.gz.part has what arrived of this file in packets of 10077 bytes, and the page shows packets of 5265 bytes now. The file is gone on with by the matrix that it was begun with; for another one, notes.tar.gz.part is to be removed first, and the file begins anew
```

`rm notes.tar.gz.part*` is that new beginning.

With `--check`, a run that goes on holds every section that was whole against its checksum again before
it takes a packet in, which is a fraction of a second for 200 MB: what it builds on is what the list
says, whatever the index says. Without the list, a file that is whole and has not the checksum that its
frames name is not gone on with. Nothing says which of its packets to read again, so `qqcode` removes
its index, and the next run begins the file anew.

### Checksums ahead of the file

```sh
qqcode-show disk.img --html show.html                # writes disk.img.checksums.txt next to show.html
qqcode -o disk.img --check disk.img.checksums.txt    # where the camera is, with that list at hand
```

A frame that the camera misreads is thrown away by its check, which lets a wrong one through once in four
thousand million. That is rare, and still not what a file of hours should hang on: one wrong packet makes
a wrong block, and nothing shows it before all of the file is there and its checksum is wrong, with no
telling where.

So `qqcode-show --html` writes the checksums of the file next to the page: the SHA-256 of every megabyte
of it, and of the whole. The megabytes are those of the file, whatever the size of the matrix, so one list
goes with every page of the file. This is the list of `assets/sample.bin`, which has less than a megabyte:

```
qqcode checksums 1
file sample.bin
length 65536
crc32 4042d80d
sha256 d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061
section 1048576
d060b80065d3d79686baaf6c41801d46ff93365dd414bd30501029ce5838f061  0
```

Every line after `section` is the SHA-256 of a megabyte and where in the file it begins. The list of a
file of 10 GB has 9537 of them and is 733 kB.

The list has to be where the camera is before the file, and it is worth what it is right. It is a file
like any other, so it can go the same way first, which takes seconds (`qqcode-show disk.img.checksums.txt
--html list.html`, `qqcode > disk.img.checksums.txt`). Both programs print the SHA-256 of the list,
`qqcode-show` when it writes it and `qqcode --check` when it reads it: comparing those two once stands for
every megabyte of the file.

With `--check`, `qqcode`

- takes no other file than the one that the list names by its length and its CRC-32, and ends with an
  error at the first frame of another one;
- with `-o`, holds every megabyte against its checksum as soon as the blocks it lies in are whole, and
  says how many have theirs as it goes;
- finds a packet that was misread. A block that is whole goes on taking sums, 16 at most, until its
  megabytes are known to be right. Each of those sums is a test of the block as it was solved; the tests
  that fail all miss by the same bytes, and which of them fail is the mark of one packet. That packet is
  left out and the block made of the others. If no one packet accounts for a megabyte that is wrong, more
  than one was misread: what there is of its blocks is dropped, and they are read anew from the sums of
  the rounds that follow;
- without `-o`, where the file is kept in memory, holds it against the list when it is whole, and starts
  over if a megabyte is wrong;
- compares the SHA-256 of the whole file with that of the list before it gives the file its name.

Measured with the file of 1.5 GB as above, three frames in ten lost: the checksums took a second on top of
the 14 that putting the file together takes, and 20 MB of sums. With one packet in 2000 misread on
purpose, 157 of them and far more than a camera could manage, 62 were found and left out, 9 times a block
had more than one and was read anew, and the file was the right one, after 19 rounds of sums where 4 had
done.

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

### What a frame says

What the tiles of a frame hold, once they are read and put right. A frame is always full, and says:

```
start    "qq" 0x10 checksum length 0     0        length of the name, name, zeros     check
packet   "qq" 0x11 checksum length block sequence bytes of the packet                 check
```

The checksum is the CRC-32 of the whole file and names the file together with its length, which is a
number of 64 bits; block, sequence and the check are of 32 bits, the length of the name is of 16, and all
of them have their most significant byte first. Block and sequence say which packet of which block this
is, from which follows what it is the sum of; and the check is the CRC-32 of the frame, which keeps a frame
that was put right wrongly out of the sums, where it would spoil a whole block. Frames of another file
than the one being read make `qqcode` start over with that file. The SHA-256 and the MD5 that both
programs print are not in the frames.

## Files to try it with

`assets/` has two files to show:

| File | Bytes | What | On the page |
| --- | --- | --- | --- |
| `assets/sample.txt` | 3758 | text to look at once it has arrived: prose, every printable ASCII character, two lines of Chinese, numbered lines | 1 frame |
| `assets/sample.bin` | 65536 | every value of a byte in order, then bytes in no order | 7 frames |

```sh
qqcode-show assets/sample.bin --html sample.html     # a page for a browser that the camera sees
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

`cargo test` needs neither a camera nor a browser: unit tests for the pure logic, the tests of
`qqcode-show`, which have the frames of its page drawn as images, photographed at an angle and out of focus
and read back as the file, and `tests/pipeline.rs`, which does the same with single frames and runs the
program `qqcode` on pictures.

The page computes in JavaScript what the library computes in Rust, and one test has both make the tiles
of the same frames and compares them. It runs the script of the page in `deno`; without `deno` on `PATH`
it says so and passes.
