//! The show as a web page: one file that loads nothing else, which a browser puts on the screen as a matrix
//! of tiles ([`qqcode::tiles`]), frame after frame.
//!
//! A page holds the file it shows if that is short. A long file would make a page that no browser opens,
//! so the page of a long file holds only what it is called, how long it is and its checksum, and asks for
//! the file when it is opened; it then reads of the file what it is about to show and no more.
//!
//! The page does in JavaScript what the library does in Rust: it cuts the file into packets, puts each
//! into a frame and draws the tiles. What is here writes the page and, for the tests, says what the page
//! has to show.

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use qqcode::checks::{self, Checks};
use qqcode::crc::Checksum;
use qqcode::digest::{self, Digests};
use qqcode::fountain::{self, Plan, Schedule};
use qqcode::packet::{OVERHEAD, Origin};
use qqcode::tiles::{self, Grid};
#[cfg(test)]
use qqcode::{fountain::Encoder, packet::Frame};

const TEMPLATE: &str = include_str!("page.html");

/// A page holds a file of up to this many bytes; a longer one it asks for.
pub const HELD: u64 = 64 << 20;

/// How a page shows its file, see the command line options of the same names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub grid: Grid,
    /// Frames a second
    pub rate: u32,
    /// Sums that a round shows on top of the packets of a block, in a hundred
    pub spare: usize,
}

/// A file as the page shows it, for the tests: the library's own account of what the JavaScript of the
/// page does.
#[cfg(test)]
pub struct Show<'a> {
    pub grid: Grid,
    pub origin: Origin,
    name: &'a str,
    encoder: Encoder<'a>,
    schedule: Schedule,
}

#[cfg(test)]
impl<'a> Show<'a> {
    pub fn new(name: &'a str, bytes: &'a [u8], format: Format) -> Self {
        let origin = Origin::of(bytes);
        let encoder = Encoder::new(bytes, origin.checksum, format.grid.capacity() - OVERHEAD);
        Self {
            grid: format.grid,
            origin,
            name,
            schedule: Schedule::new(encoder.plan().clone(), format.spare),
            encoder,
        }
    }

    /// How many frames it takes at least to make the file: one for each symbol.
    pub fn frames(&self) -> usize {
        self.encoder.plan().symbols()
    }

    /// What the start screen says, and after it the frame `frame` of the file, counted from 0 without
    /// end.
    pub fn frame(&self, frame: Option<u64>) -> Frame {
        match frame {
            None => Frame::Start {
                origin: self.origin,
                name: self.name.to_owned(),
            },
            Some(frame) => {
                let (block, sequence) = self.schedule.turn(frame);
                Frame::Packet {
                    origin: self.origin,
                    block,
                    sequence,
                    bytes: self.encoder.packet(block, sequence),
                }
            }
        }
    }

    /// The tiles of that frame, as [`tiles::pack`] has them.
    pub fn tiles(&self, frame: Option<u64>) -> Vec<u8> {
        tiles::pack(self.grid, &self.frame(frame).encode(self.grid.capacity()))
    }
}

/// `text` between the quotes of a JSON string that stands in a page: nothing in it ends the string or the
/// script it is in.
fn json(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '"' | '\\' => {
                escaped.push('\\');
                escaped.push(c);
            }
            '<' | '>' | '&' | '\u{2028}' | '\u{2029}' => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped
}

fn html(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn base64(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let bits = (0..3).fold(0u32, |bits, at| bits << 8 | *group.get(at).unwrap_or(&0) as u32);
        for digit in 0..4 {
            if digit <= group.len() {
                text.push(DIGITS[(bits >> (18 - 6 * digit) & 63) as usize] as char);
            } else {
                text.push('=');
            }
        }
    }
    text
}

/// The page that shows the file `origin`, which is called `name`. `held` are the bytes of the file if the
/// page is to hold them; a page without them asks for the file when it is opened.
pub fn page(name: &str, origin: Origin, held: Option<&[u8]>, format: Format) -> String {
    let grid = format.grid;
    let colours = tiles::COLOURS.map(|[red, green, blue]| format!("[{red},{green},{blue}]"));
    let meta = format!(
        "{{\"name\":\"{}\",\"checksum\":{},\"length\":{},\"embedded\":{},\"pace\":{},\"plan\":{{\"block\":{},\
         \"group\":{},\"spare\":{},\"again\":{}}},\"colours\":[{}],\"matrix\":{{\"columns\":{},\"rows\":{},\"pitch\":{},\
         \"sub\":{},\"border\":{},\"anchor\":{},\"module\":{},\"bits\":{},\"data\":[{},{}]}}}}",
        json(name),
        origin.checksum,
        origin.length,
        held.is_some(),
        format.rate,
        fountain::BLOCK,
        fountain::GROUP,
        format.spare,
        fountain::AGAIN,
        colours.join(","),
        grid.columns,
        grid.rows,
        tiles::PITCH,
        tiles::SUB,
        tiles::BORDER,
        tiles::ANCHOR,
        tiles::MODULE,
        tiles::BITS,
        tiles::DATA.0,
        tiles::DATA.1,
    );
    TEMPLATE
        .replace("{{TITLE}}", &html(name))
        .replace("{{META}}", &meta)
        .replace("{{FILE}}", &held.map(base64).unwrap_or_default())
}

/// What a file is, read from beginning to end without keeping it: its MD5, and its checksums section by
/// section ([`qqcode::checks`]), which have its length, its CRC-32 and its SHA-256 as well. `name` is
/// what they call it.
fn identify(file: &Path, name: &str) -> Result<(Checks, String)> {
    let unreadable = || format!("can't read {}", file.display());
    let mut source = File::open(file).with_context(unreadable)?;
    let (mut checksum, mut digests, mut length) = (Checksum::new(), Digests::default(), 0u64);
    let mut sections = checks::Maker::new(checks::SECTION);
    let mut piece = vec![0u8; 1 << 20];
    loop {
        let read = source.read(&mut piece).with_context(unreadable)?;
        if read == 0 {
            break;
        }
        checksum.update(&piece[..read]);
        digests.update(&piece[..read]);
        sections.update(&piece[..read]);
        length += read as u64;
    }
    let origin = Origin {
        checksum: checksum.finish(),
        length,
    };
    let (sha256, md5) = digests.finish();
    Ok((sections.finish(name, origin, &sha256), md5))
}

/// Where the checksums of the file that a page shows are written: next to the page, under the name of
/// the file.
pub fn list_of(page: &Path, name: &str) -> PathBuf {
    page.with_file_name(format!("{name}.checksums.txt"))
}

/// Writes the page for `file` to `to`, and next to it the checksums of the file for `qqcode --check`
/// ([`list_of`]), and says what is in them. The page holds the file unless that is longer than [`HELD`]
/// or the page is to `ask` for it.
pub fn write(file: &Path, to: &Path, format: Format, ask: bool) -> Result<()> {
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    let (checks, md5) = identify(file, &name)?;
    let origin = checks.origin;
    let held = if ask || origin.length > HELD {
        None
    } else {
        Some(fs::read(file).with_context(|| format!("can't read {}", file.display()))?)
    };
    fs::write(to, page(&name, origin, held.as_deref(), format))
        .with_context(|| format!("can't write {}", to.display()))?;
    println!("{}", summary(to, &name, origin, held.is_some(), format));
    let (list, text) = (list_of(to, &name), checks.to_string());
    fs::write(&list, &text).with_context(|| format!("can't write {}", list.display()))?;
    // what qqcode --check says of the list it is given has to be this, or the list is not the same
    println!(
        "wrote {}: the checksums of the file, {} of {} bytes at most, for qqcode --check; the SHA-256 of \
         this list is {}",
        list.display(),
        crate::count(checks.sections(), "section"),
        checks.section,
        digest::sha256(text.as_bytes())
    );
    println!("SHA-256 {}", checks.sha256);
    println!("MD5 {md5}");
    // last, and on a line of its own: what a browser is told to open, and what a terminal makes a link of
    let whole = fs::canonicalize(to).or_else(|_| std::path::absolute(to));
    println!("{}", address(&whole.unwrap_or_else(|_| to.to_owned())));
    Ok(())
}

/// The page at `path`, which begins at the root, as a browser is told to open it: `file://` and the path,
/// with every byte that an address has another use for written as a percent sign and two digits.
fn address(path: &Path) -> String {
    let mut address = String::from("file://");
    for byte in path.as_os_str().as_encoded_bytes() {
        if byte.is_ascii_alphanumeric() || b"/-._~".contains(byte) {
            address.push(*byte as char);
        } else {
            address.push_str(&format!("%{byte:02X}"));
        }
    }
    address
}

/// `2 h 5 min` from an hour on, `3 min 20 s` from two minutes on, `45 s`.
fn span(seconds: u64) -> String {
    match seconds {
        0..120 => format!("{seconds} s"),
        120..3600 => format!("{} min {} s", seconds / 60, seconds % 60),
        _ => format!("{} h {} min", seconds / 3600, seconds % 3600 / 60),
    }
}

/// What a page that was written shows, and what it asks of the screen and of the camera.
fn summary(to: &Path, name: &str, origin: Origin, held: bool, format: Format) -> String {
    let grid = format.grid;
    let packet = grid.capacity() - OVERHEAD;
    let schedule = Schedule::new(Plan::new(origin.length, packet), format.spare);
    let rate = format.rate as u64;
    let time = |frames: u64| span(frames.div_ceil(rate));
    let file = if held {
        "the file is in the page".to_owned()
    } else {
        let group = (fountain::GROUP * fountain::BLOCK * packet) as u64;
        format!(
            "the page asks for the file when it is opened, and reads it {} MB at a time",
            group.min(origin.length).div_ceil(1_000_000)
        )
    };
    format!(
        "wrote {}: {name}, {} bytes in {} frames of {packet} bytes, {rate} a second: {} kB a second, {} if \
         every frame is read\n\
         the first round is {} frames, {}, with {} sums to spare in a hundred; the rounds after it are sums \
         alone, {}, {} each; {file}\n\
         matrix of {}x{} tiles, {}x{} px: best shown at that size or a whole multiple of it, and filling the \
         frames of the camera",
        to.display(),
        origin.length,
        schedule.plan().symbols(),
        packet as u64 * rate / 1000,
        time(schedule.least()),
        schedule.round(),
        time(schedule.round()),
        format.spare,
        crate::count(schedule.again() as usize, "frame"),
        time(schedule.again()),
        grid.columns,
        grid.rows,
        grid.width(),
        grid.height(),
    )
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use qqcode::tiles::Shape;

    use super::*;

    fn bytes(count: usize) -> Vec<u8> {
        let mut state = 0x1234_5678u32;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    fn format(shape: Shape, scale: usize) -> Format {
        Format {
            grid: Grid::new(shape, scale),
            rate: 15,
            spare: 25,
        }
    }

    #[test]
    fn base64_is_the_usual_one() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(&[0xfb, 0xff, 0xfe]), "+//+");
    }

    #[test]
    fn name_does_not_end_the_script_it_stands_in() {
        assert_eq!(json(r#"a "b" \c"#), r#"a \"b\" \\c"#);
        assert_eq!(json("</script>\n"), "\\u003c/script\\u003e\\u000a");
        assert_eq!(html("<a&b>"), "&lt;a&amp;b&gt;");
        let page = page(
            "</script><b>",
            Origin::of(b"abc"),
            Some(b"abc"),
            format(Shape::Square, 4),
        );
        assert!(page.contains("<title>&lt;/script&gt;&lt;b&gt;</title>"));
        assert!(page.contains("\"name\":\"\\u003c/script\\u003e\\u003cb\\u003e\""));
        // the two scripts of data and the one of the program, each ended once
        assert_eq!(page.matches("</script>").count(), 3);
    }

    #[test]
    fn page_holds_the_file_and_what_it_takes_to_show_it() {
        let file = bytes(1000);
        let origin = Origin::of(&file);
        let wide = Format {
            rate: 20,
            ..format(Shape::Wide, 11)
        };
        let held = page("notes.bin", origin, Some(&file), wide);
        assert!(!held.contains("{{"), "nothing left to fill in");
        assert!(held.contains(&format!(">{}</script>", base64(&file))));
        assert!(held.contains(&format!(
            "{{\"name\":\"notes.bin\",\"checksum\":{},\"length\":1000,\"embedded\":true,\"pace\":20,\"plan\":{{\"block\":512,\
             \"group\":16,\"spare\":25,\"again\":10}},\"colours\":[[255,255,255],[0,255,255],[255,0,255],[255,255,0]],\
             \"matrix\":{{\"columns\":176,\"rows\":99,\"pitch\":9,\"sub\":2,\"border\":4,\"anchor\":6,\"module\":6,\
             \"bits\":6,\"data\":[3,4]}}}}",
            origin.checksum
        )));
        // a page that asks for its file has all of that but the file
        let asking = page("notes.bin", origin, None, wide);
        assert!(asking.contains("\"length\":1000,\"embedded\":false,"));
        assert!(asking.contains("<script id=\"file\" type=\"application/octet-stream\"></script>"));
        // "false" for "true"
        assert_eq!(asking.len(), held.len() - base64(&file).len() + 1);
    }

    #[test]
    fn show_is_the_start_screen_and_then_the_packets_by_turns() {
        let file = bytes(3000);
        let square = format(Shape::Square, 4);
        // 648 bytes a frame, 27 of which are not the packet
        assert_eq!(square.grid.capacity(), 648);
        let show = Show::new("notes.bin", &file, square);
        assert_eq!(show.frames(), 5);
        assert_eq!(
            show.frame(None),
            Frame::Start {
                origin: Origin::of(&file),
                name: "notes.bin".to_owned()
            }
        );
        let Frame::Packet {
            block,
            sequence,
            bytes,
            ..
        } = show.frame(Some(2))
        else {
            panic!("a packet");
        };
        assert_eq!((block, sequence, &bytes[..]), (0, 2, &file[1242..1863]));
        assert!(matches!(
            show.frame(Some(1000)),
            Frame::Packet { sequence: 1000, .. }
        ));
        assert_eq!(show.tiles(Some(5)).len(), square.grid.tiles());
    }

    #[test]
    fn page_is_named_as_a_browser_opens_it() {
        assert_eq!(
            address(Path::new("/home/me/show.html")),
            "file:///home/me/show.html"
        );
        // a space, a number sign, a question mark and a letter of two bytes
        assert_eq!(
            address(Path::new("/tmp/my files/a#b?ü.html")),
            "file:///tmp/my%20files/a%23b%3F%C3%BC.html"
        );
        assert_eq!(address(Path::new("/a-b_c.d~e/100%")), "file:///a-b_c.d~e/100%25");
    }

    #[test]
    fn writing_a_page_leaves_it_where_it_was_asked_for() {
        let dir = std::env::temp_dir().join(format!("qqcode-show-write-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (file, to) = (dir.join("notes.bin"), dir.join("show.html"));
        fs::write(&file, bytes(1000)).unwrap();
        let square = format(Shape::Square, 4);
        let origin = Origin::of(&bytes(1000));
        // what a file is comes out the same whether it is read at once or piece by piece
        let (identified, md5) = identify(&file, "notes.bin").unwrap();
        assert_eq!(identified, Checks::of("notes.bin", &bytes(1000)));
        assert_eq!(identified.origin, origin);
        assert_eq!(identified.sha256, qqcode::digest::sha256(&bytes(1000)));
        assert_eq!(md5, qqcode::digest::md5(&bytes(1000)));

        write(&file, &to, square, false).unwrap();
        let written = fs::read_to_string(&to).unwrap();
        assert_eq!(written, page("notes.bin", origin, Some(&bytes(1000)), square));
        // next to the page are the checksums of the file, under its name
        let list = dir.join("notes.bin.checksums.txt");
        assert_eq!(list_of(&to, "notes.bin"), list);
        assert_eq!(
            Checks::parse(&fs::read_to_string(&list).unwrap()),
            Ok(Checks::of("notes.bin", &bytes(1000)))
        );
        // and the address that is printed is that of the file, from the root
        let address = address(&fs::canonicalize(&to).unwrap());
        assert!(
            address.starts_with("file:///") && address.ends_with("/show.html"),
            "{address}"
        );
        // a page that is to ask for its file is written without it
        write(&file, &to, square, true).unwrap();
        assert_eq!(
            fs::read_to_string(&to).unwrap(),
            page("notes.bin", origin, None, square)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn summary_says_what_the_page_takes() {
        let origin = Origin {
            checksum: 1,
            length: 3_000_000,
        };
        let text = summary(
            Path::new("show.html"),
            "notes.bin",
            origin,
            true,
            format(Shape::Wide, 11),
        );
        assert_eq!(
            text,
            "wrote show.html: notes.bin, 3000000 bytes in 310 frames of 9693 bytes, 15 a second: 145 kB a second, \
             21 s if every frame is read\n\
             the first round is 388 frames, 26 s, with 25 sums to spare in a hundred; the rounds after it are \
             sums alone, 78 frames, 6 s each; the file is in the page\n\
             matrix of 176x99 tiles, 1592x899 px: best shown at that size or a whole multiple of it, and filling \
             the frames of the camera"
        );
        // ten thousand million bytes: 2015 blocks in 126 groups, of which the page reads one at a time
        let long = Origin {
            checksum: 1,
            length: 10_000_000_000,
        };
        let text = summary(
            Path::new("show.html"),
            "disk.img",
            long,
            false,
            format(Shape::Wide, 11),
        );
        assert!(
            text.starts_with(
                "wrote show.html: disk.img, 10000000000 bytes in 1031673 frames of 9693 bytes, 15 a second: 145 kB a \
                 second, 23 h 50 min if every frame is read\n\
                 the first round is 1289600 frames, 23 h 52 min, with 25 sums to spare in a hundred; the rounds \
                 after it are sums alone, 257920 frames, 4 h 46 min each; the page asks for the file when it is \
                 opened, and reads it 80 MB at a time\n"
            ),
            "{text}"
        );
        assert_eq!(
            (
                span(45),
                span(119),
                span(200),
                span(3599),
                span(3600),
                span(6722),
                span(7260)
            ),
            (
                "45 s".into(),
                "119 s".into(),
                "3 min 20 s".into(),
                "59 min 59 s".into(),
                "1 h 0 min".into(),
                "1 h 52 min".into(),
                "2 h 1 min".into()
            )
        );
    }

    /// What the JavaScript of the page makes of a file, next to what the library makes of it: the tiles of
    /// the start screen and of some frames, from a page that holds the file and from one that is given
    /// it. It takes a JavaScript engine to find out, and the test says so and passes if there is none.
    #[test]
    fn page_computes_the_tiles_that_the_library_does() {
        // 18 blocks, so two groups, the last symbol short, and a name that is cut in the middle of a
        // character
        let file = bytes(5_495_352);
        let square = format(Shape::Square, 4);
        let name = format!("{}ü.bin", "n".repeat(618));
        let show = Show::new(&name, &file, square);
        assert_eq!(show.frames(), 8850);
        // the first group and its sums, the second group, the end of a round and the rounds after it
        let frames = [
            0u64, 1, 15, 16, 7872, 7873, 9839, 9840, 9841, 10_000, 11_067, 11_068, 11_069, 13_035, 13_036,
            13_281, 13_282, 500_000,
        ];
        let driver = r#"
            const [page, bytes, ...frames] = Deno.args;
            const text = await Deno.readTextFile(page);
            const script = text.split("<script>")[1].split("</script>")[0];
            const meta = JSON.parse(text.split('<script id="meta" type="application/json">')[1].split("</script>")[0]);
            const held_text = text.split('<script id="file" type="application/octet-stream">')[1].split("</script>")[0];
            const { Show, fileBytes, held, fetched, fileChecksum } =
              new Function(script + "; return { Show, fileBytes, held, fetched, fileChecksum };")();
            let read;
            if (meta.embedded) {
              read = held(fileBytes(held_text));
            } else {
              // as the page does with the file it is given
              const file = new File([await Deno.readFile(bytes)], "given");
              if (file.size !== meta.length || await fileChecksum(file, () => {}) !== meta.checksum) throw new Error("another file");
              read = fetched(file);
            }
            const show = new Show(meta, read);
            const hex = (tiles) => Array.from(tiles, (tile) => tile.toString(16).padStart(2, "0")).join("");
            console.log(hex(show.tiles(null)));
            for (const frame of frames.map(Number)) {
              await show.prepare(frame, 75);
              console.log(hex(show.tiles(frame)));
              if (show.fountain.held.size > 2) throw new Error("more than two groups of the file are held");
            }
        "#;
        let dir = std::env::temp_dir().join(format!("qqcode-show-page-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("driver.js"), driver).unwrap();
        fs::write(dir.join("file.bin"), &file).unwrap();
        let origin = Origin::of(&file);
        let hex = |tiles: Vec<u8>| tiles.iter().map(|tile| format!("{tile:02x}")).collect::<String>();
        for held in [Some(&file[..]), None] {
            fs::write(dir.join("page.html"), page(&name, origin, held, square)).unwrap();
            let run = Command::new("deno")
                .arg("run")
                .arg("--allow-read")
                .arg(dir.join("driver.js"))
                .arg(dir.join("page.html"))
                .arg(dir.join("file.bin"))
                .args(frames.map(|frame| frame.to_string()))
                .output();
            let Ok(run) = run else {
                eprintln!("deno is not installed: what the page computes is not compared with the library");
                break;
            };
            assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
            let printed = String::from_utf8(run.stdout).unwrap();
            let mut lines = printed.lines();
            assert_eq!(
                lines.next(),
                Some(hex(show.tiles(None)).as_str()),
                "the start screen"
            );
            for frame in frames {
                assert_eq!(
                    lines.next(),
                    Some(hex(show.tiles(Some(frame))).as_str()),
                    "frame {frame}"
                );
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Where `+` and `-` take a page from the pace it has: a frame a second further, and no further than
    /// `--rate` goes. Like the tiles, it takes a JavaScript engine to find out.
    #[test]
    fn page_goes_faster_and_slower_by_a_frame_a_second() {
        let driver = r#"
            const text = await Deno.readTextFile(Deno.args[0]);
            const script = text.split("<script>")[1].split("</script>")[0];
            const { faster, slower } = new Function(script + "; return { faster, slower };")();
            for (const pace of Deno.args.slice(1).map(Number)) console.log(slower(pace), pace, faster(pace));
        "#;
        let dir = std::env::temp_dir().join(format!("qqcode-show-paces-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("driver.js"), driver).unwrap();
        fs::write(dir.join("page.html"), TEMPLATE).unwrap();
        let run = Command::new("deno")
            .arg("run")
            .arg("--allow-read")
            .arg(dir.join("driver.js"))
            .arg(dir.join("page.html"))
            .args(["1", "15", "25", "59", "60"])
            .output();
        let _ = fs::remove_dir_all(&dir);
        let Ok(run) = run else {
            eprintln!("deno is not installed: where the keys of the page take its pace is not tried");
            return;
        };
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        assert_eq!(
            String::from_utf8(run.stdout).unwrap(),
            "1 1 2\n14 15 16\n24 25 26\n58 59 60\n59 60 60\n"
        );
    }

    #[test]
    fn file_longer_than_what_is_read_at_a_time_is_identified_whole() {
        // two megabytes and a half, which are read a megabyte at a time
        let dir = std::env::temp_dir().join(format!("qqcode-show-long-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("long.bin");
        let long = bytes(2_500_000);
        fs::write(&file, &long).unwrap();
        let (identified, md5) = identify(&file, "long.bin").unwrap();
        assert_eq!(identified.origin, Origin::of(&long));
        assert_eq!(identified.sections(), 3);
        assert!(
            identified == Checks::of("long.bin", &long),
            "the checksums of the file"
        );
        assert_eq!(md5, qqcode::digest::md5(&long));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn name_does_not_end_a_line_of_the_script_it_stands_in() {
        // the line and the paragraph separator, which end a line in a script and not in JSON
        assert_eq!(json("a\u{2028}b\u{2029}"), "a\\u2028b\\u2029");
    }

    #[test]
    fn bytes_of_an_address_are_two_digits_each() {
        // a tab and a line break in the name of a page
        assert_eq!(
            address(Path::new("/tmp/a\tb\n.html")),
            "file:///tmp/a%09b%0A.html"
        );
    }

    #[test]
    fn time_is_said_in_minutes_from_two_of_them_on() {
        assert_eq!(
            [span(0), span(119), span(120), span(121)],
            ["0 s", "119 s", "2 min 0 s", "2 min 1 s"]
        );
    }

    #[test]
    fn summary_of_a_short_file_that_the_page_asks_for_says_no_more_than_the_file() {
        // three megabytes, far less than the 83 that a group of blocks comes to
        let origin = Origin {
            checksum: 1,
            length: 3_000_000,
        };
        let text = summary(
            Path::new("show.html"),
            "notes.bin",
            origin,
            false,
            format(Shape::Wide, 11),
        );
        assert!(
            text.contains("; the page asks for the file when it is opened, and reads it 3 MB at a time\n"),
            "{text}"
        );
    }
}
