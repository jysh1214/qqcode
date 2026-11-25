//! The show as a web page: one file that loads nothing else and holds the file to show, which a browser
//! puts on the screen as a matrix of tiles ([`qqcode::tiles`]), frame after frame.
//!
//! The page does in JavaScript what the library does in Rust: it cuts the file into packets, puts each
//! into a frame and draws the tiles. What is here writes the page and, for the tests, says what the page
//! has to show.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

use qqcode::fountain::{self, Plan};
use qqcode::packet::{OVERHEAD, Origin};
use qqcode::tiles::{self, Grid};
#[cfg(test)]
use qqcode::{fountain::Encoder, packet::Frame};

const TEMPLATE: &str = include_str!("page.html");

/// A file as the page shows it, for the tests: the library's own account of what the JavaScript of the
/// page does.
#[cfg(test)]
pub struct Show<'a> {
    pub grid: Grid,
    pub origin: Origin,
    name: &'a str,
    encoder: Encoder<'a>,
}

#[cfg(test)]
impl<'a> Show<'a> {
    pub fn new(name: &'a str, bytes: &'a [u8], grid: Grid) -> Self {
        let origin = Origin::of(bytes);
        Self {
            grid,
            origin,
            name,
            encoder: Encoder::new(bytes, origin.checksum, grid.capacity() - OVERHEAD),
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
                let (block, sequence) = self.encoder.plan().turn(frame);
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

/// The page that shows `bytes`, which are called `name`, at `rate` frames a second.
pub fn page(name: &str, bytes: &[u8], grid: Grid, rate: u32) -> String {
    let origin = Origin::of(bytes);
    let colours = tiles::COLOURS.map(|[red, green, blue]| format!("[{red},{green},{blue}]"));
    let meta = format!(
        "{{\"name\":\"{}\",\"checksum\":{},\"pace\":{rate},\"block\":{},\"colours\":[{}],\"matrix\":{{\"columns\":{},\
         \"rows\":{},\"pitch\":{},\"sub\":{},\"border\":{},\"anchor\":{},\"module\":{},\"symbolParity\":{},\
         \"colourParity\":{}}}}}",
        json(name),
        origin.checksum,
        fountain::BLOCK,
        colours.join(","),
        grid.columns,
        grid.rows,
        tiles::PITCH,
        tiles::SUB,
        tiles::BORDER,
        tiles::ANCHOR,
        tiles::MODULE,
        tiles::SYMBOL_PARITY,
        tiles::COLOUR_PARITY,
    );
    TEMPLATE
        .replace("{{TITLE}}", &html(name))
        .replace("{{META}}", &meta)
        .replace("{{FILE}}", &base64(bytes))
}

/// Writes the page for `file` to `to` and says what is in it.
pub fn write(file: &Path, to: &Path, grid: Grid, rate: u32) -> Result<()> {
    let bytes = fs::read(file).with_context(|| format!("can't read {}", file.display()))?;
    let name = file.file_name().unwrap_or_default().to_string_lossy();
    fs::write(to, page(&name, &bytes, grid, rate))
        .with_context(|| format!("can't write {}", to.display()))?;
    println!("{}", summary(to, &name, &bytes, grid, rate));
    println!("SHA-256 {}", qqcode::digest::sha256(&bytes));
    println!("MD5 {}", qqcode::digest::md5(&bytes));
    Ok(())
}

/// What a page that was written holds, and what it asks of the screen and of the camera.
fn summary(to: &Path, name: &str, bytes: &[u8], grid: Grid, rate: u32) -> String {
    let packet = grid.capacity() - OVERHEAD;
    // one frame for each symbol of the file at least
    let frames = Plan::new(bytes.len() as u64, packet).symbols();
    let seconds = (frames as f64 / rate as f64).ceil();
    format!(
        "wrote {}: {name}, {} bytes in {} frames of {packet} bytes, {rate} a second: {} kB a second, {seconds} s \
         if every frame is read\n\
         matrix of {}x{} tiles, {}x{} px: best shown at that size or a whole multiple of it, and filling the \
         frames of the camera",
        to.display(),
        bytes.len(),
        frames,
        packet * rate as usize / 1000,
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
        let page = page("</script><b>", b"abc", Grid::new(Shape::Square, 4), 15);
        assert!(page.contains("<title>&lt;/script&gt;&lt;b&gt;</title>"));
        assert!(page.contains("\"name\":\"\\u003c/script\\u003e\\u003cb\\u003e\""));
        // the two scripts of data and the one of the program, each ended once
        assert_eq!(page.matches("</script>").count(), 3);
    }

    #[test]
    fn page_holds_the_file_and_what_it_takes_to_show_it() {
        let file = bytes(1000);
        let page = page("notes.bin", &file, Grid::new(Shape::Wide, 11), 20);
        assert!(!page.contains("{{"), "nothing left to fill in");
        assert!(page.contains(&format!(">{}</script>", base64(&file))));
        let checksum = Origin::of(&file).checksum;
        assert!(page.contains(&format!(
            "{{\"name\":\"notes.bin\",\"checksum\":{checksum},\"pace\":20,\"block\":512,\"colours\":[[255,255,255],\
             [0,255,255],[255,0,255],[255,255,0]],\"matrix\":{{\"columns\":176,\"rows\":99,\"pitch\":9,\"sub\":2,\
             \"border\":4,\"anchor\":6,\"module\":6,\"symbolParity\":64,\"colourParity\":40}}}}"
        )));
    }

    #[test]
    fn show_is_the_start_screen_and_then_the_packets_by_turns() {
        let file = bytes(3000);
        let grid = Grid::new(Shape::Square, 4);
        // 592 bytes a frame, 27 of which are not the packet
        assert_eq!(grid.capacity(), 592);
        let show = Show::new("notes.bin", &file, grid);
        assert_eq!(show.frames(), 6);
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
        assert_eq!((block, sequence, &bytes[..]), (0, 2, &file[1130..1695]));
        assert!(matches!(
            show.frame(Some(1000)),
            Frame::Packet { sequence: 1000, .. }
        ));
        assert_eq!(show.tiles(Some(5)).len(), grid.tiles());
    }

    #[test]
    fn summary_says_what_the_page_takes() {
        let file = bytes(3_000_000);
        let text = summary(
            Path::new("show.html"),
            "notes.bin",
            &file,
            Grid::new(Shape::Wide, 11),
            15,
        );
        assert_eq!(
            text,
            "wrote show.html: notes.bin, 3000000 bytes in 298 frames of 10077 bytes, 15 a second: 151 kB a second, \
             20 s if every frame is read\n\
             matrix of 176x99 tiles, 1592x899 px: best shown at that size or a whole multiple of it, and filling \
             the frames of the camera"
        );
    }

    /// What the JavaScript of the page makes of a file, next to what the library makes of it: the tiles of
    /// the start screen and of some frames. It takes a JavaScript engine to find out, and the test says
    /// so and passes if there is none.
    #[test]
    fn page_computes_the_tiles_that_the_library_does() {
        // three blocks, the last symbol short, and a name that is cut in the middle of a character
        let file = bytes(700_123);
        let grid = Grid::new(Shape::Square, 4);
        let name = format!("{}ü.bin", "n".repeat(465));
        let show = Show::new(&name, &file, grid);
        let frames = [0u64, 1, 2, 1493, 1494, 4479, 4480, 4481, 4500, 9000, 123_456];
        let page = page(&name, &file, grid, 15);
        let driver = r#"
            const page = await Deno.readTextFile(Deno.args[0]);
            const script = page.split("<script>")[1].split("</script>")[0];
            const meta = JSON.parse(page.split('<script id="meta" type="application/json">')[1].split("</script>")[0]);
            const file = page.split('<script id="file" type="application/octet-stream">')[1].split("</script>")[0];
            const { Show, fileBytes } = new Function(script + "; return { Show, fileBytes };")();
            const show = new Show(meta, fileBytes(file));
            const hex = (bytes) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
            console.log(hex(show.tiles(null)));
            for (const frame of Deno.args.slice(1)) console.log(hex(show.tiles(Number(frame))));
        "#;
        let dir = std::env::temp_dir().join(format!("qqcode-show-page-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("page.html"), page).unwrap();
        fs::write(dir.join("driver.js"), driver).unwrap();
        let run = Command::new("deno")
            .arg("run")
            .arg("--allow-read")
            .arg(dir.join("driver.js"))
            .arg(dir.join("page.html"))
            .args(frames.map(|frame| frame.to_string()))
            .output();
        let _ = fs::remove_dir_all(&dir);
        let Ok(run) = run else {
            eprintln!("deno is not installed: what the page computes is not compared with the library");
            return;
        };
        assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
        let hex = |tiles: Vec<u8>| tiles.iter().map(|tile| format!("{tile:02x}")).collect::<String>();
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
}
