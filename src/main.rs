mod preview;

use std::fs;
use std::io::{self, IsTerminal, Write};
use std::ops::Range;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::videoio;

use qqcode::assembly::Assembly;
use qqcode::{Outcome, Pipeline, digest};

use preview::Preview;

/// The frame size that a camera is asked for unless another one is.
const CAMERA_SIZE: Size = Size {
    width: 1920,
    height: 1080,
};

/// Reads the file that `qqcode-show` has on show from the frames of a camera and writes it to the standard
/// output: `qqcode > file`. It ends by itself when the file is whole, and writes nothing before that.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// Camera index, or in place of a camera a video file, an image or an image sequence (frame_%04d.png)
    #[arg(default_value = "0")]
    input: String,

    /// File to write instead of the standard output
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Frame size to ask a camera for, which gives the nearest one it has [default: 1920x1080]
    #[arg(long, value_parser = parse_size, value_name = "WIDTHxHEIGHT")]
    camera_size: Option<Size>,

    /// Show what the camera sees on this terminal while reading, with the codes that are found in it
    /// marked, for pointing the camera; q quits
    #[arg(long)]
    preview: bool,
}

fn parse_size(text: &str) -> Result<Size, String> {
    let parsed = text.split_once(['x', 'X']).and_then(|(width, height)| {
        let (width, height) = (
            width.trim().parse::<i32>().ok()?,
            height.trim().parse::<i32>().ok()?,
        );
        (width > 0 && height > 0).then(|| Size::new(width, height))
    });
    parsed.ok_or_else(|| format!("expected WIDTHxHEIGHT such as 1600x900, got {text:?}"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    ensure!(
        cli.output.is_some() || !io::stdout().is_terminal(),
        "the file goes to the standard output, which is this terminal: send it to a file (qqcode > FILE) or \
         name one with -o"
    );
    let mut capture = open(&cli.input, cli.camera_size)?;
    let mut log = Log {
        preview: if cli.preview { Some(Preview::open()?) } else { None },
        held: Vec::new(),
    };
    let mut pipeline = Pipeline::new();

    let (mut frames, mut unread) = (0u64, 0u64);
    // the screen of qqcode-show that the last frame with one of its codes showed
    let mut screen = None;
    let mut file = None;
    let (mut frame_rate, mut frame_at) = (0.0, Instant::now());
    let mut frame = Mat::default();
    while capture.read(&mut frame)? && !frame.empty() {
        frames += 1;
        let files_before = pipeline.files();
        let outcome = pipeline
            .process(&frame)
            .with_context(|| format!("frame {frames}"))?;
        if files_before > 0 && pipeline.files() > files_before {
            log.say(format!(
                "frame {frames}: the codes of another file, which is the one that is read from here on"
            ));
            screen = None;
        }
        let screen_before = screen;
        match outcome {
            Outcome::NoCode | Outcome::Other => {}
            Outcome::Unread => unread += 1,
            Outcome::Ready => {
                screen = Some(Shown::Start);
                if screen_before != screen {
                    log.say(format!(
                        "frame {frames}: start screen in view{}, waiting for the first code",
                        named(&pipeline)
                    ));
                }
            }
            Outcome::Piece { new, skipped } => {
                screen = Some(Shown::Codes);
                if skipped > 0 {
                    log.say(format!(
                        "frame {frames}: {} went by unread",
                        count(skipped, "byte")
                    ));
                }
                if new > 0 {
                    log.tell(format!("frame {frames}: {}", progress(&pipeline)));
                }
            }
            Outcome::Done => {
                screen = Some(Shown::End);
                if screen_before != screen {
                    let missing = pipeline
                        .assembly()
                        .map_or(String::new(), |assembly| format!(", {}", lacking(assembly)));
                    log.say(format!(
                        "frame {frames}: end screen in view{missing}; g in qqcode-show shows the codes again"
                    ));
                }
            }
            Outcome::Corrupt => {
                screen = None;
                log.say(format!(
                    "frame {frames}: all pieces have arrived and the checksum of the file is wrong, so a code \
                     was misread: starting over, g in qqcode-show shows the codes again"
                ));
            }
            Outcome::Complete => {
                file = pipeline.take_file();
                log.say(format!("frame {frames}: the file is whole"));
            }
        }
        if let Some(preview) = &mut log.preview {
            // smoothed over the last few frames
            let now = Instant::now();
            let rate = 1.0 / now.duration_since(frame_at).as_secs_f64().max(1e-3);
            frame_rate = if frames == 1 {
                rate
            } else {
                0.8 * frame_rate + 0.2 * rate
            };
            frame_at = now;
            let size = frame.size()?;
            let facts = format!(
                "{}x{} | frame {frames} | {frame_rate:.1} frames a second | {}",
                size.width,
                size.height,
                progress(&pipeline)
            );
            let remark = log.held.last().map_or("", String::as_str);
            preview.draw(&frame, pipeline.last_survey(), &facts, remark)?;
            if preview.quit_requested()? {
                log.say(format!("frame {frames}: stopped by a key"));
                break;
            }
        }
        if file.is_some() {
            break;
        }
    }
    // the terminal comes back before anything else is said
    log.close();

    if let Some(file) = file {
        let to = match &cli.output {
            Some(path) => {
                fs::write(path, &file).with_context(|| format!("can't write {}", path.display()))?;
                path.display().to_string()
            }
            None => {
                let mut out = io::stdout().lock();
                out.write_all(&file)
                    .and_then(|()| out.flush())
                    .context("can't write the file to the standard output")?;
                "the standard output".to_owned()
            }
        };
        eprintln!(
            "{frames} frames read, {} written to {to}",
            count(file.len(), "byte")
        );
        // to be compared with what qqcode-show prints, or sha256sum and md5sum where the file comes from
        eprintln!("SHA-256 {}", digest::sha256(&file));
        eprintln!("MD5 {}", digest::md5(&file));
        return Ok(());
    }
    if frames == 0 {
        bail!("no frame could be read from {}", cli.input);
    }
    eprintln!("{frames} frames read, nothing written");
    match pipeline.assembly() {
        Some(assembly) => bail!("the file is not whole: {}", lacking(assembly)),
        None if unread > 0 => bail!(
            "no code of qqcode-show was read; {unread} frames showed a QR code that could not be: nearer, \
             sharper, or larger modules"
        ),
        None => bail!("no frame showed a code of qqcode-show"),
    }
}

/// The screens of `qqcode-show`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shown {
    Start,
    Codes,
    End,
}

/// `1 byte`, `2 bytes`.
fn count(number: usize, of: &str) -> String {
    match number {
        1 => format!("1 {of}"),
        number => format!("{number} {of}s"),
    }
}

/// `: notes.txt, 12345 bytes`, as far as the start screen said so.
fn named(pipeline: &Pipeline) -> String {
    let Some(file) = pipeline.file() else {
        return String::new();
    };
    let size = count(file.length as usize, "byte");
    match pipeline.name() {
        Some(name) => format!(": {name}, {size}"),
        None => format!(": {size}"),
    }
}

/// `2048 of 12345 bytes`: how much of the file on show has arrived.
fn progress(pipeline: &Pipeline) -> String {
    match (pipeline.received(), pipeline.file()) {
        (Some(received), Some(file)) => {
            format!("{received} of {}", count(file.length as usize, "byte"))
        }
        _ => "no code of qqcode-show read".to_owned(),
    }
}

/// `1092 of 3000 bytes are missing: 0 to 545, 1638 to 2183`
fn lacking(assembly: &Assembly) -> String {
    let missing = assembly.missing();
    let bytes = missing.iter().map(Range::len).sum::<usize>();
    let are = if bytes == 1 { "is" } else { "are" };
    format!(
        "{bytes} of {} {are} missing: {}",
        count(assembly.file().length as usize, "byte"),
        ranges(&missing)
    )
}

/// `0 to 545, 1638 to 2183`, and no more than a line of it.
fn ranges(ranges: &[Range<usize>]) -> String {
    const SHOWN: usize = 6;
    let mut parts = ranges
        .iter()
        .take(SHOWN)
        .map(|range| match range.len() {
            1 => range.start.to_string(),
            _ => format!("{} to {}", range.start, range.end - 1),
        })
        .collect::<Vec<_>>();
    if ranges.len() > SHOWN {
        parts.push(format!("and {} more parts", ranges.len() - SHOWN));
    }
    parts.join(", ")
}

/// Where the remarks about the run go: to stderr, at once or, while the preview has the terminal, when it
/// has given it back.
struct Log {
    preview: Option<Preview>,
    /// The remarks made while the preview was on
    held: Vec<String>,
}

impl Log {
    fn say(&mut self, remark: impl Into<String>) {
        let remark = remark.into();
        if self.preview.is_some() {
            self.held.push(remark);
        } else {
            eprintln!("{remark}");
        }
    }

    /// Says what the preview shows as it happens, and what is of no interest afterwards: nothing while
    /// the preview is on.
    fn tell(&mut self, remark: impl Into<String>) {
        if self.preview.is_none() {
            eprintln!("{}", remark.into());
        }
    }

    /// Ends the preview, if there is one, and prints what was said meanwhile.
    fn close(&mut self) {
        self.preview = None;
        for remark in self.held.drain(..) {
            eprintln!("{remark}");
        }
    }
}

impl Drop for Log {
    /// Also when an error ends the run, the terminal comes back before anything is printed.
    fn drop(&mut self) {
        self.close();
    }
}

/// Opens a camera for a plain number, a file or image sequence for anything else. A camera is asked for
/// frames of `camera_size`, or of [`CAMERA_SIZE`], and says on stderr what frames it gives.
fn open(input: &str, camera_size: Option<Size>) -> Result<videoio::VideoCapture> {
    let Ok(camera) = input.parse::<i32>() else {
        ensure!(
            camera_size.is_none(),
            "--camera-size is for a camera, which {input} is not"
        );
        let capture = videoio::VideoCapture::from_file(input, videoio::CAP_ANY)?;
        ensure!(capture.is_opened()?, "can't open {input}");
        return Ok(capture);
    };
    let mut capture = videoio::VideoCapture::new(camera, videoio::CAP_ANY)?;
    ensure!(capture.is_opened()?, "can't open camera {camera}");
    let size = request_size(&mut capture, camera_size.unwrap_or(CAMERA_SIZE))?;
    let rate = capture.property(videoio::CAP_PROP_FPS)?;
    let pace = if rate > 0.0 {
        format!(", {rate:.0} a second")
    } else {
        String::new()
    };
    eprintln!("camera {camera}: {}x{} frames{pace}", size.width, size.height);
    if let Some(wanted) = camera_size.filter(|wanted| *wanted != size) {
        eprintln!(
            "  it has no {}x{} frames, this is the nearest size (`v4l2-ctl --list-formats-ext` lists them)",
            wanted.width, wanted.height
        );
    }
    Ok(capture)
}

/// What is asked of a camera apart from frames. A trait, so that the asking can be tested without a camera.
trait Camera {
    fn property(&self, id: i32) -> Result<f64>;
    /// Whether the camera took the value
    fn set_property(&mut self, id: i32, value: f64) -> Result<bool>;
}

impl Camera for videoio::VideoCapture {
    fn property(&self, id: i32) -> Result<f64> {
        Ok(self.get(id)?)
    }

    fn set_property(&mut self, id: i32, value: f64) -> Result<bool> {
        Ok(self.set(id, value)?)
    }
}

/// The size of the frames a camera delivers.
fn frame_size(camera: &impl Camera) -> Result<Size> {
    let pixels = |id| Ok::<_, anyhow::Error>(camera.property(id)?.round() as i32);
    Ok(Size::new(
        pixels(videoio::CAP_PROP_FRAME_WIDTH)?,
        pixels(videoio::CAP_PROP_FRAME_HEIGHT)?,
    ))
}

/// Motion JPEG as OpenCV and V4L2 name pixel formats, by the four characters `MJPG`.
const MOTION_JPEG: f64 = u32::from_le_bytes(*b"MJPG") as f64;

/// Asks a camera for frames of `wanted` size. Returns the size it delivers from then on, which is the
/// nearest it has if it doesn't have that one.
///
/// A camera has each of its sizes in certain pixel formats only, and at a frame rate of its own in each.
/// USB cameras tend to have their large sizes as Motion JPEG alone, or uncompressed at a few frames a
/// second, because uncompressed frames of that size don't fit through the cable any faster. OpenCV asks in
/// the uncompressed format it prefers, so the question is repeated for Motion JPEG, and that is taken
/// where it comes nearer to the size or brings more frames a second: a code that is on show for half a
/// second is easily missed at five frames a second, and it is read from a compressed frame as well as from
/// any other.
fn request_size(camera: &mut impl Camera, wanted: Size) -> Result<Size> {
    fn ask(camera: &mut impl Camera, wanted: Size) -> Result<Size> {
        // the camera changes its size when the second of the two arrives
        camera.set_property(videoio::CAP_PROP_FRAME_WIDTH, wanted.width as f64)?;
        camera.set_property(videoio::CAP_PROP_FRAME_HEIGHT, wanted.height as f64)?;
        frame_size(camera)
    }

    let given = ask(camera, wanted)?;
    let format = camera.property(videoio::CAP_PROP_FOURCC)?;
    if format == MOTION_JPEG {
        return Ok(given);
    }
    let rate = camera.property(videoio::CAP_PROP_FPS)?;
    if !camera.set_property(videoio::CAP_PROP_FOURCC, MOTION_JPEG)? {
        return Ok(given);
    }
    let compressed = ask(camera, wanted)?;
    let nearer = given != wanted && compressed == wanted;
    let faster = compressed == given && camera.property(videoio::CAP_PROP_FPS)? > rate;
    if nearer || faster {
        return Ok(compressed);
    }
    // no better for it: back to what the camera gave first
    camera.set_property(videoio::CAP_PROP_FOURCC, format)?;
    ask(camera, wanted)
}

#[cfg(test)]
mod tests {
    use qqcode::protocol::FileId;

    use super::*;

    const WIDTH: i32 = videoio::CAP_PROP_FRAME_WIDTH;
    const HEIGHT: i32 = videoio::CAP_PROP_FRAME_HEIGHT;
    const FORMAT: i32 = videoio::CAP_PROP_FOURCC;
    const RATE: i32 = videoio::CAP_PROP_FPS;
    const YUYV: f64 = u32::from_le_bytes(*b"YUYV") as f64;

    /// A size that a camera has in a pixel format: width, height, frames a second.
    type Mode = (i32, i32, f64);

    /// A camera that has some sizes in each of its pixel formats and settles on the nearest one to what it
    /// is asked for, the way a V4L2 camera behaves behind OpenCV.
    struct Fake {
        formats: Vec<(f64, Vec<Mode>)>,
        format: f64,
        mode: Mode,
        /// Width and height asked for and not yet in effect, 0 if not asked for
        asked: (i32, i32),
    }

    impl Fake {
        /// Starts like OpenCV does: in the first format, at the size nearest to 640x480.
        fn new(formats: &[(f64, &[Mode])]) -> Self {
            let mut fake = Self {
                formats: formats
                    .iter()
                    .map(|(format, modes)| (*format, modes.to_vec()))
                    .collect(),
                format: formats[0].0,
                mode: (0, 0, 0.0),
                asked: (0, 0),
            };
            fake.settle((640, 480));
            fake
        }

        fn settle(&mut self, wanted: (i32, i32)) {
            let modes = &self
                .formats
                .iter()
                .find(|(format, _)| *format == self.format)
                .unwrap()
                .1;
            let off = |mode: &&Mode| (mode.0 - wanted.0).abs() + (mode.1 - wanted.1).abs();
            self.mode = *modes.iter().min_by_key(off).unwrap();
        }
    }

    impl Camera for Fake {
        fn property(&self, id: i32) -> Result<f64> {
            Ok(match id {
                WIDTH => self.mode.0 as f64,
                HEIGHT => self.mode.1 as f64,
                FORMAT => self.format,
                RATE => self.mode.2,
                _ => 0.0,
            })
        }

        fn set_property(&mut self, id: i32, value: f64) -> Result<bool> {
            match id {
                WIDTH => self.asked.0 = value as i32,
                HEIGHT => self.asked.1 = value as i32,
                FORMAT => {
                    if !self.formats.iter().any(|(format, _)| *format == value) {
                        return Ok(false);
                    }
                    self.format = value;
                    self.settle((self.mode.0, self.mode.1));
                }
                _ => return Ok(false),
            }
            if self.asked.0 > 0 && self.asked.1 > 0 {
                self.settle(self.asked);
                self.asked = (0, 0);
            }
            Ok(true)
        }
    }

    /// Uncompressed up to 1920x1080, of which it gets five frames a second through the cable, and Motion
    /// JPEG at thirty of any size: a common webcam.
    fn webcam() -> Fake {
        Fake::new(&[
            (YUYV, &[(640, 480, 30.0), (1280, 720, 10.0), (1920, 1080, 5.0)]),
            (
                MOTION_JPEG,
                &[(640, 480, 30.0), (1280, 720, 30.0), (1920, 1080, 30.0)],
            ),
        ])
    }

    #[test]
    fn camera_starts_small() -> Result<()> {
        assert_eq!(frame_size(&webcam())?, Size::new(640, 480));
        Ok(())
    }

    #[test]
    fn size_that_is_faster_compressed_switches_to_motion_jpeg() -> Result<()> {
        let mut camera = webcam();
        assert_eq!(request_size(&mut camera, CAMERA_SIZE)?, CAMERA_SIZE);
        assert_eq!((camera.format, camera.mode.2), (MOTION_JPEG, 30.0));
        Ok(())
    }

    #[test]
    fn size_that_is_as_fast_uncompressed_stays_uncompressed() -> Result<()> {
        let mut camera = webcam();
        assert_eq!(
            request_size(&mut camera, Size::new(640, 480))?,
            Size::new(640, 480)
        );
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn size_the_camera_only_has_compressed_switches_to_motion_jpeg() -> Result<()> {
        let mut camera = Fake::new(&[
            (YUYV, &[(640, 480, 30.0), (1280, 720, 10.0)]),
            (MOTION_JPEG, &[(640, 480, 30.0), (1920, 1080, 30.0)]),
        ]);
        assert_eq!(request_size(&mut camera, CAMERA_SIZE)?, CAMERA_SIZE);
        assert_eq!(camera.format, MOTION_JPEG);
        Ok(())
    }

    #[test]
    fn size_the_camera_does_not_have_gives_the_nearest_one() -> Result<()> {
        let mut camera = Fake::new(&[
            (YUYV, &[(640, 480, 30.0), (1280, 720, 30.0)]),
            (MOTION_JPEG, &[(640, 480, 30.0), (1280, 720, 30.0)]),
        ]);
        assert_eq!(
            request_size(&mut camera, Size::new(1400, 800))?,
            Size::new(1280, 720)
        );
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn camera_without_motion_jpeg_gives_what_it_has() -> Result<()> {
        let mut camera = Fake::new(&[(YUYV, &[(640, 480, 30.0), (800, 600, 15.0)])]);
        assert_eq!(request_size(&mut camera, CAMERA_SIZE)?, Size::new(800, 600));
        assert_eq!(camera.format, YUYV);
        Ok(())
    }

    #[test]
    fn camera_with_motion_jpeg_alone_is_asked_once() -> Result<()> {
        let mut camera = Fake::new(&[(MOTION_JPEG, &[(640, 480, 30.0), (1920, 1080, 30.0)])]);
        assert_eq!(request_size(&mut camera, CAMERA_SIZE)?, CAMERA_SIZE);
        assert_eq!(
            request_size(&mut camera, Size::new(3840, 2160))?,
            Size::new(1920, 1080)
        );
        Ok(())
    }

    #[test]
    fn camera_size_is_refused_for_a_file() {
        let error = open("recording.mp4", Some(Size::new(1920, 1080))).unwrap_err();
        assert!(
            error.to_string().contains("--camera-size is for a camera"),
            "{error}"
        );
    }

    #[test]
    fn sizes_are_width_x_height() {
        assert_eq!(parse_size("1920x1080"), Ok(Size::new(1920, 1080)));
        assert_eq!(parse_size(" 640 X 480 "), Ok(Size::new(640, 480)));
        for wrong in ["1920", "1920x", "0x1080", "1920x-1080", "wide x high"] {
            assert!(parse_size(wrong).is_err(), "{wrong:?}");
        }
    }

    #[test]
    fn camera_is_the_first_one_unless_another_input_is_named() {
        assert_eq!(Cli::try_parse_from(["qqcode"]).unwrap().input, "0");
        let cli = Cli::try_parse_from(["qqcode", "show.mp4", "-o", "file", "--preview"]).unwrap();
        assert_eq!((cli.input.as_str(), cli.preview), ("show.mp4", true));
        assert_eq!(cli.output, Some(PathBuf::from("file")));
    }

    #[test]
    fn missing_parts_are_named() {
        let mut assembly = Assembly::new(FileId {
            checksum: 0,
            length: 3000,
        });
        assert_eq!(lacking(&assembly), "3000 of 3000 bytes are missing: 0 to 2999");
        assembly.add(546, &[0; 1092]);
        assembly.add(2184, &[0; 815]);
        assert_eq!(
            lacking(&assembly),
            "1093 of 3000 bytes are missing: 0 to 545, 1638 to 2183, 2999"
        );
        assembly.add(0, &[0; 2999]);
        assert_eq!(lacking(&assembly), "1 of 3000 bytes is missing: 2999");
        let many = (0..10).map(|part| 10 * part..10 * part + 5).collect::<Vec<_>>();
        assert_eq!(
            ranges(&many),
            "0 to 4, 10 to 14, 20 to 24, 30 to 34, 40 to 44, 50 to 54, and 4 more parts"
        );
        assert_eq!(
            (count(1, "byte"), count(2, "byte")),
            ("1 byte".to_owned(), "2 bytes".to_owned())
        );
    }
}
