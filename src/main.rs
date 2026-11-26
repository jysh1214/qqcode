mod preview;

use std::collections::{BTreeMap, VecDeque};
use std::fs::{self, File};
use std::io::{self, IsTerminal, Write};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use opencv::core::{Mat, Size};
use opencv::prelude::*;
use opencv::videoio;

use qqcode::assembly::Assembly;
use qqcode::digest::{self, Digests};
use qqcode::packet::Origin;
use qqcode::protocol::Checksum;
use qqcode::scan::{Found, Scanner};
use qqcode::{Outcome, Pipeline};

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

    /// File to write instead of the standard output. A file that a web page shows is put together on
    /// disk as it arrives, under the name FILE.part until it is whole, where the standard output gets it
    /// in one piece at the end: a file that is larger than the memory needs this
    #[arg(short, long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Frame size to ask a camera for, which gives the nearest one it has [default: 1920x1080]
    #[arg(long, value_parser = parse_size, value_name = "WIDTHxHEIGHT")]
    camera_size: Option<Size>,

    /// Show what the camera sees on this terminal while reading, with the codes that are found in it
    /// marked, for pointing the camera; q quits
    #[arg(long)]
    preview: bool,

    /// How many frames are searched for codes at a time [default: one for each processor but one]
    #[arg(long, value_name = "COUNT", value_parser = clap::value_parser!(u64).range(1..=256))]
    threads: Option<u64>,

    /// How many frames may wait to be searched. A camera never waits: what it takes while this many are
    /// waiting is dropped
    #[arg(long, default_value_t = 256, value_name = "FRAMES", value_parser = clap::value_parser!(u64).range(1..=65536))]
    buffer: u64,
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
    let capture = open(&cli.input, cli.camera_size)?;
    // a camera goes on whatever becomes of its frames, a file can wait
    let live = cli.input.parse::<i32>().is_ok();
    let mut log = Log {
        preview: if cli.preview { Some(Preview::open()?) } else { None },
        held: Vec::new(),
    };

    // The frames go from the camera into a buffer, are searched for codes by as many threads as there
    // are, and what is found in them is taken in here, in the order of the frames.
    let buffer = Arc::new(Buffer::new(cli.buffer as usize));
    let dropped = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let camera = {
        let (buffer, dropped, stop) = (buffer.clone(), dropped.clone(), stop.clone());
        thread::spawn(move || capture_frames(capture, &buffer, live, &dropped, &stop))
    };
    let threads = cli.threads.map_or_else(
        || thread::available_parallelism().map_or(1, |count| count.get().saturating_sub(1).max(1)),
        |count| count as usize,
    );
    let scanner = Arc::new(Scanner::default());
    let (found_by, found) = mpsc::channel();
    for _ in 0..threads {
        let (buffer, scanner, found_by) = (buffer.clone(), scanner.clone(), found_by.clone());
        thread::spawn(move || {
            while let Some((number, frame)) = buffer.take() {
                let codes = scanner.scan(&frame);
                if found_by.send((number, frame, codes)).is_err() {
                    break;
                }
            }
        });
    }
    drop(found_by);

    // A file goes into the memory until it is whole, or with -o to its place as it arrives.
    let store = cli.output.as_deref().map(Store::new);
    let mut pipeline = match &store {
        Some(store) => Pipeline::storing(&store.part, &store.sums),
        None => Pipeline::new(),
    };
    let mut done = false;
    let (mut frames, mut unread) = (0u64, 0u64);
    // the screen of qqcode-show that the last frame with one of its codes showed
    let mut screen = None;
    let mut file = None;
    let (mut frame_rate, mut frame_at) = (0.0, Instant::now());
    let (mut told_at, mut drawn_at) = (Instant::now(), None::<Instant>);
    // what was found in frames that came back before an earlier one
    let mut early: BTreeMap<u64, (Mat, Vec<Found>)> = BTreeMap::new();
    'frames: for (number, frame, codes) in found {
        early.insert(number, (frame, codes.with_context(|| format!("frame {number}"))?));
        while let Some((frame, codes)) = early.remove(&(frames + 1)) {
            frames += 1;
            let files_before = pipeline.files();
            let outcome = pipeline.take(codes).with_context(|| format!("frame {frames}"))?;
            if files_before > 0 && pipeline.files() > files_before {
                log.say(format!(
                    "frame {frames}: the codes of another file, which is the one that is read from here on"
                ));
                screen = None;
            }
            // once for a file: what it is to be kept in
            if files_before < pipeline.files()
                && store.is_none()
                && let Some(file) = pipeline.file().filter(|file| file.length > 1 << 30)
            {
                log.say(format!(
                    "frame {frames}: the file has {} bytes and is kept in memory until it is whole; with -o FILE \
                     it is put together on disk as it arrives",
                    file.length
                ));
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
                Outcome::Packet { new } => {
                    screen = Some(Shown::Codes);
                    // a line a second, not one for each of the frames
                    if new && told_at.elapsed() >= Duration::from_secs(1) {
                        told_at = Instant::now();
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
                        "frame {frames}: all of the file has arrived and its checksum is wrong, so something \
                         was misread: starting over with what is still on show"
                    ));
                }
                Outcome::Complete => {
                    done = true;
                    file = pipeline.take_file();
                    log.say(format!("frame {frames}: the file is whole"));
                }
            }
            // smoothed over the last few frames
            let now = Instant::now();
            let rate = 1.0 / now.duration_since(frame_at).as_secs_f64().max(1e-3);
            frame_rate = if frames == 1 {
                rate
            } else {
                0.9 * frame_rate + 0.1 * rate
            };
            frame_at = now;
            // the terminal takes its time over a picture: a dozen of them a second are enough to point by
            let due = drawn_at.is_none_or(|at| at.elapsed() >= Duration::from_millis(80)) || done;
            if let Some(preview) = &mut log.preview
                && due
            {
                drawn_at = Some(Instant::now());
                let size = frame.size()?;
                let facts = format!(
                    "{}x{} | frame {frames} | {frame_rate:.1} frames a second | {} waiting | {}",
                    size.width,
                    size.height,
                    buffer.waiting(),
                    progress(&pipeline)
                );
                let remark = log.held.last().map_or("", String::as_str);
                preview.draw(&frame, pipeline.last_survey(), &facts, remark)?;
                if preview.quit_requested()? {
                    log.say(format!("frame {frames}: stopped by a key"));
                    break 'frames;
                }
            }
            if done {
                break 'frames;
            }
        }
    }
    // nothing more is wanted of the camera or of the frames that wait
    stop.store(true, Ordering::Relaxed);
    buffer.close();
    let taken = camera
        .join()
        .map_err(|_| anyhow::anyhow!("the camera's thread failed"))??;
    // the terminal comes back before anything else is said
    log.close();

    let dropped = dropped.load(Ordering::Relaxed);
    if dropped > 0 {
        eprintln!(
            "{dropped} frames of the camera were dropped, as {} were waiting to be searched: more --threads or \
             a larger --buffer keep them",
            cli.buffer
        );
    }
    if done {
        let origin = pipeline.file().context("a file that is whole has a name")?;
        let (written, to, digests) = match store {
            Some(store) => {
                let to = store.path.display().to_string();
                let (written, digests) = store.finish(origin, file)?;
                (written, to, digests)
            }
            None => {
                let file = file.unwrap_or_default();
                let mut out = io::stdout().lock();
                out.write_all(&file)
                    .and_then(|()| out.flush())
                    .context("can't write the file to the standard output")?;
                let digests = (digest::sha256(&file), digest::md5(&file));
                (file.len() as u64, "the standard output".to_owned(), digests)
            }
        };
        let bytes = if written == 1 {
            "1 byte".to_owned()
        } else {
            format!("{written} bytes")
        };
        eprintln!("{frames} frames read, {bytes} written to {to}");
        // to be compared with what qqcode-show prints, or sha256sum and md5sum where the file comes from
        eprintln!("SHA-256 {}", digests.0);
        eprintln!("MD5 {}", digests.1);
        return Ok(());
    }
    if let Some(store) = store {
        store.discard();
    }
    if taken == 0 {
        bail!("no frame could be read from {}", cli.input);
    }
    eprintln!("{frames} frames read, nothing written");
    if let Some(assembly) = pipeline.assembly() {
        bail!("the file is not whole: {}", lacking(assembly));
    }
    if let Some((held, needed)) = pipeline.packets() {
        bail!("the file is not whole: {held} of the {needed} packets it takes have arrived");
    }
    if unread > 0 {
        bail!(
            "no code of qqcode-show was read; {unread} frames showed one that could not be: nearer, sharper, \
             or larger modules"
        );
    }
    bail!("no frame showed a code of qqcode-show")
}

/// Where a file is put together while it arrives: next to the place it is to have, under another name,
/// and moved there when it is whole and its checksum is right.
struct Store {
    path: PathBuf,
    part: PathBuf,
    /// The sums of the packets of a web page, which are of no use once the file is whole
    sums: PathBuf,
}

impl Store {
    fn new(path: &Path) -> Self {
        let beside = |ending: &str| {
            let mut name = path.as_os_str().to_owned();
            name.push(ending);
            PathBuf::from(name)
        };
        Self {
            path: path.to_owned(),
            part: beside(".part"),
            sums: beside(".part.sums"),
        }
    }

    /// Ends the writing of a file that is whole. `kept` is the file if it was kept in memory, which is
    /// checked already. A file that was put together on disk is read once more, for its checksum and its
    /// length, which have to be those of `origin`. Returns how many bytes the file has, and its SHA-256
    /// and MD5.
    fn finish(self, origin: Origin, kept: Option<Vec<u8>>) -> Result<(u64, (String, String))> {
        let part = self.part.display().to_string();
        let digests = match kept {
            Some(file) => {
                fs::write(&self.part, &file).with_context(|| format!("can't write {part}"))?;
                (digest::sha256(&file), digest::md5(&file))
            }
            None => {
                let file = File::open(&self.part).with_context(|| format!("can't read {part}"))?;
                let (mut checksum, mut digests, mut length) = (Checksum::new(), Digests::default(), 0u64);
                let mut piece = vec![0u8; 1 << 20];
                loop {
                    let read = file
                        .read_at(&mut piece, length)
                        .with_context(|| format!("can't read {part}"))?;
                    if read == 0 {
                        break;
                    }
                    checksum.update(&piece[..read]);
                    digests.update(&piece[..read]);
                    length += read as u64;
                }
                // nothing but this stands between a block that was solved wrongly and the file
                ensure!(
                    length == origin.length && checksum.finish() == origin.checksum,
                    "the file that was put together has not the length and the checksum that its frames name: \
                     it is left as {part}"
                );
                digests.finish()
            }
        };
        fs::rename(&self.part, &self.path)
            .with_context(|| format!("can't move {part} to {}", self.path.display()))?;
        let _ = fs::remove_file(&self.sums);
        Ok((origin.length, digests))
    }

    /// Removes what there is of a file that did not become whole.
    fn discard(self) {
        let _ = fs::remove_file(&self.part);
        let _ = fs::remove_file(&self.sums);
    }
}

/// The frames that wait to be searched for codes, in the order they were taken.
struct Buffer {
    room: usize,
    state: Mutex<Waiting>,
    changed: Condvar,
}

#[derive(Default)]
struct Waiting {
    frames: VecDeque<(u64, Mat)>,
    /// How many frames have been put in, which numbers them from 1
    taken: u64,
    closed: bool,
}

impl Buffer {
    fn new(room: usize) -> Self {
        Self {
            room,
            state: Mutex::default(),
            changed: Condvar::new(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, Waiting> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Puts a frame in. If there is no room for it, the frame is dropped, unless it is to `wait` for room.
    /// Whether it was put in.
    fn put(&self, frame: Mat, wait: bool) -> bool {
        let mut state = self.state();
        while wait && state.frames.len() >= self.room && !state.closed {
            state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
        if state.closed || state.frames.len() >= self.room {
            return false;
        }
        state.taken += 1;
        let number = state.taken;
        state.frames.push_back((number, frame));
        self.changed.notify_all();
        true
    }

    /// The frame that has waited longest, with its number; `None` once the buffer is closed and empty.
    fn take(&self) -> Option<(u64, Mat)> {
        let mut state = self.state();
        loop {
            if let Some(frame) = state.frames.pop_front() {
                self.changed.notify_all();
                return Some(frame);
            }
            if state.closed {
                return None;
            }
            state = self.changed.wait(state).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn waiting(&self) -> usize {
        self.state().frames.len()
    }

    /// No more frames come: those that wait are still given out.
    fn end(&self) {
        self.state().closed = true;
        self.changed.notify_all();
    }

    /// No more frames are wanted: those that wait are dropped.
    fn close(&self) {
        let mut state = self.state();
        state.closed = true;
        state.frames.clear();
        self.changed.notify_all();
    }
}

/// Puts the frames of a camera or file into the buffer until they end or `stop` says so. Returns how many
/// frames there were.
fn capture_frames(
    mut capture: videoio::VideoCapture,
    buffer: &Buffer,
    live: bool,
    dropped: &AtomicU64,
    stop: &AtomicBool,
) -> Result<u64> {
    let mut taken = 0;
    while !stop.load(Ordering::Relaxed) {
        let mut frame = Mat::default();
        if !capture.read(&mut frame)? || frame.empty() {
            break;
        }
        taken += 1;
        if !buffer.put(frame, !live) && !stop.load(Ordering::Relaxed) {
            dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    buffer.end();
    Ok(taken)
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
            let bytes = format!("{received} of {}", count(file.length as usize, "byte"));
            match pipeline.packets() {
                Some((held, needed)) => format!("{bytes}, {held} of {needed} packets"),
                None => bytes,
            }
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
