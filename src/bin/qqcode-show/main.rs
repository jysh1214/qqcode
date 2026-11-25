//! `qqcode-show` puts a file on the terminal as a sequence of QR codes, as something to point a camera at
//! and have `qqcode` read back.

mod chart;
mod terminal;
mod text;
mod web;

use std::fs;
use std::ops::{Range, RangeInclusive};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::Parser;
use clap::builder::RangedU64ValueParser;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use qqcode::code::{self, Level, MAX_VERSION};
use qqcode::digest;
use qqcode::protocol::{self, FileId, HEADER, Message, PIECE_HEADER};
use qqcode::tiles::{Grid, SCALES, Shape};

use chart::{Geometry, Rgb, Screen, Settings};
use terminal::{Size, Terminal};

const MODULE: RangeInclusive<usize> = 1..=16;
const MARGIN: RangeInclusive<usize> = 0..=16;
const VERSION: RangeInclusive<usize> = 1..=MAX_VERSION;

/// The paces that the keys step through, in seconds a code.
const PACES: [f64; 12] = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0];

const KEYS: &str = "\
Keys:
  enter, space, n, right, down    next code; from the start screen to the first code
  p, left, up                     previous code
  g, G                            first, last code
  + -                             larger, smaller modules
  } {                             more, less margin around the code
  . ,                             larger, smaller codes: the largest version they may have
  e                               the next error correction level
  f s                             faster, slower: less, more time for a code
  h                               hide or show the notes
  q, esc                          quit, and print the options for the layout as it was then

A code is as large as the terminal has room for at the size of its modules, and a module has the size of
the terminal's font: a smaller font or a larger window makes for larger codes, which hold more, and fewer
of them. The file is cut anew whenever that changes, and qqcode goes on with what it has.";

/// Shows a file on the terminal as a sequence of QR codes, as something to point a camera at: `qqcode`
/// reads the codes and puts the file together. It starts on a start screen and ends on an end screen,
/// and every code says which file it is a piece of and where in it the piece goes.
#[derive(Parser)]
#[command(version, after_help = KEYS)]
struct Cli {
    /// File to show, of any kind
    file: PathBuf,

    /// Size of a module of the code: this many columns wide and half as many rows high
    #[arg(long, default_value_t = 1, value_parser = within(MODULE), value_name = "COLUMNS")]
    module: usize,

    /// Blank space on every side of the code, in modules
    #[arg(long, default_value_t = 4, value_parser = within(MARGIN), value_name = "MODULES")]
    margin: usize,

    /// How much of a code may be misread and put right; the higher the level, the less a code holds
    #[arg(long, value_enum, default_value_t = Level::M)]
    level: Level,

    /// The largest version a code may have, where the terminal has room for more
    #[arg(long, default_value_t = MAX_VERSION, value_parser = within(VERSION), value_name = "VERSION")]
    max_version: usize,

    /// Seconds after which the next code comes up by itself, from the first code on to the end screen; 0
    /// leaves that to the keys
    #[arg(long, default_value_t = 0.5, value_name = "SECONDS")]
    interval: f64,

    /// The dark colour, of the modules and the text
    #[arg(long, default_value_t = Rgb::BLACK, value_name = "RRGGBB")]
    ink: Rgb,

    /// The light colour, of the background; a grey such as c0c0c0 dims a screen that glares
    #[arg(long, default_value_t = Rgb::WHITE, value_name = "RRGGBB")]
    paper: Rgb,

    /// Start without the notes in the panel: what is on show, the measurements and the keys
    #[arg(long)]
    no_notes: bool,

    /// Write the show as a web page to this file instead of drawing QR codes on the terminal: a matrix
    /// of coloured tiles that holds far more than a QR code, frame after frame without end, of which
    /// `qqcode` needs any that are enough
    #[arg(long, value_name = "FILE")]
    html: Option<PathBuf>,

    /// Web page: the shape of the matrix, best that of the camera's frames
    #[arg(long, value_enum, default_value_t = Shape::Wide, requires = "html")]
    shape: Shape,

    /// Web page: the size of the matrix, which has 9 rows of tiles for each; the fewer, the larger a
    /// tile is in the frames of the camera
    #[arg(long, default_value_t = 11, value_parser = within(SCALES), value_name = "SCALE", requires = "html")]
    scale: usize,

    /// Web page: frames a second, best half of what the camera takes or less
    #[arg(long, default_value_t = 15, value_parser = within(1..=60), value_name = "FRAMES", requires = "html")]
    rate: usize,
}

fn within(range: RangeInclusive<usize>) -> RangedU64ValueParser<usize> {
    RangedU64ValueParser::new().range(*range.start() as u64..=*range.end() as u64)
}

/// What a key press calls for.
#[derive(Debug, PartialEq, Eq)]
enum Reaction {
    Quit,
    Redraw,
    Nothing,
}

/// What is on show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    /// The start screen, until a key brings the first code
    Ready,
    /// The code [`App::code`]
    Codes,
    /// The end screen, after the last code
    Done,
}

/// How the file is cut for the terminal as it is.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Layout {
    geometry: Geometry,
    /// Bytes of the file in a code, fewer in the last one
    piece: usize,
    /// How many codes the file makes
    codes: usize,
}

struct App {
    file: PathBuf,
    bytes: Vec<u8>,
    id: FileId,
    /// The SHA-256 and the MD5 of the file, see [`digest`]
    sha256: String,
    md5: String,
    /// Seconds that a code stays; `None` if the keys alone turn the codes
    interval: Option<f64>,
    settings: Settings,
    level: Level,
    /// Whether the notes are shown
    notes: bool,
    /// Where in the file the code on show begins. The code is found again by this when a new layout cuts
    /// the file into other pieces.
    anchor: usize,
    /// The layout for the terminal as it was when it was last drawn on; `None` if it was too small
    layout: Option<Layout>,
    stage: Stage,
    /// Index of the code that is on show, or will be when the codes are
    code: usize,
}

impl App {
    fn new(cli: Cli) -> Result<Self> {
        let bytes = fs::read(&cli.file).with_context(|| format!("can't read {}", cli.file.display()))?;
        let id = FileId::of(&bytes).with_context(|| {
            format!(
                "{} has {} bytes, and 4 GiB or more are beyond what the codes can count",
                cli.file.display(),
                bytes.len()
            )
        })?;
        ensure!(
            cli.ink.luma() < cli.paper.luma(),
            "--ink must be darker than --paper, a code is only read dark on light"
        );
        ensure!(
            cli.interval >= 0.0 && Duration::try_from_secs_f64(cli.interval).is_ok(),
            "--interval must be a number of seconds, or 0 to turn the codes by key"
        );
        Ok(Self {
            file: cli.file,
            sha256: digest::sha256(&bytes),
            md5: digest::md5(&bytes),
            bytes,
            id,
            interval: Some(cli.interval).filter(|seconds| *seconds > 0.0),
            settings: Settings {
                module: cli.module,
                margin: cli.margin,
                max_version: cli.max_version,
                ink: cli.ink,
                paper: cli.paper,
            },
            level: cli.level,
            notes: !cli.no_notes,
            anchor: 0,
            layout: None,
            stage: Stage::Ready,
            code: 0,
        })
    }

    /// Cuts the file for a terminal of this size and finds the code that was on show among the new ones.
    fn lay_out(&mut self, columns: usize, rows: usize) {
        self.layout = Geometry::new(columns, rows, &self.settings).and_then(|geometry| {
            // a code has to hold a byte of the file at least, next to what says where it goes
            let piece = code::capacity(geometry.version, self.level)
                .checked_sub(PIECE_HEADER)
                .filter(|piece| *piece > 0)?;
            Some(Layout {
                geometry,
                piece,
                codes: self.bytes.len().div_ceil(piece),
            })
        });
        self.code = match &self.layout {
            Some(layout) => (self.anchor / layout.piece).min(layout.codes.saturating_sub(1)),
            None => 0,
        };
    }

    fn codes(&self) -> usize {
        self.layout.as_ref().map_or(0, |layout| layout.codes)
    }

    /// Puts a code on show, whatever was before, if there is such a code.
    fn turn_to(&mut self, code: usize) {
        if let Some(layout) = &self.layout
            && code < layout.codes
        {
            self.stage = Stage::Codes;
            self.code = code;
            self.anchor = code * layout.piece;
        }
    }

    /// On to the next screen: from the start screen to the first code, from the last code to the end
    /// screen.
    fn forward(&mut self) {
        match self.stage {
            Stage::Ready if self.codes() > 0 => self.turn_to(0),
            Stage::Codes if self.code + 1 < self.codes() => self.turn_to(self.code + 1),
            // only the end screen follows the last code, or a file without a byte
            Stage::Ready | Stage::Codes if self.layout.is_some() => self.stage = Stage::Done,
            Stage::Ready | Stage::Codes | Stage::Done => {}
        }
    }

    /// Back to the code before, and from the end screen to the last code.
    fn back(&mut self) {
        match self.stage {
            Stage::Ready => {}
            Stage::Codes => self.turn_to(self.code.saturating_sub(1)),
            Stage::Done => self.turn_to(self.codes().saturating_sub(1)),
        }
    }

    /// Lets the codes be one version smaller than they are, or one larger than they may be.
    fn resize(&mut self, larger: bool) {
        let limit = &mut self.settings.max_version;
        if larger {
            *limit = (*limit + 1).min(MAX_VERSION);
        } else {
            let shown = self
                .layout
                .as_ref()
                .map_or(*limit, |layout| layout.geometry.version);
            *limit = shown.min(*limit).saturating_sub(1).max(1);
        }
    }

    /// Moves the time that a code stays to the next of the [`PACES`], if the codes turn by themselves.
    fn hurry(&mut self, faster: bool) {
        if let Some(seconds) = &mut self.interval {
            let next = if faster {
                PACES.iter().rev().find(|pace| **pace < *seconds - 1e-9)
            } else {
                PACES.iter().find(|pace| **pace > *seconds + 1e-9)
            };
            *seconds = next.copied().unwrap_or(*seconds);
        }
    }

    fn on_key(&mut self, key: KeyEvent) -> Reaction {
        let settings = &mut self.settings;
        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => return Reaction::Quit,
            KeyCode::Char('q') | KeyCode::Esc => return Reaction::Quit,
            KeyCode::Char(' ' | 'n')
            | KeyCode::Right
            | KeyCode::Down
            | KeyCode::PageDown
            | KeyCode::Enter => self.forward(),
            KeyCode::Char('p') | KeyCode::Left | KeyCode::Up | KeyCode::PageUp | KeyCode::Backspace => {
                self.back()
            }
            KeyCode::Char('g') | KeyCode::Home => self.turn_to(0),
            KeyCode::Char('G') | KeyCode::End => self.turn_to(self.codes().saturating_sub(1)),
            KeyCode::Char('+' | '=') => step(&mut settings.module, 1, MODULE),
            KeyCode::Char('-' | '_') => step(&mut settings.module, -1, MODULE),
            KeyCode::Char('}') => step(&mut settings.margin, 1, MARGIN),
            KeyCode::Char('{') => step(&mut settings.margin, -1, MARGIN),
            KeyCode::Char('.' | '>') => self.resize(true),
            KeyCode::Char(',' | '<') => self.resize(false),
            KeyCode::Char('e') => self.level = self.level.next(),
            KeyCode::Char('f') => self.hurry(true),
            KeyCode::Char('s') => self.hurry(false),
            KeyCode::Char('h' | '?') => self.notes = !self.notes,
            _ => return Reaction::Nothing,
        }
        Reaction::Redraw
    }

    /// The bytes of the file that the code on show holds.
    fn piece(&self, layout: &Layout) -> Range<usize> {
        let start = self.code * layout.piece;
        start..(start + layout.piece).min(self.bytes.len())
    }

    /// What the code on show says.
    fn message(&self, layout: &Layout) -> Message {
        match self.stage {
            Stage::Ready => {
                // the name is there to be read by people, and cut where the code ends
                let room = code::capacity(layout.geometry.version, self.level) - HEADER;
                let name = self.file.file_name().unwrap_or_default().to_string_lossy();
                Message::Start {
                    file: self.id,
                    name: protocol::fit_name(&name, room).to_owned(),
                }
            }
            Stage::Codes => {
                let piece = self.piece(layout);
                Message::Piece {
                    file: self.id,
                    offset: piece.start as u32,
                    bytes: self.bytes[piece].to_vec(),
                }
            }
            Stage::Done => Message::End { file: self.id },
        }
    }

    /// The picture for the layout made by [`App::lay_out`].
    fn screen(&self, size: &Size) -> Screen {
        let Some(layout) = &self.layout else {
            let lines = [
                format!(
                    "{}x{} cells are too few for a code that holds a piece of the file.",
                    size.columns, size.rows
                ),
                String::new(),
                "A smaller font, or: - smaller modules, { less margin, e another level".to_owned(),
            ];
            return Screen::notice(size.columns, size.rows, &self.settings, &lines);
        };
        let message = self.message(layout).encode();
        let code = code::encode(&message, layout.geometry.version, self.level)
            .expect("the pieces are cut to what a code holds");
        let said = match self.stage {
            Stage::Ready => self.start_message(layout),
            Stage::Codes => Vec::new(),
            Stage::Done => self.end_message(),
        };
        let notes = if self.notes {
            self.notes(layout, size)
        } else {
            Vec::new()
        };
        chart::compose(&layout.geometry, &self.settings, &code, &said, &notes)
    }

    /// `, a new one every 0.5 s`, if the codes turn by themselves.
    fn pace(&self) -> String {
        match self.interval {
            Some(seconds) => format!(", a new one every {seconds} s"),
            None => String::new(),
        }
    }

    /// What the start screen says.
    fn start_message(&self, layout: &Layout) -> Vec<String> {
        vec![
            format!(
                "Ready to show {}: {} in {}{}.",
                self.file.display(),
                count(self.bytes.len(), "byte"),
                count(layout.codes, "code"),
                self.pace()
            ),
            String::new(),
            "Point the camera at this code. qqcode says so when it has it in view.".to_owned(),
            String::new(),
            "Enter starts.".to_owned(),
        ]
    }

    /// What the end screen says.
    fn end_message(&self) -> Vec<String> {
        vec![
            format!("End of {}.", self.file.display()),
            String::new(),
            format!("SHA-256 {}", self.sha256),
            format!("MD5 {}", self.md5),
            String::new(),
            "qqcode has ended by itself if it has read every code, and has printed the same SHA-256 and \
             MD5. If it names parts that are missing, g shows the codes again."
                .to_owned(),
            String::new(),
            "p goes back, g starts again, q quits.".to_owned(),
        ]
    }

    /// What is on show and what the layout measures, then the keys.
    fn notes(&self, layout: &Layout, size: &Size) -> Vec<String> {
        let settings = &self.settings;
        let geometry = &layout.geometry;
        let stage = match self.stage {
            Stage::Ready => format!("start, {} to come", count(layout.codes, "code")),
            Stage::Codes => {
                let piece = self.piece(layout);
                format!(
                    "code {}/{}, bytes {} to {} of {}",
                    self.code + 1,
                    layout.codes,
                    piece.start,
                    piece.end - 1,
                    self.bytes.len()
                )
            }
            Stage::Done => format!("end, {} shown", count(layout.codes, "code")),
        };
        let code = format!(
            "version {}, level {}, {} a code{}",
            geometry.version,
            self.level,
            count(layout.piece, "byte"),
            self.pace()
        );
        let sizes = format!(
            "module {}, margin {}, {}",
            settings.module,
            settings.margin,
            code_size(geometry, size)
        );
        let keys = [
            "",
            "enter/space/n next, p previous, g/G first/last code",
            "+/- modules, }/{ margin, ./, version, e level",
            "f/s faster/slower, h hide these notes, q quit",
        ];
        [stage, code, sizes]
            .into_iter()
            .chain(keys.map(str::to_owned))
            .collect()
    }

    /// The command line that starts with the layout as it is now.
    fn command_line(&self) -> String {
        let settings = &self.settings;
        let mut words = vec![
            "qqcode-show".to_owned(),
            shell_word(&self.file.to_string_lossy()),
            format!(
                "--module {} --margin {} --level {}",
                settings.module, settings.margin, self.level
            ),
        ];
        if settings.max_version != MAX_VERSION {
            words.push(format!("--max-version {}", settings.max_version));
        }
        words.push(format!("--interval {}", self.interval.unwrap_or(0.0)));
        if settings.ink != Rgb::BLACK {
            words.push(format!("--ink {}", settings.ink));
        }
        if settings.paper != Rgb::WHITE {
            words.push(format!("--paper {}", settings.paper));
        }
        if !self.notes {
            words.push("--no-notes".to_owned());
        }
        words.join(" ")
    }

    /// What the terminal and the layout measured when they were last drawn.
    fn measurements(&self, size: &Size) -> String {
        let mut parts = vec![format!("terminal {}x{} cells", size.columns, size.rows)];
        if let Some((width, height)) = size.cell {
            parts.push(format!("cell {}x{} px", short(width), short(height)));
        }
        match &self.layout {
            Some(layout) => {
                parts.push(format!("version {}", layout.geometry.version));
                parts.push(code_size(&layout.geometry, size));
                parts.push(format!(
                    "{} a code, {}",
                    count(layout.piece, "byte"),
                    count(layout.codes, "code")
                ));
            }
            None => parts.push("too small for a code".to_owned()),
        }
        parts.join(", ")
    }
}

/// `1 code`, `2 codes`.
fn count(number: usize, of: &str) -> String {
    match number {
        1 => format!("1 {of}"),
        number => format!("{number} {of}s"),
    }
}

/// Moves `value` one step up or down, as far as `range` goes.
fn step(value: &mut usize, by: isize, range: RangeInclusive<usize>) {
    *value = value
        .saturating_add_signed(by)
        .clamp(*range.start(), *range.end());
}

/// `code 109x54.5 cells`, with the size in pixels if the terminal tells its cell size.
fn code_size(geometry: &Geometry, size: &Size) -> String {
    let dots = geometry.code_dots() as f64;
    let cells = format!("code {}x{} cells", dots, short(dots / 2.0));
    match size.cell {
        Some((width, height)) => format!("{cells} = {:.0}x{:.0} px", dots * width, dots * height / 2.0),
        None => cells,
    }
}

/// A number with one decimal, or none if that would be a 0.
fn short(value: f64) -> String {
    let text = format!("{value:.1}");
    text.strip_suffix(".0").map(str::to_owned).unwrap_or(text)
}

/// `text` as one word for a shell.
fn shell_word(text: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "_-./,:+=@%".contains(c);
    if !text.is_empty() && text.chars().all(plain) && !text.starts_with('-') {
        text.to_owned()
    } else {
        format!("'{}'", text.replace('\'', r"'\''"))
    }
}

/// Shows the chart and follows the keys until one of them quits; returns the size the terminal had then.
fn run(app: &mut App, terminal: &mut Terminal) -> Result<Size> {
    let next_turn = |app: &App| Some(Instant::now() + Duration::try_from_secs_f64(app.interval?).ok()?);
    let mut turn_at = next_turn(app);
    let mut size = terminal.size()?;
    let mut redraw = true;
    loop {
        if redraw {
            size = terminal.size()?;
            app.lay_out(size.columns, size.rows);
            terminal.draw(&app.screen(&size))?;
            redraw = false;
        }
        // the codes turn by themselves, the start screen and the end screen wait for a key
        if let Some(at) = turn_at
            && app.stage == Stage::Codes
            && !event::poll(at.saturating_duration_since(Instant::now()))?
        {
            app.forward();
            turn_at = next_turn(app);
            redraw = true;
            continue;
        }
        match event::read()? {
            Event::Key(key) if key.kind != KeyEventKind::Release => match app.on_key(key) {
                Reaction::Quit => return Ok(size),
                Reaction::Redraw => {
                    // a code turned by hand stays for a whole interval too
                    turn_at = next_turn(app);
                    redraw = true;
                }
                Reaction::Nothing => {}
            },
            Event::Resize(..) => redraw = true,
            _ => {}
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(page) = &cli.html {
        return web::write(&cli.file, page, Grid::new(cli.shape, cli.scale), cli.rate as u32);
    }
    let mut app = App::new(cli)?;
    let mut terminal = Terminal::open()?;
    let outcome = run(&mut app, &mut terminal);
    // back to the normal screen before anything is printed
    drop(terminal);
    let size = outcome?;
    println!("{}", app.command_line());
    println!("{}", app.measurements(&size));
    // to be compared with what qqcode prints when it has written the file
    println!("SHA-256 {}", app.sha256);
    println!("MD5 {}", app.md5);
    Ok(())
}

#[cfg(test)]
mod tests;
