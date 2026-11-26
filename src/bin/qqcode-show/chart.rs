//! What goes on the terminal: a QR code as large as there is room for, and beside or below it a panel of
//! text for whoever sits at the terminal.
//!
//! The picture is made of dots of half a character cell each, one column wide and half a row high. A
//! terminal can draw those with the half block characters, and in the usual fonts they are about square. A
//! module of the code is a square of dots, so the code comes out about square as well.

use std::fmt;
use std::ops::Range;
use std::str::FromStr;

use qqcode::code::{self, Code, MAX_VERSION};

use crate::text::wrap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rgb(pub u8, pub u8, pub u8);

impl Rgb {
    pub const BLACK: Rgb = Rgb(0, 0, 0);
    pub const WHITE: Rgb = Rgb(255, 255, 255);

    /// Perceived brightness, 0 to 255.
    pub fn luma(self) -> f32 {
        0.299 * self.0 as f32 + 0.587 * self.1 as f32 + 0.114 * self.2 as f32
    }

    /// The colour that lies `share` (0 to 1) of the way from this one to `other`.
    pub fn towards(self, other: Rgb, share: f32) -> Rgb {
        let mix = |from: u8, to: u8| (from as f32 + (to as f32 - from as f32) * share).round() as u8;
        Rgb(mix(self.0, other.0), mix(self.1, other.1), mix(self.2, other.2))
    }
}

impl FromStr for Rgb {
    type Err = String;

    /// `RRGGBB` in hexadecimal, with or without a `#` in front.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let digits = text.strip_prefix('#').unwrap_or(text);
        if digits.len() != 6 || !digits.bytes().all(|digit| digit.is_ascii_hexdigit()) {
            return Err(format!(
                "expected a colour as RRGGBB such as d0d0d0, got {text:?}"
            ));
        }
        let channel = |at: usize| u8::from_str_radix(&digits[at..at + 2], 16).expect("two hex digits");
        Ok(Rgb(channel(0), channel(2), channel(4)))
    }
}

impl fmt::Display for Rgb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }
}

/// How the chart is laid out and coloured; the command line options and keys of the same names set these.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Side of a module of the code in dots: this many columns, half as many rows
    pub module: usize,
    /// Paper on every side of the code, in modules. A QR code is to have four of them.
    pub margin: usize,
    /// The largest version a code may have, however much room there is
    pub max_version: usize,
    /// The dark colour: the modules and the text
    pub ink: Rgb,
    /// The light colour: everything else
    pub paper: Rgb,
}

/// A panel beside the code is at least this wide; a terminal that leaves less has it below the code.
const PANEL_COLUMNS: usize = 28;

/// A panel below the code is at least this high.
const PANEL_ROWS: usize = 6;

/// A rectangle of character cells: the columns `left..right` of the rows `top..bottom`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cells {
    pub left: usize,
    pub top: usize,
    pub right: usize,
    pub bottom: usize,
}

impl Cells {
    pub fn width(&self) -> usize {
        self.right - self.left
    }

    /// The rectangle less `columns` on its left and right and `rows` at its top and bottom, if that leaves
    /// anything.
    fn inset(&self, columns: usize, rows: usize) -> Option<Cells> {
        let inner = Cells {
            left: self.left + columns,
            top: self.top + rows,
            right: self.right.checked_sub(columns)?,
            bottom: self.bottom.checked_sub(rows)?,
        };
        (inner.left < inner.right && inner.top < inner.bottom).then_some(inner)
    }
}

/// Where everything is on a terminal of a given size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub columns: usize,
    pub rows: usize,
    /// Side of a module in dots
    module: usize,
    /// The version of the codes: the largest that there is room for
    pub version: usize,
    /// The dot at the top-left of the code
    origin: (usize, usize),
    /// Where the text goes, if there is room for any
    pub panel: Option<Cells>,
}

impl Geometry {
    /// `None` if a terminal of `columns` by `rows` has no room for a code of version 1.
    pub fn new(columns: usize, rows: usize, settings: &Settings) -> Option<Self> {
        // The code is a square and the panel takes what is left of the terminal, on the side where that
        // leaves the larger square. These are the sides of the two squares in dots.
        let beside = columns.saturating_sub(PANEL_COLUMNS).min(2 * rows);
        let below = (2 * rows.saturating_sub(PANEL_ROWS)).min(columns);
        let square = beside.max(below);

        let modules = (square / settings.module).checked_sub(2 * settings.margin)?;
        let version = (modules.checked_sub(code::side(0))? / 4).min(settings.max_version.min(MAX_VERSION));
        if version == 0 {
            return None;
        }
        let code = code::side(version) * settings.module;
        // the code goes into the middle of its square, and the square to the top-left of the terminal
        let whole = Cells {
            left: 0,
            top: 0,
            right: columns,
            bottom: rows,
        };
        let (origin, panel) = if beside >= below {
            let origin = ((square - code) / 2, (2 * rows - code) / 2);
            let panel = Cells {
                left: square,
                ..whole
            };
            (origin, panel.inset(1, 1))
        } else {
            let band = square.div_ceil(2);
            let origin = ((columns - code) / 2, (2 * band - code) / 2);
            let panel = Cells { top: band, ..whole };
            (origin, panel.inset(2, 0))
        };
        Some(Self {
            columns,
            rows,
            module: settings.module,
            version,
            origin,
            panel,
        })
    }

    /// Side of the code in dots, which is its width in columns and twice its height in rows.
    pub fn code_dots(&self) -> usize {
        code::side(self.version) * self.module
    }
}

/// A piece of text at a place on the screen.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub column: usize,
    pub row: usize,
    pub text: String,
    pub colour: Rgb,
    pub background: Rgb,
}

/// A terminal's worth of picture: a colour for every dot, and text to write on top of that.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Screen {
    pub columns: usize,
    pub rows: usize,
    /// Row by row, two rows of dots for every row of cells
    dots: Vec<Rgb>,
    pub runs: Vec<Run>,
}

impl Screen {
    fn new(columns: usize, rows: usize, colour: Rgb) -> Self {
        Self {
            columns,
            rows,
            dots: vec![colour; columns * rows * 2],
            runs: Vec::new(),
        }
    }

    /// The colour of the dot in column `x` and half-row `y`.
    pub fn dot(&self, x: usize, y: usize) -> Rgb {
        self.dots[y * self.columns + x]
    }

    fn fill_dots(&mut self, xs: Range<usize>, ys: Range<usize>, colour: Rgb) {
        for y in ys {
            let row = y * self.columns;
            self.dots[row + xs.start..row + xs.end].fill(colour);
        }
    }

    /// Writes `texts` into `cells` from the top, each broken into as many rows as it needs, as far as
    /// there are rows. Returns the cells that are left below.
    fn write(&mut self, cells: Cells, texts: &[String], colour: Rgb, background: Rgb) -> Option<Cells> {
        let mut row = cells.top;
        for line in texts.iter().flat_map(|text| wrap(text, cells.width())) {
            if row == cells.bottom {
                break;
            }
            if !line.is_empty() {
                self.runs.push(Run {
                    column: cells.left,
                    row,
                    text: line,
                    colour,
                    background,
                });
            }
            row += 1;
        }
        Cells { top: row, ..cells }.inset(0, 0)
    }

    /// A screen that only says something, for when there is no room for the chart.
    pub fn notice(columns: usize, rows: usize, settings: &Settings, lines: &[String]) -> Self {
        let mut screen = Self::new(columns, rows, settings.paper);
        let everything = Cells {
            left: 0,
            top: 0,
            right: columns,
            bottom: rows,
        };
        if let Some(cells) = everything.inset(1, 1) {
            screen.write(cells, lines, settings.ink, settings.paper);
        }
        screen
    }
}

/// The chart: `code`, which has to be of the version that `geometry` is for, and in the panel the
/// `message` of the screen and then, fainter, as many of the `notes` as there is room for.
pub fn compose(
    geometry: &Geometry,
    settings: &Settings,
    code: &Code,
    message: &[String],
    notes: &[String],
) -> Screen {
    assert_eq!(
        code.side(),
        code::side(geometry.version),
        "a code of the version there is room for"
    );
    let mut screen = Screen::new(geometry.columns, geometry.rows, settings.paper);
    let module = geometry.module;
    let (x, y) = geometry.origin;
    for row in 0..code.side() {
        for column in 0..code.side() {
            if code.is_dark(row, column) {
                let (left, top) = (x + column * module, y + row * module);
                screen.fill_dots(left..left + module, top..top + module, settings.ink);
            }
        }
    }

    if let Some(panel) = geometry.panel {
        let mut rest = screen.write(panel, message, settings.ink, settings.paper);
        if !message.is_empty() {
            // an empty row between the two
            rest = rest.and_then(|cells| {
                Cells {
                    top: cells.top + 1,
                    ..cells
                }
                .inset(0, 0)
            });
        }
        if let Some(cells) = rest {
            // plain to read from the chair, faint from where the camera is
            let faint = settings.paper.towards(settings.ink, 0.55);
            screen.write(cells, notes, faint, settings.paper);
        }
    }
    screen
}

#[cfg(test)]
pub mod tests {
    use qqcode::code::Level;

    use super::*;

    pub fn settings() -> Settings {
        Settings {
            module: 1,
            margin: 4,
            max_version: MAX_VERSION,
            ink: Rgb::BLACK,
            paper: Rgb::WHITE,
        }
    }

    fn strings(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|text| (*text).to_owned()).collect()
    }

    #[test]
    fn colours_are_six_hex_digits() {
        assert_eq!("d0D0d0".parse(), Ok(Rgb(208, 208, 208)));
        assert_eq!("#0a141e".parse(), Ok(Rgb(10, 20, 30)));
        assert_eq!(Rgb(10, 20, 30).to_string(), "0a141e");
        for wrong in ["", "fff", "#fffff", "+fffff", "gggggg", "ffffff0"] {
            assert!(wrong.parse::<Rgb>().is_err(), "{wrong:?}");
        }
    }

    #[test]
    fn wide_terminal_has_the_panel_beside_the_code() {
        // 24 rows are 48 dots, which leave 32 of the 80 columns: the code gets a square of 48 dots, less
        // 4 modules of margin on either side 40, into which a version 5 of 37 modules fits
        let geometry = Geometry::new(80, 24, &settings()).unwrap();
        assert_eq!((geometry.version, geometry.code_dots()), (5, 37));
        assert_eq!(geometry.origin, (5, 5));
        assert_eq!(
            geometry.panel,
            Some(Cells {
                left: 49,
                top: 1,
                right: 79,
                bottom: 23
            })
        );
        // a very wide one has the same code and more panel
        let wide = Geometry::new(200, 24, &settings()).unwrap();
        assert_eq!((wide.version, wide.origin), (5, (5, 5)));
        assert_eq!(wide.panel.unwrap().width(), 150);
    }

    #[test]
    fn narrow_terminal_has_the_panel_below_the_code() {
        // beside the code there would be 22 columns for it: below it, the code can be 50 dots
        let geometry = Geometry::new(50, 40, &settings()).unwrap();
        assert_eq!((geometry.version, geometry.code_dots()), (6, 41));
        assert_eq!(geometry.origin, (4, 4));
        assert_eq!(
            geometry.panel,
            Some(Cells {
                left: 2,
                top: 25,
                right: 48,
                bottom: 40
            })
        );
        // a terminal that is as high as wide in dots gives up six rows for the panel
        let square = Geometry::new(100, 50, &settings()).unwrap();
        assert_eq!(square.version, 15);
        assert_eq!(square.panel.unwrap().top, 44);
    }

    #[test]
    fn code_is_as_large_as_the_settings_allow() {
        let version = |columns, rows, settings: &Settings| {
            Geometry::new(columns, rows, settings).map(|geometry| geometry.version)
        };
        // 120 dots: 177 modules and their margin don't fit, 109 do
        assert_eq!(version(240, 60, &settings()), Some(23));
        assert_eq!(version(500, 100, &settings()), Some(40));
        let mut larger = settings();
        larger.module = 2;
        assert_eq!(version(240, 60, &larger), Some(8));
        larger.margin = 1;
        assert_eq!(version(240, 60, &larger), Some(10));
        let mut capped = settings();
        capped.max_version = 12;
        assert_eq!(version(240, 60, &capped), Some(12));
        // 21 modules and 8 of margin are 29 dots, which 15 rows have and 14 don't
        assert_eq!(version(80, 15, &settings()), Some(1));
        assert_eq!(version(80, 14, &settings()), None);
        assert_eq!(version(10, 3, &settings()), None);
        assert_eq!(version(0, 0, &settings()), None);
    }

    #[test]
    fn code_is_drawn_module_by_module_inside_its_margin() {
        let mut settings = settings();
        settings.module = 2;
        settings.margin = 2;
        settings.ink = Rgb(1, 2, 3);
        let geometry = Geometry::new(100, 30, &settings).unwrap();
        // 60 dots are 30 modules, less the margin 26: version 2 with its 25
        assert_eq!((geometry.version, geometry.origin), (2, (5, 5)));
        let code = code::encode(b"qqcode", 2, Level::M).unwrap();
        let screen = compose(&geometry, &settings, &code, &[], &[]);
        for row in 0..25 {
            for column in 0..25 {
                let colour = if code.is_dark(row, column) {
                    settings.ink
                } else {
                    Rgb::WHITE
                };
                for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let dot = screen.dot(5 + 2 * column + x, 5 + 2 * row + y);
                    assert_eq!(dot, colour, "module {row},{column}");
                }
            }
        }
        // nothing but paper around it
        for at in 0..60 {
            for (x, y) in [(at, 4), (at, 55), (4, at), (55, at)] {
                assert_eq!(screen.dot(x, y), Rgb::WHITE);
            }
        }
        assert!(screen.runs.is_empty());
    }

    #[test]
    fn panel_has_the_message_and_then_the_notes() {
        let settings = settings();
        let geometry = Geometry::new(80, 24, &settings).unwrap();
        let code = code::encode(b"qqcode", 5, Level::M).unwrap();
        let message = strings(&["Ready to show a file that has a long name.", "", "Enter starts."]);
        let notes = strings(&["start", "q quit"]);
        let screen = compose(&geometry, &settings, &code, &message, &notes);
        let shown = |run: &Run| (run.column, run.row, run.text.clone());
        assert_eq!(
            screen.runs.iter().map(shown).collect::<Vec<_>>(),
            [
                (49, 1, "Ready to show a file that has".to_owned()),
                (49, 2, "a long name.".to_owned()),
                (49, 4, "Enter starts.".to_owned()),
                (49, 6, "start".to_owned()),
                (49, 7, "q quit".to_owned()),
            ]
        );
        // the message in ink, the notes fainter but darker than the paper
        assert_eq!(screen.runs[0].colour, Rgb::BLACK);
        let faint = screen.runs[4].colour;
        assert!(faint.luma() > 60.0 && faint.luma() < 200.0, "{faint}");
        // without a message the notes begin at the top
        let screen = compose(&geometry, &settings, &code, &[], &notes);
        assert_eq!(shown(&screen.runs[0]), (49, 1, "start".to_owned()));
    }

    #[test]
    fn text_that_has_no_room_is_left_out() {
        let settings = settings();
        let geometry = Geometry::new(50, 40, &settings).unwrap();
        let code = code::encode(b"qqcode", 6, Level::M).unwrap();
        let many = vec!["line".to_owned(); 40];
        let screen = compose(&geometry, &settings, &code, &many, &many);
        // the panel has the rows 25 to 39
        assert_eq!(screen.runs.len(), 15);
        assert_eq!(screen.runs.last().unwrap().row, 39);
    }

    #[test]
    fn notice_says_its_lines() {
        let screen = Screen::notice(30, 5, &settings(), &strings(&["too small", "press -"]));
        assert_eq!(screen.runs.len(), 2);
        assert_eq!((screen.runs[0].column, screen.runs[0].row), (1, 1));
        assert_eq!(screen.dot(0, 0), Rgb::WHITE);
    }

    #[test]
    fn panel_stays_beside_the_code_where_below_would_leave_as_much() {
        // 64 columns less the panel are 36 dots, and so are 24 rows less the panel: a square of 36 either
        // way, into which a version 2 of 25 modules fits with its margin
        let geometry = Geometry::new(64, 24, &settings()).unwrap();
        assert_eq!((geometry.version, geometry.origin), (2, (5, 11)));
        assert_eq!(
            geometry.panel,
            Some(Cells {
                left: 37,
                top: 1,
                right: 63,
                bottom: 23
            })
        );
    }

    #[test]
    fn code_above_the_panel_takes_whole_rows() {
        // A narrow terminal: 41 columns are a square of 41 dots, which is 20 rows of cells and a half.
        // The code gets 21 of them, a version 4 of 33 modules in their middle, and the panel begins
        // below that.
        let geometry = Geometry::new(41, 60, &settings()).unwrap();
        assert_eq!((geometry.version, geometry.origin), (4, (4, 4)));
        assert_eq!(
            geometry.panel,
            Some(Cells {
                left: 2,
                top: 21,
                right: 39,
                bottom: 60
            })
        );
    }

    #[test]
    fn code_is_drawn_where_it_belongs_when_that_is_further_down_than_across() {
        // the code is in the middle of the terminal's height and of a square that is less wide than that
        let settings = settings();
        let geometry = Geometry::new(64, 24, &settings).unwrap();
        assert_eq!((geometry.version, geometry.origin), (2, (5, 11)));
        let code = code::encode(b"qqcode", 2, Level::M).unwrap();
        let screen = compose(&geometry, &settings, &code, &[], &[]);
        for row in 0..25 {
            for column in 0..25 {
                let colour = if code.is_dark(row, column) {
                    settings.ink
                } else {
                    Rgb::WHITE
                };
                assert_eq!(screen.dot(5 + column, 11 + row), colour, "module {row},{column}");
            }
        }
    }
}
