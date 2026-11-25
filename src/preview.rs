//! A live view for pointing the camera, part of the `qqcode` program only: what the camera sees, drawn on
//! the terminal out of half block characters, with the codes that are made out in it.
//!
//! It is drawn on the standard error, as the standard output is where the file goes.

use std::io::{self, IsTerminal, Write};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors, SetForegroundColor};
use crossterm::terminal::{
    self, BeginSynchronizedUpdate, Clear, ClearType, DisableLineWrap, EnableLineWrap, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, queue};
use opencv::core::{Mat, Point, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

use qqcode::{Kind, Sighting, Survey};

/// Rows below the picture: the code, the advice, the facts and the latest remark.
const TEXT_ROWS: usize = 4;

/// Colours of the marks in the picture, as blue, green, red.
const SHOWN: (u8, u8, u8) = (0, 230, 0);
const OTHER: (u8, u8, u8) = (255, 0, 255);
const UNREAD: (u8, u8, u8) = (0, 0, 255);

/// The terminal as a view of the camera, from [`Preview::open`] until this is dropped.
pub struct Preview {
    out: io::Stderr,
}

impl Preview {
    pub fn open() -> Result<Self> {
        let out = io::stderr();
        ensure!(
            out.is_terminal(),
            "--preview draws on the terminal through the standard error, which can't be redirected then"
        );
        terminal::enable_raw_mode().context("can't take over the terminal")?;
        // from here on `Drop` undoes what was done, also when the next step fails
        let mut preview = Self { out };
        execute!(preview.out, EnterAlternateScreen, DisableLineWrap, Hide)?;
        Ok(preview)
    }

    /// Shows a frame with what `survey` made out in it. `facts` and `remark` are lines of text to put
    /// below it.
    pub fn draw(&mut self, frame: &Mat, survey: &Survey, facts: &str, remark: &str) -> Result<()> {
        let window = terminal::window_size().context("can't ask the terminal for its size")?;
        let (columns, rows) = (window.columns as usize, window.rows as usize);
        let mut out = Vec::new();
        queue!(out, BeginSynchronizedUpdate, ResetColor, Clear(ClearType::All))?;
        let picture_rows = rows.saturating_sub(TEXT_ROWS);
        if columns > 0 && picture_rows > 0 {
            // a dot is one column wide and half a row high, and about square unless the terminal says so
            let dot_shape = if window.width > 0 && window.height > 0 {
                (window.width as f64 / columns as f64) / (window.height as f64 / rows as f64 / 2.0)
            } else {
                1.0
            };
            let size = picture_size(frame.size()?, columns, picture_rows, dot_shape);
            let picture = picture(frame, survey, size)?;
            let left = (columns - size.width as usize) / 2;
            encode_picture(&picture, left as u16, &mut out)?;
        }
        for (line, row) in status(survey, facts, remark).iter().zip(picture_rows..rows) {
            queue!(out, MoveTo(0, row as u16), ResetColor)?;
            let mut room = columns;
            for (colour, text) in line {
                let text = text.chars().take(room).collect::<String>();
                room -= text.chars().count();
                queue!(out, SetForegroundColor(*colour), Print(text))?;
            }
        }
        queue!(out, ResetColor, EndSynchronizedUpdate)?;
        self.out.write_all(&out)?;
        self.out.flush()?;
        Ok(())
    }

    /// Whether q, escape or control-c was pressed since the last call.
    pub fn quit_requested(&mut self) -> Result<bool> {
        let mut quit = false;
        while event::poll(Duration::ZERO)? {
            if let Event::Key(key) = event::read()?
                && key.kind != KeyEventKind::Release
            {
                let interrupt =
                    key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL);
                quit |= interrupt || matches!(key.code, KeyCode::Char('q') | KeyCode::Esc);
            }
        }
        Ok(quit)
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        // nothing more can be done about a terminal that doesn't take these
        let _ = execute!(self.out, ResetColor, EnableLineWrap, Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

/// How many dots (width and height) a frame is shown in: as large as `columns` by `rows` cells allow at
/// two dots a cell, in the shape of the frame. `dot_shape` is the width of a dot over its height.
fn picture_size(frame: Size, columns: usize, rows: usize, dot_shape: f64) -> Size {
    let (widest, highest) = (columns as f64, 2.0 * rows as f64);
    // dots across for every dot down
    let across = frame.width as f64 / frame.height as f64 / dot_shape;
    let height = highest.min(widest / across);
    // whole cells
    let height = ((height / 2.0).floor() * 2.0).max(2.0);
    let width = (height * across).round().clamp(1.0, widest);
    Size::new(width as i32, height as i32)
}

/// The colour that a code of this kind is marked in.
fn mark_of(kind: Kind) -> (u8, u8, u8) {
    match kind {
        Kind::Start | Kind::Piece | Kind::End => SHOWN,
        Kind::Other => OTHER,
        Kind::Unread => UNREAD,
    }
}

/// The frame at one pixel a dot, in three channels, with the outlines of the codes that the survey made
/// out.
fn picture(frame: &Mat, survey: &Survey, size: Size) -> Result<Mat> {
    let mut picture = Mat::default();
    imgproc::resize(frame, &mut picture, size, 0.0, 0.0, imgproc::INTER_AREA)?;
    if picture.channels() == 1 {
        let mut coloured = Mat::default();
        imgproc::cvt_color_def(&picture, &mut coloured, imgproc::COLOR_GRAY2BGR)?;
        picture = coloured;
    }
    let frame = frame.size()?;
    let scale = (
        size.width as f32 / frame.width as f32,
        size.height as f32 / frame.height as f32,
    );
    // what matters most goes on last
    let mut codes = survey.codes.iter().collect::<Vec<_>>();
    codes.sort_by_key(|code| match code.kind {
        Kind::Unread => 0,
        Kind::Other => 1,
        Kind::Start | Kind::Piece | Kind::End => 2,
    });
    for code in codes {
        let (blue, green, red) = mark_of(code.kind);
        let colour = Scalar::new(blue as f64, green as f64, red as f64, 0.0);
        let outline = &code.outline;
        for (from, to) in outline.iter().zip(outline.iter().cycle().skip(1)) {
            let [from, to] = [from, to].map(|point| {
                Point::new(
                    (point.x * scale.0).floor() as i32,
                    (point.y * scale.1).floor() as i32,
                )
            });
            imgproc::line(&mut picture, from, to, colour, 1, imgproc::LINE_8, 0)?;
        }
    }
    Ok(picture)
}

/// The escape sequences that draw `picture` from the top of the terminal, `left` columns in: two rows of
/// pixels to a row of cells, as an upper half block in the colour of the upper pixel on the colour of the
/// lower one.
fn encode_picture(picture: &Mat, left: u16, out: &mut impl Write) -> Result<()> {
    let (width, height) = (picture.cols() as usize, picture.rows() as usize);
    ensure!(
        picture.channels() == 3 && picture.is_continuous(),
        "a frame is expected in three channels"
    );
    let bytes = picture.data_bytes()?;
    let pixel = |column: usize, row: usize| {
        let at = (row * width + column) * 3;
        Color::Rgb {
            r: bytes[at + 2],
            g: bytes[at + 1],
            b: bytes[at],
        }
    };
    for row in 0..height / 2 {
        queue!(out, MoveTo(left, row as u16))?;
        let mut colours = None;
        for column in 0..width {
            let pair = (pixel(column, 2 * row), pixel(column, 2 * row + 1));
            if colours != Some(pair) {
                queue!(out, SetColors(Colors::new(pair.0, pair.1)))?;
                colours = Some(pair);
            }
            queue!(out, Print('▀'))?;
        }
    }
    Ok(())
}

fn colour((blue, green, red): (u8, u8, u8)) -> Color {
    Color::Rgb {
        r: red,
        g: green,
        b: blue,
    }
}

/// The lines below the picture, as pieces of text with their colours: what code is in view, what to do
/// about it, then `facts` and `remark`.
fn status(survey: &Survey, facts: &str, remark: &str) -> Vec<Vec<(Color, String)>> {
    let plain = Color::Reset;
    let code = match survey.leading() {
        Some(code) => (colour(mark_of(code.kind)), format!("code: {}", describe(code))),
        None => (Color::Rgb { r: 255, g: 90, b: 90 }, "code: -----".to_owned()),
    };
    let legend = [
        (plain, "q quits   marks: ".to_owned()),
        (colour(SHOWN), "code of qqcode-show  ".to_owned()),
        (colour(OTHER), "other QR code  ".to_owned()),
        (colour(UNREAD), "QR code that can't be read".to_owned()),
    ];
    let remark = if remark.is_empty() {
        legend.to_vec()
    } else {
        vec![(plain, remark.to_owned())]
    };
    vec![
        vec![code],
        vec![(plain, advice(survey).to_owned())],
        vec![(plain, facts.to_owned())],
        remark,
    ]
}

/// `start screen | 89 modules | 9.2 px a module`
fn describe(code: &Sighting) -> String {
    let kind = match code.kind {
        Kind::Start => "start screen",
        Kind::Piece => "piece of the file",
        Kind::End => "end screen",
        Kind::Other => "not one of qqcode-show",
        Kind::Unread => "can't be read",
    };
    match code.wrong {
        // a matrix of tiles
        Some(wrong) if code.kind != Kind::Unread => format!(
            "{kind} | {} tiles across | {:.1} px a tile | {:.1} % put right",
            code.modules,
            code.module_size,
            100.0 * wrong
        ),
        Some(_) => format!(
            "{kind} | {} tiles across | {:.1} px a tile",
            code.modules, code.module_size
        ),
        None => format!(
            "{kind} | {} modules | {:.1} px a module",
            code.modules, code.module_size
        ),
    }
}

/// What to do about what is in view.
fn advice(survey: &Survey) -> &'static str {
    match survey.leading().map(|code| code.kind) {
        Some(Kind::Start) => "start screen in view: enter in qqcode-show starts the codes",
        Some(Kind::Piece) => "a code of the file in view",
        Some(Kind::End) => "end screen in view",
        Some(Kind::Other) => "a QR code in view that is not one of qqcode-show",
        Some(Kind::Unread) => {
            "a code in view that can't be read: nearer, sharper, or larger (+ in qqcode-show, a smaller --scale)"
        }
        None => "no QR code in view",
    }
}

#[cfg(test)]
mod tests {
    use opencv::core::{CV_8UC1, CV_8UC3, Point2f, Vec3b};

    use super::*;

    /// A code that takes the square from `(x, y)` in a frame.
    fn code(kind: Kind, x: f32, y: f32, side: f32) -> Sighting {
        Sighting {
            outline: [(x, y), (x + side, y), (x + side, y + side), (x, y + side)]
                .map(|(x, y)| Point2f::new(x, y)),
            kind,
            modules: 89,
            module_size: side / 89.0,
            wrong: None,
        }
    }

    #[test]
    fn picture_fills_the_terminal_in_the_shape_of_the_frame() {
        let camera = Size::new(1280, 960);
        // 36 rows are 72 dots down, which makes 96 across for 4:3
        assert_eq!(picture_size(camera, 120, 36, 1.0), Size::new(96, 72));
        // a narrow terminal limits the width instead: 60 across, 45 down, of which 44 are whole cells
        assert_eq!(picture_size(camera, 60, 36, 1.0), Size::new(59, 44));
        // dots that are narrower than high need more of them across
        assert_eq!(picture_size(camera, 120, 36, 0.8), Size::new(120, 72));
        assert_eq!(
            picture_size(Size::new(1920, 1080), 200, 50, 1.0),
            Size::new(178, 100)
        );
        // never nothing
        assert_eq!(picture_size(camera, 1, 1, 1.0), Size::new(1, 2));
    }

    #[test]
    fn codes_are_outlined_in_the_picture() -> Result<()> {
        let frame = Mat::new_rows_cols_with_default(960, 1280, CV_8UC3, Scalar::all(128.0))?;
        let survey = Survey {
            codes: vec![
                code(Kind::Piece, 100.0, 100.0, 400.0),
                code(Kind::Other, 700.0, 100.0, 200.0),
                code(Kind::Unread, 700.0, 500.0, 200.0),
            ],
        };
        // a tenth of the size
        let picture = picture(&frame, &survey, Size::new(128, 96))?;
        assert_eq!(picture.size()?, Size::new(128, 96));
        let at = |x: i32, y: i32| -> (u8, u8, u8) {
            let pixel = picture.at_2d::<Vec3b>(y, x).unwrap();
            (pixel[0], pixel[1], pixel[2])
        };
        let grey = (128, 128, 128);
        // the code of the show, from (10, 10) to (50, 50), and nothing inside it
        assert_eq!([at(10, 10), at(30, 10), at(50, 30), at(10, 50)], [SHOWN; 4]);
        assert_eq!(at(30, 30), grey);
        assert_eq!([at(70, 10), at(90, 20)], [OTHER; 2]);
        assert_eq!([at(70, 50), at(80, 70)], [UNREAD; 2]);
        Ok(())
    }

    #[test]
    fn grey_frame_is_shown_as_well() -> Result<()> {
        let frame = Mat::new_rows_cols_with_default(96, 128, CV_8UC1, Scalar::all(128.0))?;
        let picture = picture(&frame, &Survey::default(), Size::new(64, 48))?;
        assert_eq!(picture.channels(), 3);
        encode_picture(&picture, 0, &mut Vec::new())?;
        Ok(())
    }

    #[test]
    fn picture_becomes_half_blocks() -> Result<()> {
        // two columns by two rows of cells; blue, green, red
        let mut picture = Mat::new_rows_cols_with_default(4, 2, CV_8UC3, Scalar::new(1.0, 2.0, 3.0, 0.0))?;
        *picture.at_2d_mut::<Vec3b>(1, 0)? = Vec3b::from([10, 20, 30]);
        let mut out = Vec::new();
        encode_picture(&picture, 7, &mut out)?;
        assert_eq!(
            String::from_utf8(out)?,
            "\x1b[1;8H\x1b[38;2;3;2;1;48;2;30;20;10m▀\x1b[38;2;3;2;1;48;2;3;2;1m▀\
             \x1b[2;8H\x1b[38;2;3;2;1;48;2;3;2;1m▀▀"
        );
        Ok(())
    }

    fn texts(lines: &[Vec<(Color, String)>]) -> Vec<String> {
        let text = |line: &Vec<(Color, String)>| line.iter().map(|(_, text)| text.as_str()).collect();
        lines.iter().map(text).collect()
    }

    #[test]
    fn status_names_the_code_and_what_to_do() {
        let nothing = Survey::default();
        let lines = texts(&status(&nothing, "1920x1080", ""));
        assert_eq!(lines[0], "code: -----");
        assert_eq!(lines[1], "no QR code in view");
        assert_eq!(lines[2], "1920x1080");
        assert!(
            lines[3].starts_with("q quits   marks: code of qqcode-show"),
            "{}",
            lines[3]
        );

        let start = Survey {
            codes: vec![code(Kind::Start, 0.0, 0.0, 801.0)],
        };
        let lines = texts(&status(&start, "", "frame 12: start screen in view"));
        assert_eq!(lines[0], "code: start screen | 89 modules | 9.0 px a module");
        assert_eq!(
            lines[1],
            "start screen in view: enter in qqcode-show starts the codes"
        );
        assert_eq!(lines[3], "frame 12: start screen in view");
        // a code that is read and one that is not differ in colour
        let unread = Survey {
            codes: vec![code(Kind::Unread, 0.0, 0.0, 300.0)],
        };
        assert_ne!(status(&start, "", "")[0][0].0, status(&unread, "", "")[0][0].0);
        assert!(advice(&unread).contains("or larger"));
        // a matrix of tiles says how much of it had to be put right
        let matrix = Sighting {
            wrong: Some(0.0123),
            modules: 176,
            module_size: 9.04,
            ..code(Kind::Piece, 0.0, 0.0, 1591.0)
        };
        assert_eq!(
            describe(&matrix),
            "piece of the file | 176 tiles across | 9.0 px a tile | 1.2 % put right"
        );
    }

    #[test]
    fn status_goes_by_the_code_of_the_show_among_several() {
        let survey = Survey {
            codes: vec![
                code(Kind::Unread, 0.0, 0.0, 100.0),
                code(Kind::Other, 200.0, 0.0, 100.0),
                code(Kind::End, 400.0, 0.0, 100.0),
            ],
        };
        assert_eq!(advice(&survey), "end screen in view");
        let others = Survey {
            codes: survey.codes[..2].to_vec(),
        };
        assert_eq!(
            advice(&others),
            "a QR code in view that is not one of qqcode-show"
        );
        let unread = Survey {
            codes: survey.codes[..1].to_vec(),
        };
        assert!(advice(&unread).starts_with("a code in view that can't be read"));
    }
}
