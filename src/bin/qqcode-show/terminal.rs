//! The terminal itself: taking it over for as long as a [`Terminal`] lives, and drawing a [`Screen`] on it.

use std::io::{self, IsTerminal, Write};

use anyhow::{Context, Result, ensure};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::style::{Color, Colors, Print, ResetColor, SetColors};
use crossterm::terminal::{
    self, BeginSynchronizedUpdate, DisableLineWrap, EnableLineWrap, EndSynchronizedUpdate,
    EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, queue};

use crate::chart::{Rgb, Screen};

/// How large the terminal is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Size {
    pub columns: usize,
    pub rows: usize,
    /// Width and height of a character cell in pixels, if the terminal tells
    pub cell: Option<(f64, f64)>,
}

/// The terminal in raw mode and on its alternate screen; dropping this gives it back as it was.
pub struct Terminal {
    out: io::Stdout,
}

impl Terminal {
    pub fn open() -> Result<Self> {
        let out = io::stdout();
        ensure!(
            out.is_terminal(),
            "the chart is drawn on the terminal, so the output can't be redirected"
        );
        terminal::enable_raw_mode().context("can't take over the terminal")?;
        // from here on `Drop` undoes what was done, also when the next step fails
        let mut terminal = Self { out };
        execute!(terminal.out, EnterAlternateScreen, DisableLineWrap, Hide)?;
        Ok(terminal)
    }

    pub fn size(&self) -> Result<Size> {
        let window = terminal::window_size().context("can't ask the terminal for its size")?;
        let (columns, rows) = (window.columns as usize, window.rows as usize);
        let has_pixels = window.width > 0 && window.height > 0 && columns > 0 && rows > 0;
        Ok(Size {
            columns,
            rows,
            cell: has_pixels.then(|| {
                (
                    window.width as f64 / columns as f64,
                    window.height as f64 / rows as f64,
                )
            }),
        })
    }

    pub fn draw(&mut self, screen: &Screen) -> Result<()> {
        // one write for the whole picture
        let mut frame = Vec::new();
        encode(screen, &mut frame)?;
        self.out.write_all(&frame)?;
        self.out.flush()?;
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        // nothing more can be done about a terminal that doesn't take these
        let _ = execute!(self.out, ResetColor, EnableLineWrap, Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

impl From<Rgb> for Color {
    fn from(Rgb(r, g, b): Rgb) -> Self {
        Color::Rgb { r, g, b }
    }
}

/// The escape sequences that draw `screen`. A cell shows its two dots as an upper half block in the colour
/// of the upper dot on a background in the colour of the lower one, or as a space if they are the same.
fn encode(screen: &Screen, out: &mut impl Write) -> io::Result<()> {
    queue!(out, BeginSynchronizedUpdate)?;
    for row in 0..screen.rows {
        queue!(out, MoveTo(0, row as u16))?;
        let mut colours = None;
        for column in 0..screen.columns {
            let dots = (screen.dot(column, 2 * row), screen.dot(column, 2 * row + 1));
            if colours != Some(dots) {
                queue!(out, SetColors(Colors::new(dots.0.into(), dots.1.into())))?;
                colours = Some(dots);
            }
            queue!(out, Print(if dots.0 == dots.1 { ' ' } else { '▀' }))?;
        }
    }
    for run in &screen.runs {
        queue!(
            out,
            MoveTo(run.column as u16, run.row as u16),
            SetColors(Colors::new(run.colour.into(), run.background.into())),
            Print(&run.text)
        )?;
    }
    queue!(out, ResetColor, EndSynchronizedUpdate)
}

#[cfg(test)]
mod tests {
    use qqcode::code::{self, Level};

    use super::*;
    use crate::chart::{Geometry, Settings, compose};

    #[test]
    fn screen_becomes_half_blocks_and_text() {
        let settings = Settings {
            module: 1,
            margin: 1,
            max_version: 1,
            ink: Rgb(1, 2, 3),
            paper: Rgb(250, 251, 252),
        };
        // a code of 21 dots in a square of 24, so from the second dot on in either direction
        let geometry = Geometry::new(60, 12, &settings).unwrap();
        let code = code::encode(b"qqcode", 1, Level::M).unwrap();
        let mut frame = Vec::new();
        encode(
            &compose(&geometry, &settings, &code, &["Hello".to_owned()], &[]),
            &mut frame,
        )
        .unwrap();
        let frame = String::from_utf8(frame).unwrap();

        let (ink, paper) = ("1;2;3", "250;251;252");
        // the first row of cells is paper above the top of a finder pattern, which is seven modules wide
        assert!(
            frame.contains(&format!(
                "\x1b[1;1H\x1b[38;2;{paper};48;2;{paper}m \x1b[38;2;{paper};48;2;{ink}m▀▀▀▀▀▀▀\x1b[38;2;{paper};48;2;{paper}m"
            )),
            "{frame:?}"
        );
        // every row is addressed, none relies on the line before it wrapping
        for row in 1..=12 {
            assert!(frame.contains(&format!("\x1b[{row};1H")), "row {row}");
        }
        // the text: at its place in the panel beside the code, ink on paper
        assert!(
            frame.contains(&format!("\x1b[2;26H\x1b[38;2;{ink};48;2;{paper}mHello")),
            "{frame:?}"
        );
        assert!(
            frame.starts_with("\x1b[?2026h") && frame.ends_with("\x1b[?2026l"),
            "{frame:?}"
        );
    }
}
