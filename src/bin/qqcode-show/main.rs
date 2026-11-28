//! `qqcode-show` writes a web page that shows a file as a matrix of coloured tiles, as something to point a
//! camera at and have `qqcode` read back.

mod web;

use std::ops::RangeInclusive;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use clap::builder::RangedU64ValueParser;

use qqcode::tiles::{Grid, SCALES, Shape};

/// Writes a web page that shows a file, as something to point a camera at: a matrix of coloured tiles,
/// frame after frame without end, of which `qqcode` needs any that are enough to put the file together.
/// The page starts on a start screen, and every frame says which file it is of.
#[derive(Parser)]
#[command(version)]
struct Cli {
    /// File to show, of any kind
    file: PathBuf,

    /// File to write the page to; the checksums of the file go next to it, as NAME.checksums.txt
    #[arg(long, value_name = "FILE")]
    html: PathBuf,

    /// The shape of the matrix, best that of the camera's frames
    #[arg(long, value_enum, default_value_t = Shape::Wide)]
    shape: Shape,

    /// The size of the matrix, which has 9 rows of tiles for each; the fewer, the larger a tile is in the
    /// frames of the camera
    #[arg(long, default_value_t = 11, value_parser = within(SCALES), value_name = "SCALE")]
    scale: usize,

    /// Frames a second, best half of what the camera takes or less
    #[arg(long, default_value_t = 15, value_parser = within(1..=60), value_name = "FRAMES")]
    rate: usize,

    /// How many frames in a hundred the first round shows on top of those that make the file, for the
    /// ones a camera misses. What it leaves unfinished waits for the rounds after it, which are short
    #[arg(long, default_value_t = 10, value_parser = within(0..=400), value_name = "PERCENT")]
    spare: usize,

    /// Write a page that asks for the file when it is opened instead of holding it, as the page of a
    /// file of more than 64 MB always does
    #[arg(long)]
    ask: bool,
}

fn within(range: RangeInclusive<usize>) -> RangedU64ValueParser<usize> {
    RangedU64ValueParser::new().range(*range.start() as u64..=*range.end() as u64)
}

/// `1 frame`, `2 frames`.
fn count(number: usize, of: &str) -> String {
    match number {
        1 => format!("1 {of}"),
        number => format!("{number} {of}s"),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let format = web::Format {
        grid: Grid::new(cli.shape, cli.scale),
        rate: cli.rate as u32,
        spare: cli.spare,
    };
    web::write(&cli.file, &cli.html, format, cli.ask)
}

#[cfg(test)]
mod tests;
