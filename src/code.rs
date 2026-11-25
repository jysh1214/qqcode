//! Makes the QR codes.
//!
//! A code is made for a version and an error correction level that are given, not for the smallest version
//! its bytes fit into: all the codes of a show are of one size, which is the one that the terminal has room
//! for. The bytes go in as they are, in byte mode, so a code holds a known number of them whatever they are.

use std::fmt;

use anyhow::{Result, anyhow, ensure};
use qrcode::bits::Bits;
use qrcode::{Color, EcLevel, QrCode, Version};

/// The versions of a QR code are 1 to this.
pub const MAX_VERSION: usize = 40;

/// How much of a code can be misread before it can't be decoded, at the price of bytes it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Level {
    /// Some 7 % of the code may be misread; holds the most
    L,
    /// Some 15 %
    M,
    /// Some 25 %
    Q,
    /// Some 30 %; holds the least
    H,
}

impl Level {
    /// The level after this one, and the first after the last.
    pub fn next(self) -> Self {
        match self {
            Level::L => Level::M,
            Level::M => Level::Q,
            Level::Q => Level::H,
            Level::H => Level::L,
        }
    }

    fn correction(self) -> EcLevel {
        match self {
            Level::L => EcLevel::L,
            Level::M => EcLevel::M,
            Level::Q => EcLevel::Q,
            Level::H => EcLevel::H,
        }
    }
}

impl fmt::Display for Level {
    /// As the command line takes it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Level::L => "l",
            Level::M => "m",
            Level::Q => "q",
            Level::H => "h",
        })
    }
}

/// Modules along a side of a code of `version`: 21 for version 1, 177 for version 40.
pub fn side(version: usize) -> usize {
    17 + 4 * version
}

fn normal(version: usize) -> Version {
    assert!(
        (1..=MAX_VERSION).contains(&version),
        "QR codes have the versions 1 to {MAX_VERSION}, not {version}"
    );
    Version::Normal(version as i16)
}

/// How many bytes a code of `version` and `level` holds.
pub fn capacity(version: usize, level: Level) -> usize {
    let bits = Bits::new(normal(version))
        .max_len(level.correction())
        .expect("a version from 1 to 40");
    // before the bytes come four bits that say they are bytes, and their count in 8 or 16 bits
    let count = if version <= 9 { 8 } else { 16 };
    (bits - 4 - count) / 8
}

/// A QR code as a square grid of dark and light modules, without the blank space around it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    side: usize,
    dark: Vec<bool>,
}

impl Code {
    /// Modules along one side.
    pub fn side(&self) -> usize {
        self.side
    }

    /// Whether the module in `row` and `column`, counted from the top-left corner, is dark.
    pub fn is_dark(&self, row: usize, column: usize) -> bool {
        self.dark[row * self.side + column]
    }
}

/// The code of `version` and `level` that holds `bytes`.
pub fn encode(bytes: &[u8], version: usize, level: Level) -> Result<Code> {
    let room = capacity(version, level);
    ensure!(
        bytes.len() <= room,
        "{} bytes are more than the {room} that a code of version {version} holds at level {level}",
        bytes.len()
    );
    let refused = |error| anyhow!("can't make a code of version {version}: {error}");
    let mut bits = Bits::new(normal(version));
    bits.push_byte_data(bytes).map_err(refused)?;
    bits.push_terminator(level.correction()).map_err(refused)?;
    let code = QrCode::with_bits(bits, level.correction()).map_err(refused)?;
    Ok(Code {
        side: code.width(),
        dark: code
            .into_colors()
            .into_iter()
            .map(|colour| colour == Color::Dark)
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_that_of_the_standard() {
        // from the table of ISO/IEC 18004 for byte mode
        for (version, level, bytes) in [
            (1, Level::L, 17),
            (1, Level::H, 7),
            (9, Level::M, 180),
            (10, Level::L, 271),
            (10, Level::M, 213),
            (20, Level::M, 666),
            (25, Level::Q, 715),
            (40, Level::L, 2953),
            (40, Level::H, 1273),
        ] {
            assert_eq!(capacity(version, level), bytes, "version {version} {level}");
        }
    }

    #[test]
    fn code_has_the_size_of_its_version() -> Result<()> {
        for version in [1, 2, 7, 23, 40] {
            let code = encode(b"qqcode", version, Level::M)?;
            assert_eq!(code.side(), side(version));
            assert_eq!(code.side(), 17 + 4 * version);
        }
        Ok(())
    }

    #[test]
    fn code_has_its_three_finder_patterns() -> Result<()> {
        let code = encode(&[0, 255, 7], 3, Level::L)?;
        let far = code.side() - 7;
        for (top, left) in [(0, 0), (0, far), (far, 0)] {
            for at in 0..7 {
                // the dark ring
                for (row, column) in [(0, at), (6, at), (at, 0), (at, 6)] {
                    assert!(code.is_dark(top + row, left + column), "ring at {row},{column}");
                }
            }
            // the light ring inside it and the dark middle
            assert!(!code.is_dark(top + 1, left + 1) && !code.is_dark(top + 5, left + 3));
            assert!(code.is_dark(top + 3, left + 3));
        }
        // no finder pattern in the fourth corner
        let corner = (0..7).all(|at| code.is_dark(far + at, far) && code.is_dark(far, far + at));
        assert!(!corner);
        Ok(())
    }

    #[test]
    fn code_takes_as_many_bytes_as_it_holds_and_no_more() {
        let room = capacity(5, Level::Q);
        assert!(encode(&vec![0xa5; room], 5, Level::Q).is_ok());
        let error = encode(&vec![0xa5; room + 1], 5, Level::Q)
            .unwrap_err()
            .to_string();
        assert!(error.contains("version 5 holds at level q"), "{error}");
    }

    #[test]
    fn different_bytes_make_different_codes() -> Result<()> {
        assert_ne!(encode(b"a", 2, Level::M)?, encode(b"b", 2, Level::M)?);
        assert_ne!(encode(b"a", 2, Level::M)?, encode(b"a", 2, Level::H)?);
        assert_eq!(encode(b"a", 2, Level::M)?, encode(b"a", 2, Level::M)?);
        Ok(())
    }

    #[test]
    fn levels_come_round_and_print_as_they_are_typed() {
        let mut level = Level::L;
        let mut seen = String::new();
        for _ in 0..5 {
            seen.push_str(&level.to_string());
            level = level.next();
        }
        assert_eq!(seen, "lmqhl");
    }
}
