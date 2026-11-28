//! The checksums of a file, section by section, as a text that goes ahead of the file.
//!
//! `qqcode-show --html` writes it next to the page. `qqcode --check` reads it, takes no other file than
//! the one it names, and holds every section of that file against it: on disk as soon as a section is
//! whole ([`crate::stored`]), in memory when the file is. What a camera misread and the check of a frame
//! let through is found that way where it is, rather than at the end and somewhere in the file.
//!
//! The sections are of the file and of nothing else: a megabyte each, whatever the size of the matrix
//! that shows it.
//!
//! ```text
//! qqcode checksums 1
//! file notes.tar.gz
//! length 2500000
//! crc32 1c291ca3
//! sha256 the SHA-256 of the whole file
//! section 1048576
//! the SHA-256 of the bytes from 0 on, two spaces and 0
//! the SHA-256 of the bytes from 1048576 on, two spaces and 1048576
//! the SHA-256 of the bytes from 2097152 on, two spaces and 2097152
//! ```
//!
//! The text is of use only as far as it is right itself, so both programs print its SHA-256.

use std::fmt;
use std::ops::Range;

use sha2::{Digest, Sha256};

use crate::digest;
use crate::packet::Origin;

/// How many bytes of a file a checksum is of, in the texts that are written here.
pub const SECTION: u64 = 1 << 20;

/// What the text begins with; the number is that of the layout.
const TITLE: &str = "qqcode checksums 1";

/// A SHA-256.
pub type Hash = [u8; 32];

fn hash(bytes: &[u8]) -> Hash {
    let mut hash = [0; 32];
    hash.copy_from_slice(&Sha256::digest(bytes));
    hash
}

/// The checksums of one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checks {
    /// What the file was called where the checksums were made, for whoever reads them
    pub name: String,
    /// The length of the file and its CRC-32, by which the codes and the frames name it
    pub origin: Origin,
    /// The SHA-256 of the whole file in hexadecimal, as `sha256sum` prints it
    pub sha256: String,
    /// How many bytes a section has; the last one has what is left
    pub section: u64,
    /// The SHA-256 of each section
    hashes: Vec<Hash>,
}

impl Checks {
    /// The checksums of a file that is at hand, in sections of [`SECTION`].
    pub fn of(name: &str, file: &[u8]) -> Self {
        Self::in_sections(name, file, SECTION)
    }

    /// The same in sections of another size.
    pub fn in_sections(name: &str, file: &[u8], section: u64) -> Self {
        let mut maker = Maker::new(section);
        maker.update(file);
        maker.finish(name, Origin::of(file), &digest::sha256(file))
    }

    /// How many sections the file has.
    pub fn sections(&self) -> usize {
        self.hashes.len()
    }

    /// The bytes of the file that a section is.
    pub fn range(&self, section: usize) -> Range<u64> {
        let from = (section as u64 * self.section).min(self.origin.length);
        from..(from + self.section).min(self.origin.length)
    }

    /// Whether `bytes` are those of a section.
    pub fn holds(&self, section: usize, bytes: &[u8]) -> bool {
        self.hashes.get(section) == Some(&hash(bytes))
    }

    /// The sections of a file that are not what they should be. A file of another length has none that
    /// tell, so that is for whoever asks to see to.
    pub fn wrong(&self, file: &[u8]) -> Vec<usize> {
        (0..self.sections())
            .filter(|section| {
                let range = self.range(*section);
                let bytes = file.get(range.start as usize..range.end as usize);
                !bytes.is_some_and(|bytes| self.holds(*section, bytes))
            })
            .collect()
    }

    /// The checksums that a text holds, or what is wrong with it.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut lines = text
            .lines()
            .map(str::trim_end)
            .enumerate()
            .filter(|(_, line)| !line.is_empty())
            .map(|(number, line)| (number + 1, line));
        match lines.next() {
            Some((_, TITLE)) => {}
            Some((_, line)) if line.starts_with("qqcode checksums ") => {
                return Err(format!("{line:?} is a layout that this qqcode does not know"));
            }
            _ => return Err(format!("it does not begin with the line {TITLE:?}")),
        }
        let mut field = |key: &str| match lines.next() {
            Some((_, line)) if line == key => Ok(String::new()),
            Some((number, line)) => match line.split_once(' ') {
                Some((found, value)) if found == key => Ok(value.to_owned()),
                _ => Err(format!("line {number} is to begin with {key:?}")),
            },
            None => Err(format!("it ends before the line of {key:?}")),
        };
        let name = printable(&field("file")?);
        let number = |key: &str, value: String| {
            value
                .parse::<u64>()
                .map_err(|_| format!("{value:?} is no number for {key:?}"))
        };
        let length = number("length", field("length")?)?;
        let checksum = field("crc32")?;
        let checksum = Some(checksum.as_str())
            .filter(|digits| digits.len() == 8)
            .and_then(|digits| u32::from_str_radix(digits, 16).ok())
            .ok_or_else(|| format!("{checksum:?} is no CRC-32 of 8 digits"))?;
        let sha256 = field("sha256")?;
        if unhex(&sha256).is_none() {
            return Err(format!("{sha256:?} is no SHA-256 of 64 digits"));
        }
        let section = number("section", field("section")?)?;
        if section == 0 {
            return Err("a section has to have a byte at least".to_owned());
        }
        let mut hashes = Vec::new();
        for (number, line) in lines {
            let from = hashes.len() as u64 * section;
            let hash = line
                .split_once("  ")
                .filter(|(_, offset)| offset.parse() == Ok(from))
                .and_then(|(digits, _)| unhex(digits))
                .ok_or_else(|| format!("line {number} is to be a SHA-256, two spaces and {from}"))?;
            hashes.push(hash);
        }
        let sections = length.div_ceil(section);
        if hashes.len() as u64 != sections {
            return Err(format!(
                "it has the checksums of {} sections, and a file of {length} bytes has {sections}",
                hashes.len()
            ));
        }
        Ok(Self {
            name,
            origin: Origin { checksum, length },
            sha256,
            section,
            hashes,
        })
    }
}

/// The text of the checksums, as [`Checks::parse`] reads it.
impl fmt::Display for Checks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{TITLE}")?;
        writeln!(f, "file {}", self.name)?;
        writeln!(f, "length {}", self.origin.length)?;
        writeln!(f, "crc32 {:08x}", self.origin.checksum)?;
        writeln!(f, "sha256 {}", self.sha256)?;
        writeln!(f, "section {}", self.section)?;
        for (section, hash) in self.hashes.iter().enumerate() {
            writeln!(f, "{}  {}", digest::hex(hash), section as u64 * self.section)?;
        }
        Ok(())
    }
}

/// `name` on one line, without what a terminal would take for commands.
fn printable(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

/// The SHA-256 that 64 digits are, in small letters as they are written here.
fn unhex(digits: &str) -> Option<Hash> {
    let digits = digits.as_bytes();
    if digits.len() != 64 {
        return None;
    }
    let digit = |digit: u8| match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        _ => None,
    };
    let mut hash = [0; 32];
    for (byte, pair) in hash.iter_mut().zip(digits.chunks(2)) {
        *byte = digit(pair[0])? << 4 | digit(pair[1])?;
    }
    Some(hash)
}

/// Makes the checksums of a file that comes piece by piece.
pub struct Maker {
    section: u64,
    hashes: Vec<Hash>,
    /// The section that is begun, and how many bytes of it there are
    begun: Sha256,
    filled: u64,
}

impl Maker {
    pub fn new(section: u64) -> Self {
        assert!(section > 0, "a section has to have a byte at least");
        Self {
            section,
            hashes: Vec::new(),
            begun: Sha256::new(),
            filled: 0,
        }
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let room = (self.section - self.filled).min(bytes.len() as u64) as usize;
            self.begun.update(&bytes[..room]);
            self.filled += room as u64;
            bytes = &bytes[room..];
            if self.filled == self.section {
                self.close();
            }
        }
    }

    fn close(&mut self) {
        let mut hash = [0; 32];
        hash.copy_from_slice(&std::mem::take(&mut self.begun).finalize());
        self.hashes.push(hash);
        self.filled = 0;
    }

    /// The checksums of the file that went in. What it is called, what the frames name it by and its
    /// SHA-256 in hexadecimal are for whoever read the file to say.
    pub fn finish(mut self, name: &str, origin: Origin, sha256: &str) -> Checks {
        if self.filled > 0 {
            self.close();
        }
        Checks {
            name: printable(name),
            origin,
            sha256: sha256.to_owned(),
            section: self.section,
            hashes: self.hashes,
        }
    }
}

/// What holding a file against its checksums came to, for whoever watches the reading.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Note {
    /// A section that is whole has not its checksum: a packet of it was misread. Which one is looked
    /// for among the packets of its blocks, with the sums that arrive from here on.
    Doubted { section: usize, from: u64 },
    /// The packet that was misread is found, and the section is made of the others.
    Mended { section: usize, from: u64 },
    /// No one packet accounts for a section that has not its checksum, so what there is of its blocks
    /// is dropped and they are read anew.
    Dropped { section: usize, from: u64 },
    /// A file that was kept in memory is whole and so many of its sections have not their checksums;
    /// `from` is where the first of them begins.
    Wrong { sections: usize, from: u64 },
}

impl fmt::Display for Note {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Note::Doubted { section, from } => write!(
                f,
                "section {section} of the file, from byte {from}, has not its checksum: a packet was misread, \
                 and the sums that come next are to tell which"
            ),
            Note::Mended { section, from } => write!(
                f,
                "section {section} of the file, from byte {from}, has its checksum now: the packet that was \
                 misread is found and left out"
            ),
            Note::Dropped { section, from } => write!(
                f,
                "section {section} of the file, from byte {from}, still has not its checksum and no one packet \
                 accounts for it: its blocks are read anew. If this comes again, the checksums are not those \
                 of the file on show"
            ),
            Note::Wrong { sections: 1, from } => {
                write!(f, "the section of the file from byte {from} has not its checksum")
            }
            Note::Wrong { sections, from } => write!(
                f,
                "{sections} sections of the file have not their checksums, the first of them from byte {from}"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(length: usize) -> Vec<u8> {
        (0..length).map(|at| (at * 31 % 251) as u8).collect()
    }

    #[test]
    fn text_is_laid_out_as_documented() {
        // the checksums of "abc" are those that crc32 and sha256sum print
        let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(
            Checks::of("notes.txt", b"abc").to_string(),
            format!(
                "qqcode checksums 1\nfile notes.txt\nlength 3\ncrc32 352441c2\nsha256 {abc}\nsection 1048576\n\
                 {abc}  0\n"
            )
        );
        // in sections of two bytes: "ab" and "c"
        let text = Checks::in_sections("notes.txt", b"abc", 2).to_string();
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines[5], "section 2");
        assert_eq!(
            lines[6..],
            [
                "fb8e20fc2e4c3f248c60c39bd652f3c1347298bb977b8b4d5903b85055620603  0",
                "2e7d2c03a9507ae265ecf5b5356885a53393a2029d241394997265a1a25aefc6  2"
            ]
        );
        // a file without a byte has no section
        let empty = Checks::of("", b"");
        assert_eq!(empty.sections(), 0);
        assert!(empty.to_string().ends_with("length 0\ncrc32 00000000\nsha256 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\nsection 1048576\n"));
    }

    #[test]
    fn text_comes_back_as_the_checksums_it_was_made_of() {
        for (name, length, section) in [
            ("disk.img", 2_500_000, SECTION),
            ("two words.bin", 1000, 300),
            ("", 0, 16),
            ("x", 600, 300),
        ] {
            let checks = Checks::in_sections(name, &bytes(length), section);
            assert_eq!(checks.sections(), length.div_ceil(section as usize));
            assert_eq!(Checks::parse(&checks.to_string()), Ok(checks.clone()), "{name}");
            // as a text that went through a machine with other line ends, and gained an empty line
            let other = checks.to_string().replace('\n', "\r\n") + "\r\n";
            assert_eq!(Checks::parse(&other), Ok(checks));
        }
    }

    #[test]
    fn checksums_of_pieces_are_those_of_the_whole() {
        let file = bytes(2500);
        let whole = Checks::in_sections("f", &file, 1000);
        for piece in [1, 7, 999, 1000, 1001, 2500] {
            let mut maker = Maker::new(1000);
            file.chunks(piece).for_each(|bytes| maker.update(bytes));
            assert_eq!(maker.finish("f", whole.origin, &whole.sha256), whole, "{piece}");
        }
        assert_eq!(
            (whole.range(0), whole.range(1), whole.range(2), whole.range(3)),
            (0..1000, 1000..2000, 2000..2500, 2500..2500)
        );
    }

    #[test]
    fn sections_are_held_against_their_checksums() {
        let mut file = bytes(2500);
        let checks = Checks::in_sections("f", &file, 1000);
        assert!(checks.holds(0, &file[..1000]) && checks.holds(2, &file[2000..]));
        assert!(!checks.holds(1, &file[..1000]), "the bytes of another section");
        assert!(!checks.holds(2, &file[2000..2499]), "a section cut short");
        assert!(!checks.holds(3, b""), "a section that there is not");
        assert_eq!(checks.wrong(&file), [0usize; 0]);
        file[1999] ^= 1;
        file[2000] ^= 0x80;
        assert_eq!(checks.wrong(&file), [1, 2]);
        // of a file that is too short, the sections that are not all there
        assert_eq!(checks.wrong(&bytes(2400)), [2]);
        assert_eq!(checks.wrong(b""), [0, 1, 2]);
    }

    #[test]
    fn text_that_is_no_list_of_checksums_says_why() {
        let checks = Checks::in_sections("f", &bytes(2500), 1000);
        let text = checks.to_string();
        let lines = text.lines().collect::<Vec<_>>();
        let without = |line: usize| {
            let mut lines = lines.clone();
            lines.remove(line);
            lines.join("\n")
        };
        let with = |line: usize, other: &str| {
            let mut lines = lines.clone();
            lines[line] = other;
            lines.join("\n")
        };
        let error = |text: String| Checks::parse(&text).unwrap_err();
        assert!(error(String::new()).contains("does not begin with"));
        assert!(error("notes".to_owned()).contains("does not begin with"));
        assert!(error(with(0, "qqcode checksums 2")).contains("a layout that this qqcode does not know"));
        assert_eq!(error(without(2)), "line 3 is to begin with \"length\"");
        assert_eq!(
            error(lines[..3].join("\n")),
            "it ends before the line of \"crc32\""
        );
        assert!(error(with(2, "length many")).contains("no number for \"length\""));
        assert!(error(with(3, "crc32 12345")).contains("no CRC-32"));
        assert!(error(with(3, "crc32 1234567g")).contains("no CRC-32"));
        assert!(error(with(4, "sha256 1234")).contains("no SHA-256"));
        assert!(error(with(5, "section 0")).contains("a byte at least"));
        // a line that is missing, one too many, one that is damaged, and one of capital letters
        assert_eq!(
            error(without(7)),
            "line 8 is to be a SHA-256, two spaces and 1000"
        );
        assert_eq!(
            error(lines[..8].join("\n")),
            "it has the checksums of 2 sections, and a file of 2500 bytes has 3"
        );
        let more = format!("{text}{}  3000\n", "0".repeat(64));
        assert_eq!(
            error(more),
            "it has the checksums of 4 sections, and a file of 2500 bytes has 3"
        );
        assert!(error(with(6, &lines[6][1..])).contains("line 7 is to be a SHA-256"));
        assert!(error(with(6, &lines[6].to_uppercase())).contains("line 7 is to be a SHA-256"));
        // the name is whatever follows, and may be nothing
        assert_eq!(Checks::parse(&with(1, "file")).unwrap().name, "");
        assert_eq!(Checks::parse(&with(1, "file a  b ")).unwrap().name, "a  b");
        assert_eq!(Checks::in_sections("a\nb\x1b[2J", b"", 1).name, "a?b?[2J");
    }

    #[test]
    fn notes_say_where_in_the_file() {
        let note = Note::Doubted {
            section: 3,
            from: 3 << 20,
        };
        assert!(
            note.to_string()
                .starts_with("section 3 of the file, from byte 3145728, has not its checksum")
        );
        assert_eq!(
            Note::Wrong { sections: 1, from: 0 }.to_string(),
            "the section of the file from byte 0 has not its checksum"
        );
        assert_eq!(
            Note::Wrong {
                sections: 4,
                from: 2000
            }
            .to_string(),
            "4 sections of the file have not their checksums, the first of them from byte 2000"
        );
    }

    #[test]
    fn name_in_a_text_is_made_safe_to_print() {
        // the list comes from somewhere else and its name ends up on a terminal
        let text = Checks::of("notes.txt", b"abc").to_string();
        let text = text.replace("file notes.txt", "file a\u{1b}[2Jb\u{7}c");
        assert_eq!(Checks::parse(&text).unwrap().name, "a?[2Jb?c");
    }

    #[test]
    fn checksum_of_more_than_64_digits_is_none() {
        let text = Checks::in_sections("f", &bytes(2500), 1000).to_string();
        let lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
        // two digits more on the SHA-256 of the file
        let mut longer = lines.clone();
        longer[4].push_str("00");
        assert!(
            Checks::parse(&longer.join("\n"))
                .unwrap_err()
                .contains("no SHA-256")
        );
        // and on that of a section
        let mut longer = lines.clone();
        longer[6] = longer[6].replacen("  ", "00  ", 1);
        assert_eq!(
            Checks::parse(&longer.join("\n")).unwrap_err(),
            "line 7 is to be a SHA-256, two spaces and 0"
        );
    }

    #[test]
    fn control_characters_beyond_ascii_are_taken_out_of_a_name_as_well() {
        // U+009B begins a control sequence on a terminal as the escape character and a bracket do
        let text = Checks::of("notes.txt", b"abc").to_string();
        let text = text.replace("file notes.txt", "file a\u{9b}2Jb");
        assert_eq!(Checks::parse(&text).unwrap().name, "a?2Jb");
        assert_eq!(Checks::of("a\u{9b}2Jb", b"").name, "a?2Jb");
    }
}
