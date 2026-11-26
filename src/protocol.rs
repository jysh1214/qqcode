//! What the codes of `qqcode-show` say.
//!
//! Every code holds one message: the start screen, a piece of the file, or the end screen. Each of them
//! names the file it belongs to by its checksum and its length, so a code is told from one of another file
//! without having seen what came before. A piece says where in the file its bytes go rather than which
//! piece it is. The pieces can then be of any size, and a show that is laid out anew half way, with larger
//! or smaller codes, goes on filling the same file.
//!
//! ```text
//! start   "qq" 1 checksum length name...
//! piece   "qq" 2 checksum length offset bytes...
//! end     "qq" 3 checksum length
//! ```
//!
//! The numbers are of 32 bits, most significant byte first, which limits a file to 4 GiB less a byte.

/// What every message begins with, to tell the codes of `qqcode-show` from any other QR code.
const MARK: [u8; 2] = *b"qq";

const START: u8 = 1;
const PIECE: u8 = 2;
const END: u8 = 3;

/// How many bytes of a code a start or end message takes before anything else: mark, kind, checksum, length.
pub const HEADER: usize = 11;

/// How many bytes of a code a piece takes before its bytes of the file: the header and the offset.
pub const PIECE_HEADER: usize = HEADER + 4;

/// The file that a message belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileId {
    /// CRC-32 of the whole file, see [`checksum`]
    pub checksum: u32,
    /// Bytes in the file
    pub length: u32,
}

impl FileId {
    /// `None` for more bytes than a message can count.
    pub fn of(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            checksum: checksum(bytes),
            length: u32::try_from(bytes.len()).ok()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// The start screen: the pieces are still to come. `name` is what the file is called where it is
    /// shown, for whoever reads the remarks; it may be cut short or empty.
    Start { file: FileId, name: String },
    /// The bytes of the file from `offset` on
    Piece {
        file: FileId,
        offset: u32,
        bytes: Vec<u8>,
    },
    /// The end screen: all pieces have been shown
    End { file: FileId },
}

impl Message {
    pub fn file(&self) -> FileId {
        match self {
            Message::Start { file, .. } | Message::Piece { file, .. } | Message::End { file } => *file,
        }
    }

    /// The bytes that go into a code.
    pub fn encode(&self) -> Vec<u8> {
        let kind = match self {
            Message::Start { .. } => START,
            Message::Piece { .. } => PIECE,
            Message::End { .. } => END,
        };
        let file = self.file();
        let mut encoded = Vec::with_capacity(PIECE_HEADER);
        encoded.extend(MARK);
        encoded.push(kind);
        encoded.extend(file.checksum.to_be_bytes());
        encoded.extend(file.length.to_be_bytes());
        match self {
            Message::Start { name, .. } => encoded.extend(name.as_bytes()),
            Message::Piece { offset, bytes, .. } => {
                encoded.extend(offset.to_be_bytes());
                encoded.extend(bytes);
            }
            Message::End { .. } => {}
        }
        encoded
    }

    /// The message in the bytes of a code, or `None` if they are not a message of `qqcode-show`: any other
    /// QR code, or a piece that does not lie within its file.
    pub fn parse(encoded: &[u8]) -> Option<Self> {
        let rest = encoded.strip_prefix(&MARK)?;
        let (&kind, rest) = rest.split_first()?;
        let (checksum, rest) = number(rest)?;
        let (length, rest) = number(rest)?;
        let file = FileId { checksum, length };
        match kind {
            START => Some(Message::Start {
                file,
                name: printable(&String::from_utf8_lossy(rest)),
            }),
            PIECE => {
                let (offset, bytes) = number(rest)?;
                let end = offset as u64 + bytes.len() as u64;
                (!bytes.is_empty() && end <= length as u64).then(|| Message::Piece {
                    file,
                    offset,
                    bytes: bytes.to_vec(),
                })
            }
            END => rest.is_empty().then_some(Message::End { file }),
            _ => None,
        }
    }
}

/// The number that `bytes` begin with, and what follows it.
fn number(bytes: &[u8]) -> Option<(u32, &[u8])> {
    let (number, rest) = bytes.split_first_chunk::<4>()?;
    Some((u32::from_be_bytes(*number), rest))
}

/// `name` without what a terminal would take for commands: it comes out of a picture and ends up in the
/// remarks on a terminal.
fn printable(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

/// The longest beginning of `name` that is no more than `room` bytes.
pub fn fit_name(name: &str, room: usize) -> &str {
    let mut end = name.len().min(room);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

/// The remainders of the bytes 0 to 255 for [`checksum`].
const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut remainder = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            remainder = if remainder & 1 == 1 {
                0xedb8_8320 ^ (remainder >> 1)
            } else {
                remainder >> 1
            };
            bit += 1;
        }
        table[byte] = remainder;
        byte += 1;
    }
    table
};

/// CRC-32 as zip, gzip and PNG compute it, which is also what `crc32` of the shell and of Python print.
pub fn checksum(bytes: &[u8]) -> u32 {
    let mut checksum = Checksum::new();
    checksum.update(bytes);
    checksum.finish()
}

/// The [`checksum`] of bytes that come piece by piece.
pub struct Checksum(u32);

impl Checksum {
    pub fn new() -> Self {
        Self(!0)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        let step = |crc: u32, byte: &u8| CRC_TABLE[((crc ^ *byte as u32) & 0xff) as usize] ^ (crc >> 8);
        self.0 = bytes.iter().fold(self.0, step);
    }

    pub fn finish(&self) -> u32 {
        !self.0
    }
}

impl Default for Checksum {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FILE: FileId = FileId {
        checksum: 0x0102_0304,
        length: 1000,
    };

    #[test]
    fn checksum_is_the_crc_32_of_zip() {
        assert_eq!(checksum(b"123456789"), 0xcbf4_3926);
        assert_eq!(checksum(b""), 0);
        // piece by piece
        let mut pieces = Checksum::new();
        pieces.update(b"1234");
        pieces.update(b"");
        pieces.update(b"56789");
        assert_eq!(pieces.finish(), 0xcbf4_3926);
        assert_eq!(
            checksum(b"The quick brown fox jumps over the lazy dog"),
            0x414f_a339
        );
    }

    #[test]
    fn file_is_named_by_checksum_and_length() {
        assert_eq!(
            FileId::of(b"123456789"),
            Some(FileId {
                checksum: 0xcbf4_3926,
                length: 9
            })
        );
    }

    #[test]
    fn messages_come_back_as_they_went_in() {
        let messages = [
            Message::Start {
                file: FILE,
                name: "notes ü.txt".to_owned(),
            },
            Message::Start {
                file: FILE,
                name: String::new(),
            },
            Message::Piece {
                file: FILE,
                offset: 990,
                bytes: vec![0, 255, 10, 13, 0, 0, 113, 113, 2, 7],
            },
            Message::End { file: FILE },
        ];
        for message in messages {
            assert_eq!(Message::parse(&message.encode()), Some(message.clone()));
            assert_eq!(message.file(), FILE);
        }
    }

    #[test]
    fn messages_are_laid_out_as_documented() {
        let piece = Message::Piece {
            file: FILE,
            offset: 258,
            bytes: vec![9, 8],
        };
        assert_eq!(
            piece.encode(),
            [b'q', b'q', 2, 1, 2, 3, 4, 0, 0, 3, 232, 0, 0, 1, 2, 9, 8]
        );
        assert_eq!(piece.encode().len(), PIECE_HEADER + 2);
        assert_eq!(Message::End { file: FILE }.encode().len(), HEADER);
        let start = Message::Start {
            file: FILE,
            name: "a".to_owned(),
        };
        assert_eq!(start.encode(), [b'q', b'q', 1, 1, 2, 3, 4, 0, 0, 3, 232, b'a']);
    }

    #[test]
    fn other_codes_are_no_messages() {
        let piece = Message::Piece {
            file: FILE,
            offset: 0,
            bytes: vec![1, 2, 3],
        }
        .encode();
        for other in [
            &b""[..],
            b"q",
            b"https://example.org/qq",
            b"qq",
            b"qq\x02\x01\x02",
            // a kind that there is not
            b"qq\x00\x01\x02\x03\x04\x00\x00\x03\xe8",
            b"qq\x04\x01\x02\x03\x04\x00\x00\x03\xe8",
            // a piece without bytes
            &piece[..PIECE_HEADER],
            // an end with something after it
            b"qq\x03\x01\x02\x03\x04\x00\x00\x03\xe8!",
        ] {
            assert_eq!(Message::parse(other), None, "{other:?}");
        }
        // every byte of the header counts
        assert_eq!(Message::parse(&piece[1..]), None);
    }

    #[test]
    fn piece_has_to_lie_within_its_file() {
        let piece = |offset: u32, count: usize| Message::Piece {
            file: FILE,
            offset,
            bytes: vec![7; count],
        };
        assert!(Message::parse(&piece(990, 10).encode()).is_some());
        assert_eq!(Message::parse(&piece(991, 10).encode()), None);
        assert_eq!(Message::parse(&piece(u32::MAX, 1).encode()), None);
    }

    #[test]
    fn name_from_a_code_is_safe_to_print() {
        let mut start = Message::Start {
            file: FILE,
            name: String::new(),
        }
        .encode();
        start.extend(b"a\x1b[2Jb\n\xff");
        let Some(Message::Start { name, .. }) = Message::parse(&start) else {
            panic!("a start message");
        };
        assert_eq!(name, "a?[2Jb?\u{fffd}");
    }

    #[test]
    fn name_is_cut_between_characters() {
        assert_eq!(fit_name("notes.txt", 20), "notes.txt");
        assert_eq!(fit_name("notes.txt", 5), "notes");
        // two bytes for the ü
        assert_eq!(fit_name("grün", 3), "gr");
        assert_eq!(fit_name("grün", 4), "grü");
        assert_eq!(fit_name("grün", 0), "");
    }
}
