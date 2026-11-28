//! What a frame of the web page says, once its tiles are read and put right.
//!
//! ```text
//! start    "qq" 0x10 checksum length 0     0        name length, name, zeros     check
//! packet   "qq" 0x11 checksum length block sequence bytes of the packet          check
//! ```
//!
//! The checksum is the CRC-32 of the file and names it together with its length, which is a number of 64
//! bits here; block, sequence and the check, the CRC-32 of all that comes before it in the frame, are of
//! 32 bits, and all of them have their most significant byte first. A frame is always full, so that what
//! a packet holds follows from the size of the matrix: the frame less [`OVERHEAD`] bytes. The check is
//! what keeps a frame that was put right wrongly out of the sums of [`crate::fountain`], where one wrong
//! packet would spoil a whole block.

use crate::crc::checksum;

const MARK: [u8; 2] = *b"qq";
const START: u8 = 0x10;
const PACKET: u8 = 0x11;

/// Bytes of a frame before what it carries: mark, kind, checksum, length, block, sequence.
const HEADER: usize = 23;

/// Bytes of a frame that are not the packet in it: the header and the check at the end.
pub const OVERHEAD: usize = HEADER + 4;

/// The file that a frame belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Origin {
    /// CRC-32 of the whole file
    pub checksum: u32,
    /// Bytes in the file
    pub length: u64,
}

impl Origin {
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            checksum: checksum(bytes),
            length: bytes.len() as u64,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Frame {
    /// The start screen: the packets are still to come. `name` is what the file is called where it is
    /// shown.
    Start { origin: Origin, name: String },
    /// A packet of [`crate::fountain`]
    Packet {
        origin: Origin,
        block: u32,
        sequence: u32,
        bytes: Vec<u8>,
    },
}

impl Frame {
    pub fn origin(&self) -> Origin {
        match self {
            Frame::Start { origin, .. } | Frame::Packet { origin, .. } => *origin,
        }
    }

    /// The bytes of a frame that holds `capacity` of them. A name that is too long for it is cut short,
    /// and a packet has to be exactly what the frame has room for.
    pub fn encode(&self, capacity: usize) -> Vec<u8> {
        assert!(capacity > OVERHEAD + 2, "a frame that holds something");
        let origin = self.origin();
        let (kind, block, sequence) = match self {
            Frame::Start { .. } => (START, 0, 0),
            Frame::Packet { block, sequence, .. } => (PACKET, *block, *sequence),
        };
        let mut frame = Vec::with_capacity(capacity);
        frame.extend(MARK);
        frame.push(kind);
        frame.extend(origin.checksum.to_be_bytes());
        frame.extend(origin.length.to_be_bytes());
        frame.extend(block.to_be_bytes());
        frame.extend(sequence.to_be_bytes());
        match self {
            Frame::Start { name, .. } => {
                let name = fit_name(name, (capacity - OVERHEAD - 2).min(u16::MAX as usize));
                frame.extend((name.len() as u16).to_be_bytes());
                frame.extend(name.as_bytes());
                frame.resize(capacity - 4, 0);
            }
            Frame::Packet { bytes, .. } => {
                assert_eq!(bytes.len(), capacity - OVERHEAD, "a packet that fills the frame");
                frame.extend(bytes);
            }
        }
        frame.extend(checksum(&frame).to_be_bytes());
        frame
    }

    /// The frame in these bytes, or `None` if they are not one: their check is wrong, which is to say that
    /// tiles were misread beyond what the check bytes of the picture make up for and noticed.
    pub fn parse(frame: &[u8]) -> Option<Self> {
        let (body, check) = frame.split_last_chunk::<4>()?;
        if body.len() < HEADER + 2 || body[..2] != MARK || u32::from_be_bytes(*check) != checksum(body) {
            return None;
        }
        let number = |at: usize| u32::from_be_bytes(body[at..at + 4].try_into().expect("four bytes"));
        let origin = Origin {
            checksum: number(3),
            length: (number(7) as u64) << 32 | number(11) as u64,
        };
        let rest = &body[HEADER..];
        match body[2] {
            START => {
                let length = u16::from_be_bytes([rest[0], rest[1]]) as usize;
                let name = String::from_utf8_lossy(rest.get(2..2 + length)?);
                // it comes out of a picture and ends up in the remarks on a terminal
                let name = name
                    .chars()
                    .map(|c| if c.is_control() { '?' } else { c })
                    .collect();
                Some(Frame::Start { origin, name })
            }
            PACKET => Some(Frame::Packet {
                origin,
                block: number(15),
                sequence: number(19),
                bytes: rest.to_vec(),
            }),
            _ => None,
        }
    }
}

/// The longest beginning of `name` that is no more than `room` bytes.
fn fit_name(name: &str, room: usize) -> &str {
    let mut end = name.len().min(room);
    while !name.is_char_boundary(end) {
        end -= 1;
    }
    &name[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    const ORIGIN: Origin = Origin {
        checksum: 0x0102_0304,
        length: 0x0000_0005_0000_0006,
    };

    #[test]
    fn frames_come_back_as_they_went_in() {
        let packet = Frame::Packet {
            origin: ORIGIN,
            block: 7,
            sequence: 0x0100_0002,
            bytes: (0..73).collect(),
        };
        let encoded = packet.encode(100);
        assert_eq!(encoded.len(), 100);
        assert_eq!(Frame::parse(&encoded), Some(packet.clone()));
        assert_eq!(packet.origin(), ORIGIN);

        let start = Frame::Start {
            origin: ORIGIN,
            name: "notes ü.txt".to_owned(),
        };
        let encoded = start.encode(100);
        assert_eq!(encoded.len(), 100);
        assert_eq!(Frame::parse(&encoded), Some(start));
    }

    #[test]
    fn frames_are_laid_out_as_documented() {
        let packet = Frame::Packet {
            origin: ORIGIN,
            block: 7,
            sequence: 258,
            bytes: vec![9, 8, 7],
        };
        let encoded = packet.encode(OVERHEAD + 3);
        assert_eq!(
            encoded[..26],
            [
                b'q', b'q', 0x11, 1, 2, 3, 4, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 1, 2, 9, 8, 7
            ]
        );
        assert_eq!(encoded[26..], checksum(&encoded[..26]).to_be_bytes());
        let start = Frame::Start {
            origin: ORIGIN,
            name: "ab".to_owned(),
        };
        let encoded = start.encode(40);
        assert_eq!(encoded[2], 0x10);
        assert_eq!(
            encoded[15..36],
            [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 2, b'a', b'b', 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn name_that_is_too_long_is_cut_short() {
        let start = Frame::Start {
            origin: ORIGIN,
            name: "a name that is longer than the frame".to_owned(),
        };
        let Some(Frame::Start { name, .. }) = Frame::parse(&start.encode(OVERHEAD + 2 + 6)) else {
            panic!("a start frame");
        };
        assert_eq!(name, "a name");
        let mut raw = start.encode(80);
        raw[25] = 0x1b;
        let body = raw.len() - 4;
        let check = checksum(&raw[..body]).to_be_bytes();
        raw[body..].copy_from_slice(&check);
        let Some(Frame::Start { name, .. }) = Frame::parse(&raw) else {
            panic!("a start frame");
        };
        assert!(name.starts_with("?"), "{name}");
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

    #[test]
    fn frame_with_a_wrong_byte_is_none() {
        let packet = Frame::Packet {
            origin: ORIGIN,
            block: 0,
            sequence: 0,
            bytes: vec![0; 50],
        };
        let encoded = packet.encode(OVERHEAD + 50);
        for place in [0, 2, 5, 20, 40, 76] {
            let mut wrong = encoded.clone();
            wrong[place] ^= 4;
            assert_eq!(Frame::parse(&wrong), None, "byte {place}");
        }
        assert_eq!(Frame::parse(&encoded[1..]), None);
        assert_eq!(Frame::parse(b"qq"), None);
        assert_eq!(Frame::parse(&[]), None);
        // a kind that there is not, with a check that is right
        let mut other = encoded[..encoded.len() - 4].to_vec();
        other[2] = 0x12;
        other.extend(checksum(&other).to_be_bytes());
        assert_eq!(Frame::parse(&other), None);
    }

    #[test]
    fn start_frame_that_ends_before_its_name_is_none() {
        // what a start screen has before the length of its name, and one byte of that length, each
        // with a check that is right: too short to be a frame, and not to be read beyond its end
        for more in 0..2 {
            let mut frame = MARK.to_vec();
            frame.push(START);
            frame.extend([0; HEADER - 3]);
            frame.extend(vec![0; more]);
            frame.extend(checksum(&frame).to_be_bytes());
            assert_eq!(Frame::parse(&frame), None, "{more} bytes after the header");
        }
        // with both bytes of the length it is one: a start screen without a name
        let mut frame = MARK.to_vec();
        frame.push(START);
        frame.extend([0; HEADER - 3 + 2]);
        frame.extend(checksum(&frame).to_be_bytes());
        let origin = Origin {
            checksum: 0,
            length: 0,
        };
        assert_eq!(
            Frame::parse(&frame),
            Some(Frame::Start {
                origin,
                name: String::new()
            })
        );
    }
}
