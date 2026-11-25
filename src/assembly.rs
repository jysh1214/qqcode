//! Puts the pieces of a file together.
//!
//! The pieces come in any order, any number of times and in any size: a camera misses codes, a show is
//! gone through a second time for what was missed, and a show whose layout changes cuts the file anew. So
//! what is kept is not a list of pieces but the bytes themselves and which of them have arrived.

use std::ops::Range;

use crate::protocol::{self, FileId};

pub struct Assembly {
    file: FileId,
    /// The bytes of the file as far as the pieces reach; those that no piece has brought are 0
    bytes: Vec<u8>,
    /// Which bytes the pieces have brought: ranges in rising order that neither overlap nor touch
    held: Vec<Range<usize>>,
}

impl Assembly {
    pub fn new(file: FileId) -> Self {
        Self {
            file,
            bytes: Vec::new(),
            held: Vec::new(),
        }
    }

    pub fn file(&self) -> FileId {
        self.file
    }

    fn length(&self) -> usize {
        self.file.length as usize
    }

    /// Takes the bytes of the file from `offset` on. Returns how many of them had not arrived before, which
    /// is none for a piece that does not lie within the file.
    pub fn add(&mut self, offset: usize, piece: &[u8]) -> usize {
        let Some(end) = offset
            .checked_add(piece.len())
            .filter(|end| *end <= self.length())
        else {
            return 0;
        };
        if piece.is_empty() {
            return 0;
        }
        if self.bytes.len() < end {
            self.bytes.resize(end, 0);
        }
        self.bytes[offset..end].copy_from_slice(piece);

        let before = self.received();
        let mut joined = offset..end;
        let mut held = Vec::with_capacity(self.held.len() + 1);
        let mut placed = false;
        for range in self.held.drain(..) {
            if range.end < joined.start {
                held.push(range);
            } else if range.start > joined.end {
                if !placed {
                    held.push(joined.clone());
                    placed = true;
                }
                held.push(range);
            } else {
                joined = joined.start.min(range.start)..joined.end.max(range.end);
            }
        }
        if !placed {
            held.push(joined);
        }
        self.held = held;
        self.received() - before
    }

    /// How many bytes of the file have arrived.
    pub fn received(&self) -> usize {
        self.held.iter().map(Range::len).sum()
    }

    /// The parts of the file that have not arrived, in rising order.
    pub fn missing(&self) -> Vec<Range<usize>> {
        let mut missing = Vec::new();
        let mut from = 0;
        for range in &self.held {
            if range.start > from {
                missing.push(from..range.start);
            }
            from = range.end;
        }
        if from < self.length() {
            missing.push(from..self.length());
        }
        missing
    }

    /// How many of the bytes in `range` have not arrived.
    pub fn missing_in(&self, range: Range<usize>) -> usize {
        let overlap = |missing: Range<usize>| {
            missing
                .end
                .min(range.end)
                .saturating_sub(missing.start.max(range.start))
        };
        self.missing().into_iter().map(overlap).sum()
    }

    pub fn is_complete(&self) -> bool {
        self.received() == self.length()
    }

    /// The file, if all of it has arrived and it has the checksum that its codes name. A wrong checksum
    /// means that a code was misread in a way that its own error correction did not notice.
    pub fn into_bytes(mut self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }
        // nothing but the empty file has no bytes at all
        self.bytes.truncate(self.length());
        (protocol::checksum(&self.bytes) == self.file.checksum).then_some(self.bytes)
    }
}

#[cfg(test)]
// a file may well be missing in one part only
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    fn text() -> Vec<u8> {
        (0..100u8).collect()
    }

    fn assembly(bytes: &[u8]) -> Assembly {
        Assembly::new(FileId::of(bytes).unwrap())
    }

    #[test]
    fn pieces_in_order_make_the_file() {
        let text = text();
        let mut assembly = assembly(&text);
        assert_eq!(assembly.received(), 0);
        assert_eq!(assembly.missing(), [0..100]);
        for (index, piece) in text.chunks(30).enumerate() {
            assert!(!assembly.is_complete());
            assert_eq!(assembly.add(30 * index, piece), piece.len());
        }
        assert_eq!(assembly.received(), 100);
        assert!(assembly.is_complete() && assembly.missing().is_empty());
        assert_eq!(assembly.into_bytes(), Some(text));
    }

    #[test]
    fn pieces_come_in_any_order_and_any_number_of_times() {
        let text = text();
        let mut assembly = assembly(&text);
        assert_eq!(assembly.add(60, &text[60..90]), 30);
        assert_eq!(assembly.add(0, &text[..30]), 30);
        assert_eq!(assembly.missing(), [30..60, 90..100]);
        assert_eq!(assembly.missing_in(0..100), 40);
        assert_eq!(assembly.missing_in(20..50), 20);
        assert_eq!(assembly.missing_in(45..95), 20);
        assert_eq!(assembly.missing_in(0..30), 0);
        // a second time
        assert_eq!(assembly.add(60, &text[60..90]), 0);
        assert_eq!(assembly.received(), 60);
        assert_eq!(assembly.add(90, &text[90..]), 10);
        assert_eq!(assembly.add(30, &text[30..60]), 30);
        assert_eq!(assembly.into_bytes(), Some(text));
    }

    #[test]
    fn pieces_of_another_size_fill_what_is_missing() {
        let text = text();
        let mut assembly = assembly(&text);
        assembly.add(0, &text[..30]);
        assembly.add(60, &text[60..90]);
        // the file cut into pieces of 45 instead of 30: each brings what the others left
        assert_eq!(assembly.add(0, &text[..45]), 15);
        assert_eq!(assembly.missing(), [45..60, 90..100]);
        assert_eq!(assembly.add(45, &text[45..90]), 15);
        assert_eq!(assembly.add(90, &text[90..]), 10);
        assert_eq!(assembly.missing(), []);
        assert_eq!(assembly.into_bytes(), Some(text));
    }

    #[test]
    fn piece_that_touches_its_neighbours_joins_them() {
        let text = text();
        let mut assembly = assembly(&text);
        assembly.add(0, &text[..10]);
        assembly.add(20, &text[20..30]);
        assembly.add(40, &text[40..50]);
        assert_eq!(assembly.held, [0..10, 20..30, 40..50]);
        assert_eq!(assembly.add(10, &text[10..20]), 10);
        assert_eq!(assembly.held, [0..30, 40..50]);
        assert_eq!(assembly.add(25, &text[25..45]), 10);
        assert_eq!(assembly.held, [0..50]);
    }

    #[test]
    fn piece_outside_the_file_is_not_taken() {
        let text = text();
        let mut assembly = assembly(&text);
        assert_eq!(assembly.add(95, &[0; 6]), 0);
        assert_eq!(assembly.add(usize::MAX, &[0; 6]), 0);
        assert_eq!(assembly.add(50, &[]), 0);
        assert_eq!(assembly.received(), 0);
        assert_eq!(assembly.add(95, &text[95..]), 5);
    }

    #[test]
    fn file_is_not_given_out_before_it_is_whole() {
        let text = text();
        let mut assembly = assembly(&text);
        assembly.add(0, &text[..99]);
        assert!(!assembly.is_complete());
        assert_eq!(assembly.into_bytes(), None);
    }

    #[test]
    fn misread_piece_is_caught_by_the_checksum() {
        let text = text();
        let mut assembly = assembly(&text);
        assembly.add(0, &text[..50]);
        let mut wrong = text[50..].to_vec();
        wrong[7] ^= 0x10;
        assembly.add(50, &wrong);
        assert!(assembly.is_complete());
        assert_eq!(assembly.into_bytes(), None);
    }

    #[test]
    fn empty_file_is_whole_from_the_start() {
        let assembly = assembly(&[]);
        assert!(assembly.is_complete());
        assert_eq!(assembly.missing(), []);
        assert_eq!(assembly.into_bytes(), Some(Vec::new()));
    }
}
