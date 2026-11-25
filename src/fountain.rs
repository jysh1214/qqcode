//! Cuts a file into packets of which any that are enough put it together again.
//!
//! A screen can show a camera more frames than the camera catches, and nothing tells the screen which ones
//! were lost. So the file is not sent piece by piece and then over again: after its pieces, the page shows
//! sums of them without end, each the exclusive or of about half the pieces of a block, picked by numbers
//! that follow from where the packet stands in the sequence. Whoever has as many packets of a block as the
//! block has pieces, whichever they are, can solve for the pieces, give or take a packet or two.
//!
//! A file is cut into symbols, which are what a frame holds, and the symbols into blocks of at most
//! [`BLOCK`], as equal as they come. The page computes the packets in JavaScript, so nothing here can
//! change without it.

use std::ops::Range;

/// The most symbols to a block. More would make the sums slower to compute and to solve, fewer would
/// leave more to chance in which packets of a block are caught.
pub const BLOCK: usize = 512;

/// How a file of some length is cut for frames that hold symbols of some size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub length: u64,
    /// Bytes to a symbol
    pub symbol: usize,
    /// The symbols of each block, counted through the file
    blocks: Vec<Range<usize>>,
}

impl Plan {
    pub fn new(length: u64, symbol: usize) -> Self {
        assert!(symbol > 0, "a symbol has to hold a byte at least");
        let symbols = length.div_ceil(symbol as u64) as usize;
        let count = symbols.div_ceil(BLOCK).max(1);
        let mut blocks = Vec::with_capacity(count);
        let mut from = 0;
        for block in 0..count {
            let size = symbols / count + usize::from(block < symbols % count);
            blocks.push(from..from + size);
            from += size;
        }
        Self {
            length,
            symbol,
            blocks,
        }
    }

    /// How many symbols the file has.
    pub fn symbols(&self) -> usize {
        self.blocks.last().map_or(0, |block| block.end)
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    /// How many symbols a block has.
    pub fn block(&self, block: usize) -> usize {
        self.blocks.get(block).map_or(0, Range::len)
    }

    /// The block and the place in its sequence of the packet that the page shows as its frame `frame`:
    /// the blocks take turns, so that all of them are complete at about the same time.
    pub fn turn(&self, frame: u64) -> (u32, u32) {
        let blocks = self.blocks.len() as u64;
        ((frame % blocks) as u32, (frame / blocks) as u32)
    }
}

/// Which symbols of a block of `size` a packet is the sum of, as bits: the packet `sequence` of a block
/// is its symbol of that number while there is one, and after that a choice that follows from the file's
/// checksum, the block and the number.
pub fn terms(checksum: u32, block: u32, sequence: u32, size: usize) -> Vec<u64> {
    let mut terms = vec![0u64; size.div_ceil(64)];
    if (sequence as usize) < size {
        terms[sequence as usize / 64] |= 1 << (sequence % 64);
        return terms;
    }
    // Every word of the choice is stirred on its own. Numbers that follow from each other by shifts and
    // exclusive ors alone would not do: all the sums they pick would follow from the first 32 of them.
    let seed = stir(checksum ^ block.wrapping_mul(0x9e37_79b1) ^ sequence.wrapping_mul(0x85eb_ca6b));
    let mut any = false;
    for half in 0..size.div_ceil(32) {
        let mut word = stir(seed.wrapping_add((half as u32 + 1).wrapping_mul(0x9e37_79b9)));
        let spare = (32 * (half + 1)).saturating_sub(size);
        if spare > 0 {
            word &= u32::MAX >> spare;
        }
        any |= word != 0;
        terms[half / 2] |= (word as u64) << (32 * (half % 2));
    }
    if !any && size > 0 {
        terms[0] = 1;
    }
    terms
}

/// A number that has little to do with `value` and nothing with what its neighbours turn into.
fn stir(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x85eb_ca6b);
    value ^= value >> 13;
    value = value.wrapping_mul(0xc2b2_ae35);
    value ^ value >> 16
}

fn add(sum: &mut [u8], term: &[u8]) {
    for (sum, term) in sum.iter_mut().zip(term) {
        *sum ^= term;
    }
}

/// Makes the packets of a file, as the page does.
pub struct Encoder<'a> {
    plan: Plan,
    checksum: u32,
    bytes: &'a [u8],
}

impl<'a> Encoder<'a> {
    pub fn new(bytes: &'a [u8], checksum: u32, symbol: usize) -> Self {
        Self {
            plan: Plan::new(bytes.len() as u64, symbol),
            checksum,
            bytes,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The packet `sequence` of a block: a symbol's worth of bytes.
    pub fn packet(&self, block: u32, sequence: u32) -> Vec<u8> {
        let symbols = self.plan.blocks[block as usize].clone();
        let mut packet = vec![0u8; self.plan.symbol];
        for (word, terms) in terms(self.checksum, block, sequence, symbols.len())
            .into_iter()
            .enumerate()
        {
            for bit in (0..64).filter(|bit| terms >> bit & 1 == 1) {
                let from = (symbols.start + 64 * word + bit) * self.plan.symbol;
                // the last symbol of the file ends where the file ends
                let to = (from + self.plan.symbol).min(self.bytes.len());
                add(&mut packet, &self.bytes[from..to]);
            }
        }
        packet
    }
}

/// A sum of symbols as it stands while a block is being solved.
struct Row {
    terms: Vec<u64>,
    bytes: Vec<u8>,
}

impl Row {
    /// The first symbol that the row has, if any.
    fn first(&self) -> Option<usize> {
        self.terms
            .iter()
            .position(|word| *word != 0)
            .map(|word| 64 * word + self.terms[word].trailing_zeros() as usize)
    }

    fn add(&mut self, other: &Row) {
        for (terms, other) in self.terms.iter_mut().zip(&other.terms) {
            *terms ^= other;
        }
        add(&mut self.bytes, &other.bytes);
    }
}

/// The packets of one block, kept as rows of which each begins with another symbol.
struct Block {
    /// By the symbol they begin with
    rows: Vec<Option<Row>>,
    held: usize,
}

impl Block {
    /// Takes a packet in; whether it told anything that the packets before it had not.
    fn take(&mut self, mut row: Row) -> bool {
        while let Some(first) = row.first() {
            match &self.rows[first] {
                Some(held) => row.add(held),
                None => {
                    self.rows[first] = Some(row);
                    self.held += 1;
                    return true;
                }
            }
        }
        false
    }

    fn is_solvable(&self) -> bool {
        self.held == self.rows.len()
    }

    /// The symbols of the block, once there is a row for each: from the last one back, every row is rid of
    /// the symbols after its first.
    fn solve(&mut self) -> Vec<Vec<u8>> {
        let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(self.rows.len());
        for place in (0..self.rows.len()).rev() {
            let mut row = self.rows[place].take().expect("a row for every symbol");
            for (word, terms) in row.terms.iter().enumerate() {
                for bit in (0..64).filter(|bit| terms >> bit & 1 == 1) {
                    let other = 64 * word + bit;
                    if other > place {
                        // solved before this one, and pushed in that order
                        add(&mut row.bytes, &symbols[self.rows.len() - 1 - other]);
                    }
                }
            }
            symbols.push(row.bytes);
        }
        symbols.reverse();
        symbols
    }
}

/// Puts a file together from packets.
pub struct Decoder {
    plan: Plan,
    checksum: u32,
    blocks: Vec<Block>,
}

impl Decoder {
    pub fn new(length: u64, checksum: u32, symbol: usize) -> Self {
        let plan = Plan::new(length, symbol);
        let blocks = (0..plan.blocks())
            .map(|block| Block {
                rows: (0..plan.block(block)).map(|_| None).collect(),
                held: 0,
            })
            .collect();
        Self {
            plan,
            checksum,
            blocks,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Takes a packet in. Whether it told anything new: a packet does not if it came before, or if it is
    /// a sum that follows from those that came before, or if it is none of this file.
    pub fn take(&mut self, block: u32, sequence: u32, bytes: &[u8]) -> bool {
        let Some(state) = self.blocks.get_mut(block as usize) else {
            return false;
        };
        if bytes.len() != self.plan.symbol || state.is_solvable() {
            return false;
        }
        state.take(Row {
            terms: terms(self.checksum, block, sequence, state.rows.len()),
            bytes: bytes.to_vec(),
        })
    }

    /// How many symbols' worth of packets have told something, of [`Plan::symbols`].
    pub fn held(&self) -> usize {
        self.blocks.iter().map(|block| block.held).sum()
    }

    /// How many bytes of the file that comes to.
    pub fn received(&self) -> u64 {
        if self.is_complete() {
            self.plan.length
        } else {
            (self.held() as u64 * self.plan.symbol as u64).min(self.plan.length)
        }
    }

    pub fn is_complete(&self) -> bool {
        self.blocks.iter().all(Block::is_solvable)
    }

    /// The file, once the packets are enough.
    pub fn into_bytes(mut self) -> Option<Vec<u8>> {
        if !self.is_complete() {
            return None;
        }
        let mut bytes = Vec::with_capacity(self.plan.symbols() * self.plan.symbol);
        for block in &mut self.blocks {
            for symbol in block.solve() {
                bytes.extend(symbol);
            }
        }
        bytes.truncate(self.plan.length as usize);
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Random(u32);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
    }

    fn file(length: usize, seed: u32) -> Vec<u8> {
        let mut random = Random(seed | 1);
        (0..length).map(|_| random.next() as u8).collect()
    }

    /// Shows the frames of a file one after the other to a camera that misses `lost` of every hundred;
    /// how many frames it takes until the file is whole.
    fn send(bytes: &[u8], symbol: usize, lost: u32, seed: u32) -> u64 {
        let encoder = Encoder::new(bytes, 0xfeed_beef, symbol);
        let mut decoder = Decoder::new(bytes.len() as u64, 0xfeed_beef, symbol);
        let mut random = Random(seed | 1);
        let mut frame = 0;
        while !decoder.is_complete() {
            let (block, sequence) = encoder.plan().turn(frame);
            frame += 1;
            if random.next() % 100 >= lost {
                decoder.take(block, sequence, &encoder.packet(block, sequence));
            }
            assert!(frame < 100_000, "the file never came together");
        }
        assert_eq!(decoder.received(), bytes.len() as u64);
        assert_eq!(decoder.into_bytes().as_deref(), Some(bytes));
        frame
    }

    #[test]
    fn file_is_cut_into_blocks_that_are_as_equal_as_they_come() {
        let plan = Plan::new(1000, 100);
        assert_eq!((plan.symbols(), plan.blocks(), plan.block(0)), (10, 1, 10));
        // 1300 symbols: three blocks of 434, 433, 433
        let plan = Plan::new(129_950, 100);
        assert_eq!(plan.symbols(), 1300);
        assert_eq!(
            (0..plan.blocks())
                .map(|block| plan.block(block))
                .collect::<Vec<_>>(),
            [434, 433, 433]
        );
        assert_eq!(plan.block(3), 0);
        // the blocks take turns
        assert_eq!(
            [plan.turn(0), plan.turn(1), plan.turn(2), plan.turn(3)],
            [(0, 0), (1, 0), (2, 0), (0, 1)]
        );
        assert_eq!(plan.turn(3 * 500 + 2), (2, 500));
        let empty = Plan::new(0, 100);
        assert_eq!((empty.symbols(), empty.blocks(), empty.block(0)), (0, 1, 0));
        assert_eq!(Plan::new(1, 100).symbols(), 1);
    }

    #[test]
    fn first_packets_are_the_symbols_and_the_others_sums_of_about_half() {
        assert_eq!(terms(7, 0, 0, 100), [1, 0]);
        assert_eq!(terms(7, 0, 70, 100), [0, 1 << 6]);
        let mut total = 0;
        for sequence in 100..200 {
            let sum = terms(7, 3, sequence, 100);
            assert_eq!(sum[1] >> 36, 0, "no symbols beyond the block");
            assert_ne!(sum, terms(7, 3, sequence + 1, 100));
            assert_ne!(sum, terms(7, 4, sequence, 100));
            assert_ne!(sum, terms(8, 3, sequence, 100));
            total += sum.iter().map(|word| word.count_ones()).sum::<u32>();
        }
        assert!((4500..5500).contains(&total), "{total}");
        // a block of one symbol has that symbol in every packet
        assert_eq!(terms(7, 0, 5, 1), [1]);
    }

    #[test]
    fn packets_are_sums_of_the_symbols_they_name() {
        let bytes = file(950, 3);
        let encoder = Encoder::new(&bytes, 9, 100);
        assert_eq!(encoder.packet(0, 4), bytes[400..500]);
        // the last symbol is filled up with nothing
        let last = encoder.packet(0, 9);
        assert_eq!((&last[..50], &last[50..]), (&bytes[900..], &[0u8; 50][..]));
        let terms = terms(9, 0, 15, 10)[0];
        let mut sum = vec![0u8; 100];
        for symbol in (0..10).filter(|symbol| terms >> symbol & 1 == 1) {
            add(&mut sum, &encoder.packet(0, symbol));
        }
        assert_eq!(encoder.packet(0, 15), sum);
    }

    #[test]
    fn file_comes_together_when_nothing_is_lost() {
        // as many frames as symbols
        assert_eq!(send(&file(5000, 1), 100, 0, 1), 50);
        assert_eq!(send(&file(4951, 2), 100, 0, 1), 50);
        assert_eq!(send(&file(1, 3), 100, 0, 1), 1);
        assert_eq!(send(&file(200_000, 4), 100, 0, 1), 2000);
    }

    #[test]
    fn file_comes_together_from_any_packets_that_are_enough() {
        for (length, lost) in [(50_000, 10), (50_000, 30), (50_000, 60), (171_717, 30), (700, 50)] {
            let bytes = file(length, 5);
            let symbols = length.div_ceil(100) as f64;
            let frames = send(&bytes, 100, lost, 77) as f64;
            // what a camera catches of the frames has to be the symbols and little more: a twentieth for
            // the luck of the blocks, and a few packets for those sums that follow from others
            let caught = frames * (100 - lost) as f64 / 100.0;
            assert!(
                caught <= symbols * 1.08 + 12.0,
                "{length} bytes, {lost} % lost: {frames} frames"
            );
            assert!(caught >= symbols * 0.9, "{frames} frames");
        }
    }

    #[test]
    fn sums_alone_are_enough() {
        // none of the first 300 packets, which are the symbols themselves
        let bytes = file(30_000, 6);
        let encoder = Encoder::new(&bytes, 1, 100);
        let mut decoder = Decoder::new(30_000, 1, 100);
        let mut taken = 0;
        for sequence in 300.. {
            taken += u32::from(decoder.take(0, sequence, &encoder.packet(0, sequence)));
            if decoder.is_complete() {
                // each told something new, and a handful of them did not
                assert!(sequence - 300 < 310, "{sequence}");
                break;
            }
        }
        assert_eq!(taken, 300);
        assert_eq!(decoder.into_bytes(), Some(bytes));
    }

    #[test]
    fn packets_that_tell_nothing_new_are_told_apart() {
        let bytes = file(1000, 7);
        let encoder = Encoder::new(&bytes, 1, 100);
        let mut decoder = Decoder::new(1000, 1, 100);
        assert!(decoder.take(0, 3, &encoder.packet(0, 3)));
        assert!(
            !decoder.take(0, 3, &encoder.packet(0, 3)),
            "the same packet again"
        );
        assert!(
            !decoder.take(1, 0, &encoder.packet(0, 0)),
            "a block that there is not"
        );
        assert!(!decoder.take(0, 0, &[0; 99]), "a packet of another size");
        assert_eq!((decoder.held(), decoder.received()), (1, 100));
        assert_eq!(decoder.plan().symbols(), 10);
        assert!(!decoder.is_complete());
        for sequence in 0..10 {
            decoder.take(0, sequence, &encoder.packet(0, sequence));
        }
        assert!(decoder.is_complete());
        assert!(
            !decoder.take(0, 11, &encoder.packet(0, 11)),
            "a block that is whole"
        );
        assert_eq!(decoder.into_bytes(), Some(bytes));
    }

    #[test]
    fn file_without_a_byte_is_whole_from_the_start() {
        let decoder = Decoder::new(0, 0, 100);
        assert!(decoder.is_complete());
        assert_eq!(decoder.into_bytes(), Some(Vec::new()));
        let unfinished = Decoder::new(10, 0, 100);
        assert_eq!(unfinished.into_bytes(), None);
    }
}
