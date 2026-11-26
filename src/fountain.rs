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

    /// The symbols of a block, counted through the file.
    pub fn range(&self, block: usize) -> Range<usize> {
        self.blocks.get(block).cloned().unwrap_or(0..0)
    }

    /// The block that a symbol is in, counted through the file; the last block for a symbol beyond it.
    pub fn block_of(&self, symbol: usize) -> usize {
        self.blocks
            .partition_point(|block| block.end <= symbol)
            .min(self.blocks.len() - 1)
    }
}

/// How many blocks the page shows by turns before it goes on to the next ones.
pub const GROUP: usize = 16;

/// In a round after the first, a block is shown this many sums in a hundred of its packets at least,
/// however few the first round has to spare.
pub const AGAIN: usize = 10;

/// The order in which the page shows the packets of a file.
///
/// The blocks of a group take turns, so that what a camera misses in a bad moment is shared out among
/// them, and a group is shown until each of its blocks has had its packets and some sums to spare, as
/// many in a hundred as `spare` says. Then the next group follows. Neither the page nor whoever reads it
/// then has to keep more than a group at a time, however long the file is.
///
/// After the last group the first comes again, for what the round left unfinished, and so on without end.
/// These rounds are short: they show sums alone, as many as the first round had to spare, and never the
/// same sum twice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    plan: Plan,
    spare: usize,
    /// The frame of the first round at which each group begins, and after them the frames of that round
    first: Vec<u64>,
    /// The same for each of the rounds after it
    later: Vec<u64>,
}

impl Schedule {
    pub fn new(plan: Plan, spare: usize) -> Self {
        let mut schedule = Self {
            plan,
            spare,
            first: vec![0],
            later: vec![0],
        };
        for group in 0..schedule.plan.blocks().div_ceil(GROUP) {
            let blocks = schedule.blocks(group).len() as u64;
            schedule
                .first
                .push(schedule.first[group] + blocks * schedule.each(group, true));
            schedule
                .later
                .push(schedule.later[group] + blocks * schedule.each(group, false));
        }
        schedule
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The blocks of a group.
    fn blocks(&self, group: usize) -> Range<usize> {
        GROUP * group..(GROUP * (group + 1)).min(self.plan.blocks())
    }

    /// How many frames each block of a group gets in the first round, which are its packets and the sums
    /// to spare, or in any round after it, which are sums. It goes by the largest block of the group,
    /// which is its first.
    fn each(&self, group: usize, first: bool) -> u64 {
        let size = self.plan.block(GROUP * group);
        if first {
            (size * (100 + self.spare)).div_ceil(100) as u64
        } else {
            (size * self.spare.max(AGAIN)).div_ceil(100) as u64
        }
    }

    /// How many frames the first round has: all the groups once, with their packets and sums.
    pub fn round(&self) -> u64 {
        *self.first.last().expect("the end of the last group")
    }

    /// How many frames each of the rounds after the first has.
    pub fn again(&self) -> u64 {
        *self.later.last().expect("the end of the last group")
    }

    /// How many frames it takes at least to have the file: up to the last packet of the last block, which
    /// is what a camera that misses nothing needs.
    pub fn least(&self) -> u64 {
        let last = self.first.len().saturating_sub(2);
        let blocks = self.blocks(last);
        self.first[last] + (blocks.len() * self.plan.block(blocks.start)) as u64
    }

    /// The block and the place in its sequence of the packet that the page shows as its frame `frame`,
    /// counted from 0 without end.
    pub fn turn(&self, frame: u64) -> (u32, u32) {
        if self.round() == 0 {
            // a file without a byte has no packets to show
            return (0, frame as u32);
        }
        // which round, how far into it, and where the groups begin in a round of that kind
        let (rounds, within, starts) = match frame.checked_sub(self.round()) {
            None => (None, frame, &self.first),
            Some(rest) => (Some(rest / self.again()), rest % self.again(), &self.later),
        };
        let group = starts.partition_point(|start| *start <= within) - 1;
        let blocks = self.blocks(group);
        let turn = within - starts[group];
        let block = blocks.start as u64 + turn % blocks.len() as u64;
        // the sums of a later round follow those of all the rounds before it
        let before = rounds.map_or(0, |rounds| {
            self.each(group, true) + rounds * self.each(group, false)
        });
        let sequence = before + turn / blocks.len() as u64;
        (block as u32, sequence as u32)
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

pub(crate) fn add(sum: &mut [u8], term: &[u8]) {
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
pub(crate) struct Row {
    pub(crate) terms: Vec<u64>,
    pub(crate) bytes: Vec<u8>,
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
pub(crate) struct Block {
    /// By the symbol they begin with
    rows: Vec<Option<Row>>,
    held: usize,
}

impl Block {
    /// A block of `size` symbols of which nothing is known yet.
    pub(crate) fn new(size: usize) -> Self {
        Self {
            rows: (0..size).map(|_| None).collect(),
            held: 0,
        }
    }

    /// Takes a packet in; whether it told anything that the packets before it had not.
    pub(crate) fn take(&mut self, mut row: Row) -> bool {
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

    pub(crate) fn is_solvable(&self) -> bool {
        self.held == self.rows.len()
    }

    /// The bytes of the block, once there is a row for each of its symbols: from the last one back, every
    /// row is rid of the symbols after its first.
    pub(crate) fn solve(&mut self) -> Vec<u8> {
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
        symbols.into_iter().rev().flatten().collect()
    }
}

/// How far a block has come.
enum State {
    /// No packet of it is held
    Waiting,
    Collecting(Block),
    /// It was solved and handed out
    Done,
}

/// A block of the file that is solved: its bytes and where in the file they go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solved {
    pub offset: u64,
    pub bytes: Vec<u8>,
}

/// What became of a packet.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Taken {
    /// Whether it told anything new: a packet does not if it came before, if it is a sum that follows
    /// from those that came before, if its block is solved, or if it is none of this file
    pub new: bool,
    /// The block that it completed, if it did
    pub solved: Option<Solved>,
}

/// Puts a file together from packets, block by block, in memory: a block is solved and handed out as soon
/// as its packets are enough, and nothing of it is kept after that. Of the blocks that are not solved the
/// packets are held, which for a long file that a camera reads badly is most of the file;
/// [`crate::stored::Stored`] keeps them on disk instead.
pub struct Decoder {
    plan: Plan,
    checksum: u32,
    blocks: Vec<State>,
    /// How many packets are held of blocks that are not solved yet
    held: usize,
    /// How many symbols the solved blocks have
    solved: usize,
}

impl Decoder {
    pub fn new(length: u64, checksum: u32, symbol: usize) -> Self {
        let plan = Plan::new(length, symbol);
        // a block without a symbol, which is what a file without a byte has, is solved as it is
        let blocks = (0..plan.blocks())
            .map(|block| {
                if plan.block(block) == 0 {
                    State::Done
                } else {
                    State::Waiting
                }
            })
            .collect();
        Self {
            plan,
            checksum,
            blocks,
            held: 0,
            solved: 0,
        }
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Takes a packet in.
    pub fn take(&mut self, block: u32, sequence: u32, bytes: &[u8]) -> Taken {
        let index = block as usize;
        let size = self.plan.block(index);
        if index >= self.blocks.len()
            || bytes.len() != self.plan.symbol
            || matches!(self.blocks[index], State::Done)
        {
            return Taken::default();
        }
        if matches!(self.blocks[index], State::Waiting) {
            self.blocks[index] = State::Collecting(Block::new(size));
        }
        let State::Collecting(state) = &mut self.blocks[index] else {
            unreachable!("just begun");
        };
        let new = state.take(Row {
            terms: terms(self.checksum, block, sequence, size),
            bytes: bytes.to_vec(),
        });
        self.held += usize::from(new);
        if !state.is_solvable() {
            return Taken { new, solved: None };
        }
        let mut bytes = state.solve();
        let offset = (self.plan.range(index).start * self.plan.symbol) as u64;
        // the last symbol of the file is filled up with nothing
        bytes.truncate((self.plan.length - offset).min(bytes.len() as u64) as usize);
        self.blocks[index] = State::Done;
        self.held -= size;
        self.solved += size;
        Taken {
            new,
            solved: Some(Solved { offset, bytes }),
        }
    }

    /// How many symbols' worth of packets count towards the file, of [`Plan::symbols`]: those of the
    /// blocks that are solved, and those that are held.
    pub fn held(&self) -> usize {
        self.solved + self.held
    }

    /// How many bytes of the file that comes to.
    pub fn received(&self) -> u64 {
        if self.is_complete() {
            self.plan.length
        } else {
            (self.held() as u64 * self.plan.symbol as u64).min(self.plan.length)
        }
    }

    /// Whether every block is solved and handed out.
    pub fn is_complete(&self) -> bool {
        self.blocks.iter().all(|state| matches!(state, State::Done))
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

    /// What a show of a file came to.
    struct Sent {
        /// How many frames it took until the file was whole
        frames: u64,
        /// The most packets that were held of blocks that were not solved
        most_held: usize,
    }

    /// Shows the frames of a file one after the other, with `spare` sums in a hundred to a round, to a
    /// camera that misses `lost` of every hundred.
    fn send(bytes: &[u8], symbol: usize, spare: usize, lost: u32) -> Sent {
        let encoder = Encoder::new(bytes, 0xfeed_beef, symbol);
        let schedule = Schedule::new(encoder.plan().clone(), spare);
        let mut decoder = Decoder::new(bytes.len() as u64, 0xfeed_beef, symbol);
        let mut random = Random(77);
        let mut read = vec![0u8; bytes.len()];
        let (mut frame, mut most_held) = (0, 0);
        while !decoder.is_complete() {
            let (block, sequence) = schedule.turn(frame);
            frame += 1;
            if random.next() % 100 >= lost {
                let taken = decoder.take(block, sequence, &encoder.packet(block, sequence));
                if let Some(solved) = taken.solved {
                    let at = solved.offset as usize;
                    read[at..at + solved.bytes.len()].copy_from_slice(&solved.bytes);
                }
            }
            most_held = most_held.max(decoder.held);
            assert!(frame < 400_000, "the file never came together");
        }
        assert_eq!(decoder.received(), bytes.len() as u64);
        assert!(read == bytes, "the file as it was sent");
        Sent {
            frames: frame,
            most_held,
        }
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
        assert_eq!((plan.range(1), plan.block(3), plan.range(3)), (434..867, 0, 0..0));
        assert_eq!(
            [0, 433, 434, 866, 867, 1299, 5000].map(|symbol| plan.block_of(symbol)),
            [0, 0, 1, 1, 2, 2, 2]
        );
        let empty = Plan::new(0, 100);
        assert_eq!((empty.symbols(), empty.blocks(), empty.block(0)), (0, 1, 0));
        assert_eq!(empty.block_of(0), 0);
        assert_eq!(Plan::new(1, 100).symbols(), 1);
    }

    #[test]
    fn blocks_of_a_short_file_take_turns_without_end() {
        // three blocks, the largest of 434 symbols: a round is 434 frames for each
        let schedule = Schedule::new(Plan::new(129_950, 100), 0);
        assert_eq!(schedule.round(), 3 * 434);
        let turns = [0, 1, 2, 3].map(|frame| schedule.turn(frame));
        assert_eq!(turns, [(0, 0), (1, 0), (2, 0), (0, 1)]);
        assert_eq!(schedule.least(), 3 * 434);
        // and the rounds after it go on where the first ended, with a tenth as many frames each
        assert_eq!(schedule.again(), 3 * 44);
        assert_eq!(schedule.turn(3 * 434 - 1), (2, 433));
        assert_eq!(schedule.turn(3 * 434), (0, 434));
        assert_eq!(schedule.turn(3 * 500 + 2), (2, 500));
        // with sums to spare the first round is longer, and the sequence the same
        let spared = Schedule::new(Plan::new(129_950, 100), 25);
        assert_eq!(
            (spared.round(), spared.again(), spared.least()),
            (3 * 543, 3 * 109, 3 * 434)
        );
        assert_eq!(spared.turn(3 * 500 + 2), (2, 500));
        assert_eq!(spared.turn(3 * 5000 + 1), (1, 5000));
        // a file without a byte has nothing to show
        let empty = Schedule::new(Plan::new(0, 100), 25);
        assert_eq!((empty.round(), empty.again(), empty.least()), (0, 0, 0));
    }

    #[test]
    fn blocks_of_a_long_file_are_shown_group_by_group() {
        // 40 blocks of 512 symbols: groups of 16, 16 and 8, and 640 frames for each block in a round
        let plan = Plan::new(40 * 512 * 100, 100);
        assert_eq!((plan.blocks(), plan.block(0), plan.block(39)), (40, 512, 512));
        let schedule = Schedule::new(plan, 25);
        assert_eq!(schedule.first, [0, 10_240, 20_480, 25_600]);
        assert_eq!(schedule.round(), 25_600);
        // what a camera that misses nothing needs: all but the sums of the last group
        assert_eq!(schedule.least(), 20_480 + 8 * 512);
        assert_eq!([schedule.turn(0), schedule.turn(17)], [(0, 0), (1, 1)]);
        assert_eq!(schedule.turn(10_239), (15, 639));
        assert_eq!(schedule.turn(10_240), (16, 0));
        assert_eq!(schedule.turn(20_480), (32, 0));
        assert_eq!(schedule.turn(20_489), (33, 1));
        assert_eq!(schedule.turn(25_599), (39, 639));
        // The second round begins with the first group again, and with sums that were not shown yet. It
        // has 128 of them for each block, as many as the first round had to spare.
        assert_eq!(
            (schedule.later.clone(), schedule.again()),
            (vec![0, 2048, 4096, 5120], 5120)
        );
        assert_eq!(schedule.turn(25_600), (0, 640));
        assert_eq!(schedule.turn(25_600 + 2048), (16, 640));
        assert_eq!(schedule.turn(25_600 + 5119), (39, 767));
        assert_eq!(schedule.turn(25_600 + 5120), (0, 768));
        assert_eq!(
            schedule.turn(25_600 + 7 * 5120 + 2048 + 17),
            (17, 640 + 7 * 128 + 1)
        );
        // in a round every block is shown every packet of its share once, and none of a round before
        let shown = |frames: std::ops::Range<u64>| {
            let mut shown = vec![Vec::new(); 40];
            for frame in frames {
                let (block, sequence) = schedule.turn(frame);
                shown[block as usize].push(sequence);
            }
            shown.iter_mut().for_each(|sequences| sequences.sort_unstable());
            shown
        };
        assert!(
            shown(0..25_600)
                .iter()
                .all(|sequences| *sequences == (0..640).collect::<Vec<_>>())
        );
        let third = shown(25_600 + 5120..25_600 + 2 * 5120);
        assert!(
            third
                .iter()
                .all(|sequences| *sequences == (768..896).collect::<Vec<_>>())
        );
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
        assert_eq!(send(&file(5000, 1), 100, 0, 0).frames, 50);
        assert_eq!(send(&file(4951, 2), 100, 25, 0).frames, 50);
        assert_eq!(send(&file(1, 3), 100, 25, 0).frames, 1);
        assert_eq!(send(&file(200_000, 4), 100, 0, 0).frames, 2000);
    }

    #[test]
    fn file_comes_together_from_any_packets_that_are_enough() {
        for (length, lost) in [(50_000, 10), (50_000, 30), (50_000, 60), (171_717, 30), (700, 50)] {
            let bytes = file(length, 5);
            let symbols = length.div_ceil(100) as f64;
            let frames = send(&bytes, 100, 0, lost).frames as f64;
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
    fn long_file_is_read_group_by_group_with_no_more_than_a_group_held() {
        // 40 blocks of 512 symbols, 25 sums to spare in a hundred, and a camera that misses 15 of a hundred
        let bytes = file(40 * 512 * 20, 6);
        let sent = send(&bytes, 20, 25, 15);
        // within the first round: a block has 640 frames, of which 544 are caught
        assert!(sent.frames <= 25_600, "{} frames", sent.frames);
        assert!(sent.most_held <= 16 * 512, "{} packets held", sent.most_held);
    }

    #[test]
    fn blocks_that_a_round_leaves_unfinished_are_finished_by_the_next() {
        // more is missed than a round has to spare: 448 of a block's 640 frames are caught at best
        let bytes = file(40 * 512 * 20, 7);
        let sent = send(&bytes, 20, 25, 30);
        // a round or two of sums after the first, which are a fifth as long
        assert!(
            (25_600..25_600 + 3 * 5120).contains(&sent.frames),
            "{} frames",
            sent.frames
        );
        // which takes keeping all that the first round brought
        assert!(sent.most_held > 30 * 448, "{} packets held", sent.most_held);
    }

    #[test]
    fn sums_alone_are_enough() {
        // none of the first 300 packets, which are the symbols themselves
        let bytes = file(30_000, 6);
        let encoder = Encoder::new(&bytes, 1, 100);
        let mut decoder = Decoder::new(30_000, 1, 100);
        let mut new = 0;
        for sequence in 300.. {
            let taken = decoder.take(0, sequence, &encoder.packet(0, sequence));
            new += u32::from(taken.new);
            if let Some(solved) = taken.solved {
                // each told something new, and a handful of them did not
                assert!(sequence - 300 < 310, "{sequence}");
                assert_eq!((solved.offset, solved.bytes), (0, bytes));
                break;
            }
        }
        assert_eq!(new, 300);
        assert!(decoder.is_complete());
    }

    #[test]
    fn packets_that_tell_nothing_new_are_told_apart() {
        let bytes = file(950, 7);
        let encoder = Encoder::new(&bytes, 1, 100);
        let mut decoder = Decoder::new(950, 1, 100);
        assert!(decoder.take(0, 3, &encoder.packet(0, 3)).new);
        assert!(
            !decoder.take(0, 3, &encoder.packet(0, 3)).new,
            "the same packet again"
        );
        assert!(
            !decoder.take(1, 0, &encoder.packet(0, 0)).new,
            "a block that there is not"
        );
        assert!(!decoder.take(0, 0, &[0; 99]).new, "a packet of another size");
        assert_eq!((decoder.held(), decoder.received()), (1, 100));
        assert_eq!(decoder.plan().symbols(), 10);
        assert!(!decoder.is_complete());
        let mut solved = None;
        for sequence in 0..10 {
            solved = solved.or(decoder.take(0, sequence, &encoder.packet(0, sequence)).solved);
        }
        // the block as long as the file is, not as its symbols are
        assert_eq!(
            solved,
            Some(Solved {
                offset: 0,
                bytes: bytes.clone()
            })
        );
        assert!(decoder.is_complete());
        assert_eq!((decoder.held(), decoder.received()), (10, 950));
        assert_eq!(
            decoder.take(0, 11, &encoder.packet(0, 11)),
            Taken::default(),
            "a block that is solved"
        );
    }

    #[test]
    fn file_without_a_byte_is_whole_from_the_start() {
        let decoder = Decoder::new(0, 0, 100);
        assert!(decoder.is_complete());
        assert_eq!(decoder.received(), 0);
        assert!(!Decoder::new(10, 0, 100).is_complete());
    }
}
