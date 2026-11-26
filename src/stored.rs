//! Puts a long file together on disk rather than in memory.
//!
//! Of the packets of a block ([`crate::fountain`]), the first are its symbols as they are. Each of those is
//! written where it belongs in the file the moment it arrives, and a block of which all of them arrive is
//! whole without more ado. The sums are written one after the other to a second file, and looked at only
//! when a block has as many packets as symbols: then the symbols it lacks are solved for, from the sums
//! less what is in the file already, and written into their places.
//!
//! Whatever is read is kept this way, at the price of disk space: once over for the file and as much again
//! as the camera misses. Nothing is held in memory but which packets there are, and one block while it is
//! solved.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;

use crate::fountain::{self, Block, Plan, Row};

/// What there is of one block.
struct Progress {
    /// Which of its symbols are in the file, as bits
    there: Vec<u64>,
    count: usize,
    /// The sums that arrived: their place in the block's sequence and in the file of sums
    sums: Vec<(u32, u64)>,
    /// How many sums there were when the block was last tried in vain
    tried: usize,
    done: bool,
}

/// A file that is being put together on disk.
pub struct Stored {
    plan: Plan,
    checksum: u32,
    file: File,
    sums: File,
    /// How many bytes of sums are written
    written: u64,
    blocks: Vec<Progress>,
}

impl Stored {
    /// Begins a file of `length` bytes with the checksum `checksum` in `file`, which is made anew, as is
    /// `sums` for the sums of its packets. `symbol` is how many bytes a packet has.
    pub fn create(file: &Path, sums: &Path, length: u64, checksum: u32, symbol: usize) -> io::Result<Self> {
        let open = |path: &Path| {
            File::options()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
        };
        let plan = Plan::new(length, symbol);
        let blocks = (0..plan.blocks())
            .map(|block| Progress {
                there: vec![0; plan.block(block).div_ceil(64)],
                count: 0,
                sums: Vec::new(),
                tried: 0,
                // a block without a symbol, which is what a file without a byte has, is whole as it is
                done: plan.block(block) == 0,
            })
            .collect();
        Ok(Self {
            file: open(file)?,
            sums: open(sums)?,
            plan,
            checksum,
            written: 0,
            blocks,
        })
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Where a symbol of a block lies in the file, and how many bytes of it there are: fewer than a
    /// packet has for the last symbol of the file.
    fn place(&self, block: usize, symbol: usize) -> (u64, usize) {
        let offset = ((self.plan.range(block).start + symbol) * self.plan.symbol) as u64;
        (
            offset,
            (self.plan.length - offset).min(self.plan.symbol as u64) as usize,
        )
    }

    /// Takes a packet in. Whether it is one that was not there before; a packet is not if it came before,
    /// if its block is whole, or if it is none of this file.
    pub fn take(&mut self, block: u32, sequence: u32, bytes: &[u8]) -> io::Result<bool> {
        let index = block as usize;
        let size = self.plan.block(index);
        if index >= self.blocks.len() || bytes.len() != self.plan.symbol || self.blocks[index].done {
            return Ok(false);
        }
        if (sequence as usize) < size {
            let (word, bit) = (sequence as usize / 64, 1u64 << (sequence % 64));
            if self.blocks[index].there[word] & bit != 0 {
                return Ok(false);
            }
            let (offset, length) = self.place(index, sequence as usize);
            self.file.write_all_at(&bytes[..length], offset)?;
            let progress = &mut self.blocks[index];
            progress.there[word] |= bit;
            progress.count += 1;
        } else {
            if self.blocks[index]
                .sums
                .iter()
                .any(|(known, _)| *known == sequence)
            {
                return Ok(false);
            }
            self.sums.write_all_at(bytes, self.written)?;
            self.blocks[index].sums.push((sequence, self.written));
            self.written += bytes.len() as u64;
        }
        let progress = &mut self.blocks[index];
        if progress.count == size {
            progress.done = true;
            progress.sums = Vec::new();
        } else if progress.count + progress.sums.len() >= size && progress.sums.len() > progress.tried {
            self.solve(index)?;
        }
        Ok(true)
    }

    /// Solves for the symbols of a block that are not in the file, if its sums are enough, and writes
    /// them there.
    fn solve(&mut self, index: usize) -> io::Result<()> {
        let (size, symbol) = (self.plan.block(index), self.plan.symbol);
        let progress = &self.blocks[index];
        let there = |at: usize| progress.there[at / 64] >> (at % 64) & 1 == 1;
        let missing = (0..size).filter(|at| !there(*at)).collect::<Vec<_>>();
        // the place of each symbol among those that are missing
        let mut unknown = vec![usize::MAX; size];
        for (place, at) in missing.iter().enumerate() {
            unknown[*at] = place;
        }
        // the block as far as it is in the file; what is not there reads as nothing
        let mut known = vec![0u8; size * symbol];
        let (from, mut read) = (self.place(index, 0).0, 0);
        while read < known.len() {
            match self.file.read_at(&mut known[read..], from + read as u64)? {
                0 => break,
                count => read += count,
            }
        }
        // every sum less the symbols that are known is a sum of missing ones
        let mut block = Block::new(missing.len());
        for (sequence, at) in &progress.sums {
            let mut bytes = vec![0u8; symbol];
            self.sums.read_exact_at(&mut bytes, *at)?;
            let mut terms = vec![0u64; missing.len().div_ceil(64)];
            for (word, bits) in fountain::terms(self.checksum, index as u32, *sequence, size)
                .into_iter()
                .enumerate()
            {
                for at in (0..64)
                    .filter(|bit| bits >> bit & 1 == 1)
                    .map(|bit| 64 * word + bit)
                {
                    match unknown[at] {
                        usize::MAX => fountain::add(&mut bytes, &known[at * symbol..][..symbol]),
                        place => terms[place / 64] |= 1 << (place % 64),
                    }
                }
            }
            block.take(Row { terms, bytes });
            if block.is_solvable() {
                break;
            }
        }
        if !block.is_solvable() {
            // some of the sums follow from the others: the next one that arrives may do
            self.blocks[index].tried = self.blocks[index].sums.len();
            return Ok(());
        }
        let solved = block.solve();
        for (place, at) in missing.iter().enumerate() {
            let (offset, length) = self.place(index, *at);
            self.file
                .write_all_at(&solved[place * symbol..][..length], offset)?;
        }
        let progress = &mut self.blocks[index];
        progress.done = true;
        progress.count = size;
        progress.sums = Vec::new();
        Ok(())
    }

    /// How many symbols' worth of packets count towards the file, of [`Plan::symbols`].
    pub fn held(&self) -> usize {
        self.blocks
            .iter()
            .enumerate()
            .map(|(index, progress)| (progress.count + progress.sums.len()).min(self.plan.block(index)))
            .sum()
    }

    /// How many bytes of the file that comes to.
    pub fn received(&self) -> u64 {
        if self.is_complete() {
            self.plan.length
        } else {
            (self.held() as u64 * self.plan.symbol as u64).min(self.plan.length)
        }
    }

    /// Whether every block is whole, and with them the file on disk.
    pub fn is_complete(&self) -> bool {
        self.blocks.iter().all(|progress| progress.done)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::fountain::{Encoder, Schedule};

    fn bytes(length: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as u8
            })
            .collect()
    }

    /// Two names for files that no other test uses.
    fn names() -> (PathBuf, PathBuf) {
        static COUNT: AtomicUsize = AtomicUsize::new(0);
        let name = format!(
            "qqcode-stored-{}-{}",
            std::process::id(),
            COUNT.fetch_add(1, Ordering::Relaxed)
        );
        let file = std::env::temp_dir().join(name);
        let sums = file.with_extension("sums");
        (file, sums)
    }

    /// Shows the frames of a file to a camera that misses `lost` of every hundred; how many frames it
    /// takes until the file is whole on disk, and how many bytes of sums were written for it.
    fn send(bytes: &[u8], symbol: usize, spare: usize, lost: u32) -> (u64, u64) {
        let (file, sums) = names();
        let encoder = Encoder::new(bytes, 0xfeed_beef, symbol);
        let schedule = Schedule::new(encoder.plan().clone(), spare);
        let mut stored = Stored::create(&file, &sums, bytes.len() as u64, 0xfeed_beef, symbol).unwrap();
        let (mut state, mut frame) = (77u32, 0);
        while !stored.is_complete() {
            let (block, sequence) = schedule.turn(frame);
            frame += 1;
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if state % 100 >= lost {
                stored
                    .take(block, sequence, &encoder.packet(block, sequence))
                    .unwrap();
            }
            assert!(frame < 400_000, "the file never came together");
        }
        assert_eq!(stored.received(), bytes.len() as u64);
        assert!(fs::read(&file).unwrap() == bytes, "the file as it was sent");
        let written = fs::metadata(&sums).unwrap().len();
        fs::remove_file(&file).unwrap();
        fs::remove_file(&sums).unwrap();
        (frame, written)
    }

    #[test]
    fn symbols_go_where_they_belong_and_need_no_sums() {
        // three blocks, the last symbol short; nothing lost: as many frames as symbols, and no sum kept
        assert_eq!(send(&bytes(129_950, 1), 100, 10, 0), (1300, 0));
        assert_eq!(send(&bytes(1, 2), 100, 10, 0), (1, 0));
    }

    #[test]
    fn symbols_that_are_missing_are_solved_for_from_the_sums() {
        // one in twenty lost, which the first round makes up for
        let file = bytes(129_950, 3);
        let (frames, sums) = send(&file, 100, 10, 5);
        assert!(
            frames <= Schedule::new(Plan::new(129_950, 100), 10).round(),
            "{frames} frames"
        );
        // the sums of a round are a tenth of the symbols, and no more of them are kept than arrive
        assert!(sums > 0 && sums <= 130 * 100, "{sums} bytes of sums");
        // the last symbol of the file among them, which is half a packet
        let (path, sums) = names();
        let encoder = Encoder::new(&file[..950], 5, 100);
        let mut stored = Stored::create(&path, &sums, 950, 5, 100).unwrap();
        for sequence in 0..9 {
            assert!(stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap());
        }
        // every other sum has the missing symbol in it
        for sequence in 10..40 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
            if stored.is_complete() {
                break;
            }
        }
        assert!(stored.is_complete());
        assert_eq!(fs::read(&path).unwrap(), &file[..950]);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn what_a_round_leaves_unfinished_is_finished_by_the_short_rounds_after_it() {
        // 40 blocks, three frames in ten lost and one in ten to spare: every block lacks a quarter of
        // its symbols after the first round, and every round after it brings a fourteenth
        let file = bytes(40 * 512 * 20, 4);
        let schedule = Schedule::new(Plan::new(file.len() as u64, 20), 10);
        let (frames, _) = send(&file, 20, 10, 30);
        let rounds = (frames - schedule.round()).div_ceil(schedule.again());
        assert!((3..=6).contains(&rounds), "{rounds} rounds after the first");
        // far less than showing all of it again
        assert!(frames < schedule.round() * 16 / 10, "{frames} frames");
    }

    #[test]
    fn sums_alone_make_a_block() {
        let file = bytes(30_000, 5);
        let (path, sums) = names();
        let encoder = Encoder::new(&file, 1, 100);
        let mut stored = Stored::create(&path, &sums, 30_000, 1, 100).unwrap();
        let mut taken = 0;
        for sequence in 300.. {
            taken += u32::from(stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap());
            if stored.is_complete() {
                break;
            }
            assert!(sequence < 620, "the block never came together");
        }
        // 300 sums and a handful more, for those that follow from others
        assert!((300..310).contains(&taken), "{taken}");
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn packets_that_are_there_already_are_told_apart() {
        let file = bytes(1000, 6);
        let (path, sums) = names();
        let encoder = Encoder::new(&file, 1, 100);
        let mut stored = Stored::create(&path, &sums, 1000, 1, 100).unwrap();
        assert!(stored.take(0, 3, &encoder.packet(0, 3)).unwrap());
        assert!(
            !stored.take(0, 3, &encoder.packet(0, 3)).unwrap(),
            "the same symbol again"
        );
        assert!(stored.take(0, 15, &encoder.packet(0, 15)).unwrap());
        assert!(
            !stored.take(0, 15, &encoder.packet(0, 15)).unwrap(),
            "the same sum again"
        );
        assert!(
            !stored.take(1, 0, &encoder.packet(0, 0)).unwrap(),
            "a block that there is not"
        );
        assert!(!stored.take(0, 0, &[0; 99]).unwrap(), "a packet of another size");
        assert_eq!((stored.held(), stored.received()), (2, 200));
        assert_eq!(stored.plan().symbols(), 10);
        for sequence in 0..10 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert!(stored.is_complete());
        assert_eq!((stored.held(), stored.received()), (10, 1000));
        assert!(
            !stored.take(0, 20, &encoder.packet(0, 20)).unwrap(),
            "a block that is whole"
        );
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn file_without_a_byte_is_whole_from_the_start() {
        let (path, sums) = names();
        let stored = Stored::create(&path, &sums, 0, 0, 100).unwrap();
        assert!(stored.is_complete());
        assert_eq!(fs::read(&path).unwrap(), b"");
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }
}
