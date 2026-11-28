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
//!
//! With the checksums of the file at hand ([`crate::checks`]), every section of it is held against its
//! checksum as soon as the blocks it lies in are whole. One that fails has a packet in it that was
//! misread, which the check of its frame let through one time in four thousand million. A block that is
//! whole therefore goes on taking sums until its sections are known to be right, [`SPARE`] of them at
//! most, because every sum more is a test of the block as it was solved: the tests that fail all miss by
//! the same bytes, and which of them fail is the mark of one packet. That packet is left out, the block
//! is made of the others, and the sections are held against their checksums again.
//!
//! A third file, the index, says which packets there are in the two: which symbols of each block arrived,
//! and which sums and where they lie. It is written again every few seconds, and a run that finds it goes
//! on with the file where the run before it stopped ([`Stored::resume`]). Only what was whole is looked
//! at again then, section by section against the checksums.

use std::fs::{self, File};
use std::io::{self, Write};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::checks::{Checks, Note};
use crate::crc;
use crate::fountain::{self, Block, Plan, Row};

/// How many sums more than it has symbols a block takes while it is whole and not yet known to be right.
/// Each halves the packets that may be the one that was misread, so these leave no doubt.
pub const SPARE: usize = 16;

/// What an index begins with; the number is that of the layout.
const INDEX: &[u8] = b"qqcode index 1\n";

/// How long packets arrive before the index is written again. A run that is cut off, with no time to
/// write it, has read the packets of these seconds for nothing; one that is stopped loses none.
const INDEX_EVERY: Duration = Duration::from_secs(5);

/// What an index says its files are of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kept {
    /// Bytes in the file
    pub length: u64,
    /// CRC-32 of the whole file
    pub checksum: u32,
    /// Bytes to a packet
    pub symbol: usize,
}

/// The bytes of an index that are still to be read.
struct Rest<'a>(&'a [u8]);

impl Rest<'_> {
    fn take<const COUNT: usize>(&mut self) -> io::Result<[u8; COUNT]> {
        let (taken, rest) = self
            .0
            .split_first_chunk()
            .ok_or_else(|| no_index("it ends too early"))?;
        self.0 = rest;
        Ok(*taken)
    }

    fn u32(&mut self) -> io::Result<u32> {
        self.take().map(u32::from_be_bytes)
    }

    fn u64(&mut self) -> io::Result<u64> {
        self.take().map(u64::from_be_bytes)
    }
}

fn no_index(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("this is no index: {why}"))
}

/// What an index says its files are of, how many bytes of sums are written, and what follows of it: the
/// blocks.
fn index_of(bytes: &[u8]) -> io::Result<(Kept, u64, Rest<'_>)> {
    let rest = bytes
        .strip_prefix(INDEX)
        .ok_or_else(|| no_index("it does not begin as one, or it has another layout"))?;
    let (rest, checksum) = rest
        .split_last_chunk()
        .ok_or_else(|| no_index("it ends too early"))?;
    if crc::checksum(&bytes[..bytes.len() - 4]) != u32::from_be_bytes(*checksum) {
        return Err(no_index("its checksum is wrong"));
    }
    let mut rest = Rest(rest);
    let kept = Kept {
        length: rest.u64()?,
        checksum: rest.u32()?,
        symbol: rest.u64()? as usize,
    };
    if kept.symbol == 0 {
        return Err(no_index("its packets have no bytes"));
    }
    let written = rest.u64()?;
    Ok((kept, written, rest))
}

/// What there is of one block.
struct Progress {
    /// Which of its symbols arrived as they are, as bits
    there: Vec<u64>,
    count: usize,
    /// The sums that arrived: their place in the block's sequence and in the file of sums
    sums: Vec<(u32, u64)>,
    /// How many sums there were when the block was last tried in vain
    tried: usize,
    /// Every symbol is in the file: it arrived or was solved for
    done: bool,
    /// Nothing is left to find out about it: it is whole, and the sections it lies in have their
    /// checksums, if there are checksums. It takes no packet after that.
    settled: bool,
    /// How many sums there were when a packet that was misread was last looked for in vain, for as long
    /// as a section that the block lies in is whole and wrong
    searched: Option<usize>,
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
    checks: Option<Arc<Checks>>,
    /// Which sections have their checksums
    sound: Vec<bool>,
    notes: Vec<Note>,
    /// Where it is kept which packets there are, if another run is to go on with them
    index: Option<PathBuf>,
    /// When the index was last written, and whether a packet has arrived since
    indexed_at: Instant,
    unwritten: bool,
}

fn bit(bits: &[u64], at: usize) -> bool {
    bits[at / 64] >> (at % 64) & 1 == 1
}

fn set(bits: &mut [u64], at: usize) {
    bits[at / 64] |= 1 << (at % 64);
}

/// The places of the bits that are set, from the first on.
fn ones(bits: &[u64]) -> impl Iterator<Item = usize> + '_ {
    bits.iter().enumerate().flat_map(|(word, bits)| {
        (0..64)
            .filter(move |bit| bits >> bit & 1 == 1)
            .map(move |bit| 64 * word + bit)
    })
}

fn mix(bits: &mut [u64], other: &[u64]) {
    for (bits, other) in bits.iter_mut().zip(other) {
        *bits ^= other;
    }
}

/// Whether two sets of bits have an odd number in common.
fn odd(bits: &[u64], other: &[u64]) -> bool {
    let common = bits
        .iter()
        .zip(other)
        .map(|(bits, other)| (bits & other).count_ones());
    common.sum::<u32>() % 2 == 1
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
                settled: plan.block(block) == 0,
                searched: None,
            })
            .collect();
        Ok(Self {
            file: open(file)?,
            sums: open(sums)?,
            plan,
            checksum,
            written: 0,
            blocks,
            checks: None,
            sound: Vec::new(),
            notes: Vec::new(),
            index: None,
            indexed_at: Instant::now(),
            unwritten: false,
        })
    }

    /// Keeps an index of the file in `index` from here on, for a run that goes on with it
    /// ([`Stored::resume`]). This is for a file that was just begun: an index that is there is of
    /// what the two files held before, and is removed.
    pub fn indexed(mut self, index: &Path) -> io::Result<Self> {
        match fs::remove_file(index) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
            _ => {}
        }
        self.index = Some(index.to_owned());
        Ok(self)
    }

    /// What the index `index` says its files are of; `None` if there is no such file.
    pub fn kept(index: &Path) -> io::Result<Option<Kept>> {
        match fs::read(index) {
            Ok(bytes) => Ok(Some(index_of(&bytes)?.0)),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error),
        }
    }

    /// Goes on with the file that a run before this one left in `file` and `sums`, by what `index` says
    /// there is of it, and keeps the index from here on.
    ///
    /// The index is never ahead of the two files, see [`Stored::save`], so every packet it names is
    /// there. What it calls whole is not taken on trust: with `checks`, which have to be those of the
    /// file, every section that is whole is held against its checksum again, and one that fails is seen
    /// to as if it had just become whole.
    pub fn resume(file: &Path, sums: &Path, index: &Path, checks: Option<Arc<Checks>>) -> io::Result<Self> {
        let bytes = fs::read(index)?;
        let (kept, written, mut rest) = index_of(&bytes)?;
        if checks
            .as_ref()
            .is_some_and(|checks| checks.origin.length != kept.length)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "the checksums are those of a file of another length",
            ));
        }
        let plan = Plan::new(kept.length, kept.symbol);
        let mut blocks = Vec::with_capacity(plan.blocks());
        for block in 0..plan.blocks() {
            let size = plan.block(block);
            let done = u8::from_be_bytes(rest.take()?) != 0;
            let there = (0..size.div_ceil(64))
                .map(|_| rest.u64())
                .collect::<io::Result<Vec<_>>>()?;
            if ones(&there).any(|at| at >= size) {
                return Err(no_index("it has a symbol beyond the end of a block"));
            }
            let sums = (0..rest.u32()?)
                .map(|_| Ok((rest.u32()?, rest.u64()?)))
                .collect::<io::Result<Vec<_>>>()?;
            if sums
                .iter()
                .any(|(_, at)| at.checked_add(kept.symbol as u64).is_none_or(|end| end > written))
            {
                return Err(no_index("it has a sum beyond the end of the sums"));
            }
            blocks.push(Progress {
                count: ones(&there).count(),
                there,
                sums,
                tried: 0,
                done: done || size == 0,
                settled: size == 0,
                searched: None,
            });
        }
        if !rest.0.is_empty() {
            return Err(no_index("it goes on after its last block"));
        }
        let open = |path: &Path| File::options().read(true).write(true).open(path);
        let (file, sums) = (open(file)?, open(sums)?);
        if sums.metadata()?.len() < written {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the sums end before the last one that the index names",
            ));
        }
        let mut stored = Self {
            file,
            sums,
            plan,
            checksum: kept.checksum,
            written,
            blocks,
            checks: None,
            sound: Vec::new(),
            notes: Vec::new(),
            index: Some(index.to_owned()),
            indexed_at: Instant::now(),
            unwritten: false,
        };
        if let Some(checks) = checks {
            stored = stored.checked(checks);
        }
        for block in 0..stored.blocks.len() {
            if stored.blocks[block].done && !stored.blocks[block].settled {
                stored.check(block)?;
            }
        }
        Ok(stored)
    }

    /// Writes the index, if one is kept and a packet has arrived since it was last written.
    ///
    /// The two files are on disk before the index that names what is in them, and the index takes the
    /// place of the one before it in one step. So whatever cuts a run off, the index that the next run
    /// finds is whole, and names nothing that is not there; what arrived after it was written is
    /// there and not named, which costs the reading of it again and nothing else.
    pub fn save(&mut self) -> io::Result<()> {
        let Some(index) = &self.index else {
            return Ok(());
        };
        if !self.unwritten {
            return Ok(());
        }
        self.file.sync_data()?;
        self.sums.sync_data()?;
        let mut bytes = INDEX.to_vec();
        bytes.extend(self.plan.length.to_be_bytes());
        bytes.extend(self.checksum.to_be_bytes());
        bytes.extend((self.plan.symbol as u64).to_be_bytes());
        bytes.extend(self.written.to_be_bytes());
        for progress in &self.blocks {
            bytes.push(u8::from(progress.done));
            for word in &progress.there {
                bytes.extend(word.to_be_bytes());
            }
            bytes.extend((progress.sums.len() as u32).to_be_bytes());
            for (sequence, at) in &progress.sums {
                bytes.extend(sequence.to_be_bytes());
                bytes.extend(at.to_be_bytes());
            }
        }
        bytes.extend(crc::checksum(&bytes).to_be_bytes());
        let mut aside = index.clone().into_os_string();
        aside.push(".new");
        let mut file = File::create(&aside)?;
        file.write_all(&bytes)?;
        file.sync_data()?;
        fs::rename(&aside, index)?;
        self.indexed_at = Instant::now();
        self.unwritten = false;
        Ok(())
    }

    /// Holds the file against its checksums as it arrives. This is for a file that was just begun.
    pub fn checked(mut self, checks: Arc<Checks>) -> Self {
        assert_eq!(
            checks.origin.length, self.plan.length,
            "the checksums of a file of this length"
        );
        self.sound = vec![false; checks.sections()];
        self.checks = Some(checks);
        self
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

    /// Takes a packet in. Whether it is one that was not there before and that is of any use; a packet
    /// is not if it came before, if nothing is left to find out about its block, or if it is none of
    /// this file.
    pub fn take(&mut self, block: u32, sequence: u32, bytes: &[u8]) -> io::Result<bool> {
        let new = self.take_in(block, sequence, bytes)?;
        if new {
            self.unwritten = true;
            if self.indexed_at.elapsed() >= INDEX_EVERY {
                self.save()?;
            }
        }
        Ok(new)
    }

    fn take_in(&mut self, block: u32, sequence: u32, bytes: &[u8]) -> io::Result<bool> {
        let index = block as usize;
        let size = self.plan.block(index);
        if index >= self.blocks.len() || bytes.len() != self.plan.symbol || self.blocks[index].settled {
            return Ok(false);
        }
        let progress = &self.blocks[index];
        if progress.done {
            // whole, and not yet known to be right: every sum more is a test of it
            let full = progress.count + progress.sums.len() >= size + SPARE;
            if (sequence as usize) < size || full || !self.keep(index, sequence, bytes)? {
                return Ok(false);
            }
            // A section of it that is whole and wrong is looked at again with every such sum. That
            // does not go by whether the block was in doubt when a section was last put right: packets
            // may be misread in more than one place.
            self.check(index)?;
            return Ok(true);
        }
        if (sequence as usize) < size {
            if bit(&progress.there, sequence as usize) {
                return Ok(false);
            }
            let (offset, length) = self.place(index, sequence as usize);
            self.file.write_all_at(&bytes[..length], offset)?;
            let progress = &mut self.blocks[index];
            set(&mut progress.there, sequence as usize);
            progress.count += 1;
        } else if !self.keep(index, sequence, bytes)? {
            return Ok(false);
        }
        let progress = &mut self.blocks[index];
        if progress.count == size {
            progress.done = true;
        } else if progress.count + progress.sums.len() >= size && progress.sums.len() > progress.tried {
            self.solve(index)?;
        }
        if self.blocks[index].done {
            self.check(index)?;
        }
        Ok(true)
    }

    /// Writes a sum of a block to the file of sums, unless it is there already.
    fn keep(&mut self, index: usize, sequence: u32, bytes: &[u8]) -> io::Result<bool> {
        let progress = &mut self.blocks[index];
        if progress.sums.iter().any(|(known, _)| *known == sequence) {
            return Ok(false);
        }
        self.sums.write_all_at(bytes, self.written)?;
        progress.sums.push((sequence, self.written));
        self.written += bytes.len() as u64;
        Ok(true)
    }

    /// The bytes of a block as they are in the file, with nothing where the file ends before the block
    /// does.
    fn read(&self, index: usize) -> io::Result<Vec<u8>> {
        let mut block = vec![0u8; self.plan.block(index) * self.plan.symbol];
        let (from, mut read) = (self.place(index, 0).0, 0);
        while read < block.len() {
            match self.file.read_at(&mut block[read..], from + read as u64)? {
                0 => break,
                count => read += count,
            }
        }
        Ok(block)
    }

    /// Solves for the symbols of a block that are not in the file, if its sums are enough, and writes
    /// them there.
    fn solve(&mut self, index: usize) -> io::Result<()> {
        let (size, symbol) = (self.plan.block(index), self.plan.symbol);
        let progress = &self.blocks[index];
        let missing = (0..size)
            .filter(|at| !bit(&progress.there, *at))
            .collect::<Vec<_>>();
        // the place of each symbol among those that are missing
        let mut unknown = vec![usize::MAX; size];
        for (place, at) in missing.iter().enumerate() {
            unknown[*at] = place;
        }
        // the block as far as it is in the file; what is not there is not looked at
        let known = self.read(index)?;
        // every sum less the symbols that are known is a sum of missing ones
        let mut block = Block::new(missing.len());
        for (sequence, at) in &progress.sums {
            let mut bytes = vec![0u8; symbol];
            self.sums.read_exact_at(&mut bytes, *at)?;
            let mut terms = vec![0u64; missing.len().div_ceil(64)];
            for at in ones(&fountain::terms(self.checksum, index as u32, *sequence, size)) {
                match unknown[at] {
                    usize::MAX => fountain::add(&mut bytes, &known[at * symbol..][..symbol]),
                    place => set(&mut terms, place),
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
        self.blocks[index].done = true;
        Ok(())
    }

    /// The sections that a block lies in; none without checksums.
    fn sections_of(&self, block: usize) -> Range<usize> {
        let Some(checks) = &self.checks else {
            return 0..0;
        };
        let symbols = self.plan.range(block);
        let from = (symbols.start * self.plan.symbol) as u64;
        let to = ((symbols.end * self.plan.symbol) as u64).min(self.plan.length);
        if from >= to {
            return 0..0;
        }
        (from / checks.section) as usize..((to - 1) / checks.section) as usize + 1
    }

    /// The blocks that a section lies in.
    fn blocks_of(&self, checks: &Checks, section: usize) -> Range<usize> {
        let (range, symbol) = (checks.range(section), self.plan.symbol as u64);
        let last = self
            .plan
            .block_of((range.end.saturating_sub(1) / symbol) as usize);
        self.plan.block_of((range.start / symbol) as usize)..last + 1
    }

    /// Whether every block that a section lies in is whole, so that the section is.
    fn is_whole(&self, checks: &Checks, section: usize) -> bool {
        self.blocks_of(checks, section)
            .all(|block| self.blocks[block].done)
    }

    /// Whether a section, which is whole, has its checksum: as it is in the file, or with the bytes
    /// `other` in place of those that the file has from `from` on.
    fn holds(&self, checks: &Checks, section: usize, other: Option<(u64, &[u8])>) -> io::Result<bool> {
        let range = checks.range(section);
        let mut bytes = vec![0u8; (range.end - range.start) as usize];
        self.file.read_exact_at(&mut bytes, range.start)?;
        if let Some((from, other)) = other {
            // what the two have in common
            let (start, end) = (range.start.max(from), range.end.min(from + other.len() as u64));
            if start < end {
                bytes[(start - range.start) as usize..(end - range.start) as usize]
                    .copy_from_slice(&other[(start - from) as usize..(end - from) as usize]);
            }
        }
        Ok(checks.holds(section, &bytes))
    }

    /// Nothing is left to find out about a block.
    fn settle(&mut self, block: usize) {
        let progress = &mut self.blocks[block];
        progress.settled = true;
        progress.searched = None;
        // the bytes of its sums stay where they are, there is no telling them apart from the others
        progress.sums = Vec::new();
    }

    /// Holds the sections that a block lies in against their checksums, as far as they are whole now
    /// that the block is, and settles the blocks whose sections all have theirs. Without checksums a
    /// block is settled as soon as it is whole.
    fn check(&mut self, index: usize) -> io::Result<()> {
        let Some(checks) = self.checks.clone() else {
            self.settle(index);
            return Ok(());
        };
        let sections = self.sections_of(index);
        if sections.is_empty() {
            self.settle(index);
            return Ok(());
        }
        for section in sections.clone() {
            if self.sound[section] || !self.is_whole(&checks, section) {
                continue;
            }
            if self.holds(&checks, section, None)? {
                self.sound[section] = true;
            } else {
                self.search(&checks, section)?;
            }
        }
        // this block and those that it shares a section with
        let around =
            self.blocks_of(&checks, sections.start).start..self.blocks_of(&checks, sections.end - 1).end;
        for block in around {
            let progress = &self.blocks[block];
            let sound = self.sections_of(block).all(|section| self.sound[section]);
            if progress.done && !progress.settled && sound {
                self.settle(block);
            }
        }
        Ok(())
    }

    /// Looks for the packet that was misread in a section that is whole and has not its checksum, among
    /// those of the blocks it lies in.
    fn search(&mut self, checks: &Checks, section: usize) -> io::Result<()> {
        let from = checks.range(section).start;
        // a block that is settled has had all its sections right, and has not changed since
        let mut suspects = self
            .blocks_of(checks, section)
            .filter(|block| !self.blocks[*block].settled)
            .collect::<Vec<_>>();
        if suspects.is_empty() {
            suspects = self.blocks_of(checks, section).collect();
        }
        // said once, however many sections one packet spoils
        if suspects
            .iter()
            .all(|block| self.blocks[*block].searched.is_none())
        {
            self.notes.push(Note::Doubted { section, from });
        }
        for block in &suspects {
            let sums = self.blocks[*block].sums.len();
            if self.blocks[*block].searched == Some(sums) {
                // nothing more to go by than the last time
                continue;
            }
            self.blocks[*block].searched = Some(sums);
            if !self.mend(checks, *block, section)? {
                continue;
            }
            // the file has what was found: held against the checksums as it is there
            for section in self.sections_of(*block) {
                if !self.sound[section] && self.is_whole(checks, section) {
                    self.sound[section] = self.holds(checks, section, None)?;
                }
            }
            if self.sound[section] {
                // The blocks whose sections are all right now are in no doubt any more. One that has
                // another section that is whole and wrong stays in doubt: that was said, and is still
                // to be put right.
                for block in &suspects {
                    let wrong = self
                        .sections_of(*block)
                        .any(|section| !self.sound[section] && self.is_whole(checks, section));
                    if !wrong {
                        self.blocks[*block].searched = None;
                    }
                }
                self.notes.push(Note::Mended { section, from });
                return Ok(());
            }
        }
        // Every sum that a block takes has arrived, and they do not tell: more than one packet was
        // misread, or the checksums are those of other bytes. What there is of the blocks is of no use.
        let full = |block: &usize| {
            let progress = &self.blocks[*block];
            progress.count + progress.sums.len() >= self.plan.block(*block) + SPARE
        };
        if suspects.iter().all(full) {
            for block in suspects {
                for section in self.sections_of(block) {
                    self.sound[section] = false;
                }
                let progress = &mut self.blocks[block];
                progress.there.fill(0);
                (progress.count, progress.tried) = (0, 0);
                progress.sums = Vec::new();
                (progress.done, progress.settled, progress.searched) = (false, false, None);
            }
            self.notes.push(Note::Dropped { section, from });
        }
        Ok(())
    }

    /// Looks for the one packet of a block that was misread, and makes the block of the others if that
    /// gives the sections that the block lies in their checksums, the section `failed` among them.
    ///
    /// The block is solved here as it was when it became whole, from the symbols that arrived and the
    /// first sums that were enough. The sums beyond those are tests of it: each follows from some of the
    /// sums before it, and mixed with them comes to nothing if all is right. If one packet is off by some
    /// bytes, every test that it has a part in comes to those bytes, and every symbol that was solved
    /// with its help is off by them. Which tests those are is known for every packet, so the packets
    /// that the tests point at are tried one by one.
    fn mend(&mut self, checks: &Checks, index: usize, failed: usize) -> io::Result<bool> {
        let (size, symbol) = (self.plan.block(index), self.plan.symbol);
        let (there, sums) = (self.blocks[index].there.clone(), self.blocks[index].sums.clone());
        let missing = (0..size).filter(|at| !bit(&there, *at)).collect::<Vec<_>>();
        let mut unknown = vec![usize::MAX; size];
        for (place, at) in missing.iter().enumerate() {
            unknown[*at] = place;
        }
        let mut block = self.read(index)?;
        // Every sum: which symbols it is of, which of the missing ones, and its bytes less the symbols
        // that arrived.
        let (mut of, mut left, mut less) = (Vec::new(), Vec::new(), Vec::new());
        for (sequence, at) in &sums {
            let mut bytes = vec![0u8; symbol];
            self.sums.read_exact_at(&mut bytes, *at)?;
            let terms = fountain::terms(self.checksum, index as u32, *sequence, size);
            let mut lacking = vec![0u64; missing.len().div_ceil(64)];
            for at in ones(&terms) {
                match unknown[at] {
                    usize::MAX => fountain::add(&mut bytes, &block[at * symbol..][..symbol]),
                    place => set(&mut lacking, place),
                }
            }
            of.push(terms);
            left.push(lacking);
            less.push(bytes);
        }
        let one = |sum: usize| {
            let mut bits = vec![0u64; sums.len().div_ceil(64)];
            set(&mut bits, sum);
            bits
        };
        // The sums in the order they arrived, each rid of the missing symbol it begins with by the sum
        // that is kept for that symbol, until it begins with one that no sum is kept for and is kept
        // itself, or nothing is left of it and it is a test. Which sums went into it goes along.
        let mut kept: Vec<Option<(Vec<u64>, Vec<u64>)>> = vec![None; missing.len()];
        let mut tests = Vec::new();
        for (sum, mut lacking) in left.into_iter().enumerate() {
            let mut mixed = one(sum);
            loop {
                let Some(first) = ones(&lacking).next() else {
                    tests.push(mixed);
                    break;
                };
                match &kept[first] {
                    Some((terms, with)) => {
                        mix(&mut lacking, terms);
                        mix(&mut mixed, with);
                    }
                    None => {
                        kept[first] = Some((lacking, mixed));
                        break;
                    }
                }
            }
        }
        if kept.iter().any(Option::is_none) {
            return Ok(false);
        }
        // From the last missing symbol back: the sums that make each of them when they are mixed.
        let mut made: Vec<Vec<u64>> = vec![Vec::new(); missing.len()];
        for place in (0..missing.len()).rev() {
            let (terms, mut mixed) = kept[place].take().expect("a sum for every missing symbol");
            for other in ones(&terms).filter(|other| *other > place) {
                mix(&mut mixed, &made[other]);
            }
            made[place] = mixed;
        }
        for (place, at) in missing.iter().enumerate() {
            let solved = &mut block[at * symbol..][..symbol];
            solved.fill(0);
            for sum in ones(&made[place]) {
                fountain::add(solved, &less[sum]);
            }
        }
        // What the tests come to: nothing, or the bytes that the packet is off by.
        let mut off: Option<Vec<u8>> = None;
        let mut fails = Vec::with_capacity(tests.len());
        for mixed in &tests {
            let mut rest = vec![0u8; symbol];
            for sum in ones(mixed) {
                fountain::add(&mut rest, &less[sum]);
            }
            let fail = rest.iter().any(|byte| *byte != 0);
            match &off {
                Some(off) if fail && *off != rest => return Ok(false),
                None if fail => off = Some(rest),
                _ => {}
            }
            fails.push(fail);
        }
        let Some(off) = off else {
            // no test, or none that the packet has a part in: the sums that come next may be
            return Ok(false);
        };
        // The sums that every symbol is in, for the symbols that arrived: those sums are off if the
        // symbol is.
        let mut within = vec![vec![0u64; sums.len().div_ceil(64)]; size];
        for (sum, terms) in of.iter().enumerate() {
            for at in ones(terms) {
                set(&mut within[at], sum);
            }
        }
        let from = self.place(index, 0).0;
        let arrived = (0..size).filter(|at| bit(&there, *at));
        for (sum, at) in (0..sums.len())
            .map(|sum| (Some(sum), None))
            .chain(arrived.map(|at| (None, Some(at))))
        {
            // the sums that are off if this is the packet
            let spoilt = match (sum, at) {
                (Some(sum), _) => one(sum),
                (_, Some(at)) => within[at].clone(),
                _ => unreachable!("a sum or a symbol"),
            };
            if tests
                .iter()
                .zip(&fails)
                .any(|(mixed, fail)| odd(mixed, &spoilt) != *fail)
            {
                continue;
            }
            // and the symbols that are, which are put right for a look at the checksums
            let wrong = missing
                .iter()
                .enumerate()
                .filter(|(place, _)| odd(&made[*place], &spoilt))
                .map(|(_, at)| *at)
                .chain(at)
                .collect::<Vec<_>>();
            let turn = |block: &mut [u8]| {
                for at in &wrong {
                    fountain::add(&mut block[at * symbol..][..symbol], &off);
                }
            };
            turn(&mut block);
            let mut right = true;
            for section in
                std::iter::once(failed).chain(self.sections_of(index).filter(|other| *other != failed))
            {
                if self.is_whole(checks, section) && !self.holds(checks, section, Some((from, &block)))? {
                    right = false;
                    break;
                }
            }
            if !right {
                turn(&mut block);
                continue;
            }
            for at in missing.iter().copied().chain(at) {
                let (offset, length) = self.place(index, at);
                self.file.write_all_at(&block[at * symbol..][..length], offset)?;
            }
            // the packet is left out from here on
            let progress = &mut self.blocks[index];
            if let Some(sum) = sum {
                progress.sums.remove(sum);
            }
            if let Some(at) = at {
                progress.there[at / 64] &= !(1 << (at % 64));
                progress.count -= 1;
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// How many symbols' worth of packets count towards the file, of [`Plan::symbols`].
    pub fn held(&self) -> usize {
        self.blocks
            .iter()
            .enumerate()
            .map(|(index, progress)| match progress.done {
                true => self.plan.block(index),
                false => (progress.count + progress.sums.len()).min(self.plan.block(index)),
            })
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

    /// With checksums at hand: how many sections of the file have theirs, and how many there are.
    pub fn sound(&self) -> Option<(usize, usize)> {
        let sound = self.sound.iter().filter(|sound| **sound).count();
        self.checks.as_ref().map(|_| (sound, self.sound.len()))
    }

    /// What holding the file against its checksums came to since this was last asked.
    pub fn take_notes(&mut self) -> Vec<Note> {
        std::mem::take(&mut self.notes)
    }

    /// Whether every block is whole, and with them the file on disk, and whether every section of it has
    /// its checksum, if there are checksums.
    pub fn is_complete(&self) -> bool {
        self.blocks.iter().all(|progress| progress.settled)
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
    /// takes until the file is whole on disk, and how many bytes of sums were written for it. With
    /// `section`, the file is held against its checksums in sections of that size as it arrives.
    fn send(bytes: &[u8], symbol: usize, spare: usize, lost: u32, section: Option<u64>) -> (u64, u64) {
        let (file, sums) = names();
        let encoder = Encoder::new(bytes, 0xfeed_beef, symbol);
        let schedule = Schedule::new(encoder.plan().clone(), spare);
        let mut stored = Stored::create(&file, &sums, bytes.len() as u64, 0xfeed_beef, symbol).unwrap();
        if let Some(section) = section {
            stored = stored.checked(Arc::new(Checks::in_sections("f", bytes, section)));
        }
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
        // nothing was misread, so nothing is said, and every section has its checksum
        assert_eq!(stored.take_notes(), []);
        let sections = section.map(|section| bytes.len().div_ceil(section as usize));
        assert_eq!(stored.sound(), sections.map(|sections| (sections, sections)));
        let written = fs::metadata(&sums).unwrap().len();
        fs::remove_file(&file).unwrap();
        fs::remove_file(&sums).unwrap();
        (frame, written)
    }

    #[test]
    fn symbols_go_where_they_belong_and_need_no_sums() {
        // three blocks, the last symbol short; nothing lost: as many frames as symbols, and no sum kept
        assert_eq!(send(&bytes(129_950, 1), 100, 10, 0, None), (1300, 0));
        assert_eq!(send(&bytes(1, 2), 100, 10, 0, None), (1, 0));
    }

    #[test]
    fn symbols_that_are_missing_are_solved_for_from_the_sums() {
        // one in twenty lost, which the first round makes up for
        let file = bytes(129_950, 3);
        let (frames, sums) = send(&file, 100, 10, 5, None);
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
        let (frames, sums) = send(&file, 20, 10, 30, None);
        let rounds = (frames - schedule.round()).div_ceil(schedule.again());
        assert!((3..=6).contains(&rounds), "{rounds} rounds after the first");
        // far less than showing all of it again
        assert!(frames < schedule.round() * 16 / 10, "{frames} frames");
        // Held against its checksums, in sections that are larger than a block and others that are
        // smaller, the file is whole after as many frames: all that is new is the sums that a block
        // takes while it waits for the blocks next to it.
        for section in [30_000, 3000] {
            let (checked, more) = send(&file, 20, 10, 30, Some(section));
            assert_eq!(checked, frames);
            assert!(
                (sums..=sums + 40 * 16 * 20).contains(&more),
                "{more} bytes of sums, {sums} without"
            );
        }
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

    /// A file that is to be put together and held against its checksums, in sections of `section`
    /// bytes, and what makes its packets.
    fn checked<'a>(file: &'a [u8], symbol: usize, section: u64) -> (Stored, Encoder<'a>, PathBuf, PathBuf) {
        let (path, sums) = names();
        let checks = Arc::new(Checks::in_sections("f", file, section));
        let stored = Stored::create(&path, &sums, file.len() as u64, 9, symbol).unwrap();
        (stored.checked(checks), Encoder::new(file, 9, symbol), path, sums)
    }

    /// A packet as a camera misreads it once in a long while, with the check of its frame none the
    /// wiser.
    fn misread(mut packet: Vec<u8>) -> Vec<u8> {
        let length = packet.len();
        for at in [3, 40, 41, 97] {
            packet[at % length] ^= 0x5a;
        }
        packet
    }

    /// Gives a block sums from `from` on until the file is whole; how many it took.
    fn sums_until_whole(stored: &mut Stored, encoder: &Encoder, block: u32, from: u32) -> u32 {
        for sequence in from.. {
            stored
                .take(block, sequence, &encoder.packet(block, sequence))
                .unwrap();
            if stored.is_complete() {
                return sequence + 1 - from;
            }
            assert!(sequence < from + 600, "the file never came together");
        }
        unreachable!()
    }

    #[test]
    fn sections_are_held_against_their_checksums_as_they_become_whole() {
        // three blocks, of the bytes up to 43400, 86700 and 129950, in thirteen sections of 10000
        let file = bytes(129_950, 7);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 10_000);
        assert_eq!(stored.sound(), Some((0, 13)));
        for sequence in 0..434 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        // four sections lie in the first block alone; the fifth waits for the second block
        assert_eq!(stored.sound(), Some((4, 13)));
        // and so does the first block, which takes sums meanwhile, as many as could tell a misread
        // packet and no more
        let taken =
            (434..470).filter(|sequence| stored.take(0, *sequence, &encoder.packet(0, *sequence)).unwrap());
        assert_eq!(taken.count(), SPARE);
        assert!(
            !stored.take(0, 5, &encoder.packet(0, 5)).unwrap(),
            "a symbol of a block that is whole"
        );
        for sequence in 0..433 {
            stored.take(1, sequence, &encoder.packet(1, sequence)).unwrap();
        }
        assert_eq!(stored.sound(), Some((8, 13)));
        // nothing is left to find out about the first block
        assert!(!stored.take(0, 480, &encoder.packet(0, 480)).unwrap());
        assert!(!stored.is_complete());
        for sequence in 0..433 {
            stored.take(2, sequence, &encoder.packet(2, sequence)).unwrap();
        }
        assert_eq!(stored.sound(), Some((13, 13)));
        assert!(stored.is_complete());
        assert_eq!((stored.held(), stored.received()), (1300, 129_950));
        assert_eq!(stored.take_notes(), []);
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn symbol_that_was_misread_is_found_by_the_sums_that_follow() {
        // one block of 300 symbols in eight sections; every seventh symbol is missed, and the one from
        // byte 4000 on is misread
        let file = bytes(30_000, 8);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 4000);
        for sequence in (0..300).filter(|sequence| sequence % 7 != 0) {
            let packet = encoder.packet(0, sequence);
            let packet = if sequence == 40 { misread(packet) } else { packet };
            stored.take(0, sequence, &packet).unwrap();
        }
        // 43 sums make the block whole, give or take, and wrong; a few more tell which packet it was
        let taken = sums_until_whole(&mut stored, &encoder, 0, 300);
        assert!((44..43 + SPARE as u32).contains(&taken), "{taken} sums");
        let notes = stored.take_notes();
        assert!(
            matches!(
                notes[..],
                [
                    Note::Doubted { section: 0 | 1, .. },
                    Note::Mended { section: 0 | 1, .. }
                ]
            ),
            "{notes:?}"
        );
        assert_eq!(stored.sound(), Some((8, 8)));
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn sum_that_was_misread_is_found_as_well() {
        let file = bytes(30_000, 9);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 4000);
        for sequence in (0..300).filter(|sequence| sequence % 7 != 0) {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        // the first sum is one that the block is solved with
        stored.take(0, 300, &misread(encoder.packet(0, 300))).unwrap();
        let taken = sums_until_whole(&mut stored, &encoder, 0, 301);
        assert!((43..43 + SPARE as u32).contains(&taken), "{taken} sums");
        let notes = stored.take_notes();
        assert!(
            matches!(notes[..], [Note::Doubted { .. }, Note::Mended { .. }]),
            "{notes:?}"
        );
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn symbol_that_was_misread_in_a_block_that_missed_nothing_is_found() {
        let file = bytes(30_000, 10);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 4000);
        for sequence in 0..300 {
            let packet = encoder.packet(0, sequence);
            let packet = if sequence == 123 { misread(packet) } else { packet };
            stored.take(0, sequence, &packet).unwrap();
        }
        // whole, and seven of its eight sections are right
        assert_eq!(stored.sound(), Some((7, 8)));
        assert_eq!(
            stored.take_notes(),
            [Note::Doubted {
                section: 3,
                from: 12_000
            }]
        );
        assert!(!stored.is_complete());
        // no sum was needed, and a handful find the symbol: each has it or has not
        let taken = sums_until_whole(&mut stored, &encoder, 0, 300);
        assert!(taken <= SPARE as u32, "{taken} sums");
        assert_eq!(
            stored.take_notes(),
            [Note::Mended {
                section: 3,
                from: 12_000
            }]
        );
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn packet_that_was_misread_is_looked_for_in_every_block_that_its_section_lies_in() {
        // the section from byte 40000 on lies in the first block and in the second, of which it has the
        // symbol from byte 44400 on, and that one is misread
        let file = bytes(129_950, 11);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 10_000);
        for block in 0..3 {
            for sequence in 0..encoder.plan().block(block as usize) as u32 {
                let packet = encoder.packet(block, sequence);
                let packet = if (block, sequence) == (1, 10) {
                    misread(packet)
                } else {
                    packet
                };
                stored.take(block, sequence, &packet).unwrap();
            }
        }
        assert_eq!(stored.sound(), Some((12, 13)));
        assert_eq!(
            stored.take_notes(),
            [Note::Doubted {
                section: 4,
                from: 40_000
            }]
        );
        // the sums of the first block find nothing wrong with it, as many as it takes
        for sequence in 434..480 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert_eq!((stored.take_notes(), stored.is_complete()), (vec![], false));
        // those of the second find the symbol
        let taken = sums_until_whole(&mut stored, &encoder, 1, 433);
        assert!(taken <= SPARE as u32, "{taken} sums");
        assert_eq!(
            stored.take_notes(),
            [Note::Mended {
                section: 4,
                from: 40_000
            }]
        );
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn block_that_no_one_packet_accounts_for_is_read_anew() {
        // two symbols are misread, each in its own way
        let file = bytes(30_000, 12);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 4000);
        for sequence in 0..300 {
            let mut packet = encoder.packet(0, sequence);
            match sequence {
                5 => packet = misread(packet),
                200 => packet[50] ^= 1,
                _ => {}
            }
            stored.take(0, sequence, &packet).unwrap();
        }
        assert_eq!(stored.sound(), Some((6, 8)));
        // every sum that the block takes, and they do not agree on what is off
        for sequence in 300..340 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        let notes = stored.take_notes();
        assert!(
            matches!(
                notes[..],
                [
                    Note::Doubted { section: 0, from: 0 },
                    Note::Dropped { section: 0, from: 0 }
                ]
            ),
            "{notes:?}"
        );
        // Nothing is left of the block but the sums that came after that, which are the first of it as
        // it is read anew, and it is whole when it has arrived again.
        assert_eq!(
            (stored.held(), stored.sound(), stored.is_complete()),
            (24, Some((0, 8)), false)
        );
        for sequence in 0..300 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert!(stored.is_complete());
        assert_eq!((stored.take_notes(), stored.sound()), (vec![], Some((8, 8))));
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn file_is_never_whole_with_the_checksums_of_other_bytes() {
        let file = bytes(30_000, 13);
        let mut other = file.clone();
        other[29_999] ^= 1;
        let (path, sums) = names();
        let checks = Arc::new(Checks::in_sections("f", &other, 4000));
        let mut stored = Stored::create(&path, &sums, 30_000, 9, 100)
            .unwrap()
            .checked(checks);
        let encoder = Encoder::new(&file, 9, 100);
        for round in 0..2 {
            for sequence in 0..340 {
                stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
            }
            // the last section is the one, each time the block is whole
            let notes = stored.take_notes();
            let (doubted, dropped) = (
                Note::Doubted {
                    section: 7,
                    from: 28_000,
                },
                Note::Dropped {
                    section: 7,
                    from: 28_000,
                },
            );
            assert_eq!(notes, [doubted, dropped], "round {round}");
            assert!(!stored.is_complete());
        }
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn packets_that_were_misread_here_and_there_in_a_long_show_are_all_found() {
        // ten blocks of 512 symbols in sections that are smaller than a block, a camera that misses a
        // frame in five, and one packet misread in each of five blocks: a symbol in three of them, the
        // first sum that arrives in two
        let file = bytes(10 * 512 * 20, 14);
        let (mut stored, encoder, path, sums) = checked(&file, 20, 3000);
        let schedule = Schedule::new(encoder.plan().clone(), 10);
        let mut summed = [false; 10];
        let (mut state, mut frame, mut notes) = (99u32, 0, Vec::new());
        while !stored.is_complete() {
            let (block, sequence) = schedule.turn(frame);
            frame += 1;
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let symbol = [1, 4, 7].contains(&block) && sequence == 100 + block;
            let sum = [2, 8].contains(&block) && sequence >= 512 && !summed[block as usize];
            if state % 100 < 20 && !symbol && !sum {
                continue;
            }
            summed[block as usize] |= sum;
            let packet = encoder.packet(block, sequence);
            let packet = if symbol || sum { misread(packet) } else { packet };
            stored.take(block, sequence, &packet).unwrap();
            notes.extend(stored.take_notes());
            assert!(frame < 400_000, "the file never came together");
        }
        let mended = notes.iter().filter(|note| matches!(note, Note::Mended { .. }));
        assert_eq!(mended.count(), 5, "{notes:?}");
        assert!(
            !notes.iter().any(|note| matches!(note, Note::Dropped { .. })),
            "{notes:?}"
        );
        assert_eq!(stored.sound(), Some((35, 35)));
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn block_is_solved_with_as_many_packets_as_it_has_symbols() {
        // ten symbols, of which the first is missed
        let file = bytes(1000, 16);
        let (path, sums) = names();
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, 1000, 9, 100).unwrap();
        for sequence in 1..10 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert!(!stored.is_complete());
        // one sum that the missing symbol is in makes ten packets, and they are the block
        let with_first = |sequence: &u32| fountain::terms(9, 0, *sequence, 10)[0] & 1 == 1;
        let sum = (10..).find(with_first).unwrap();
        assert!(stored.take(0, sum, &encoder.packet(0, sum)).unwrap());
        assert!(stored.is_complete());
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn what_has_arrived_is_never_more_than_the_file() {
        // Ten symbols, the last of them half a packet. Nine arrive, and a sum that the missing one is
        // not in, which is a packet that counts and no help.
        let file = bytes(950, 17);
        let (path, sums) = names();
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, 950, 9, 100).unwrap();
        for sequence in 1..10 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        let without_first = |sequence: &u32| fountain::terms(9, 0, *sequence, 10)[0] & 1 == 0;
        let sum = (10..).find(without_first).unwrap();
        assert!(stored.take(0, sum, &encoder.packet(0, sum)).unwrap());
        // ten packets of 100 bytes are held, of a file of 950
        assert_eq!(
            (stored.held(), stored.received(), stored.is_complete()),
            (10, 950, false)
        );
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn section_that_ends_where_a_block_ends_does_not_wait_for_the_next_block() {
        // three blocks, and sections as long as the first of them: 43400 bytes
        let file = bytes(129_950, 18);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 43_400);
        assert_eq!(stored.sound(), Some((0, 3)));
        for sequence in 0..434 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        // the first section is the first block and nothing of the second
        assert_eq!(stored.sound(), Some((1, 3)));
        // so nothing is left to find out about that block, and it takes no sum
        assert!(!stored.take(0, 434, &encoder.packet(0, 434)).unwrap());
        assert_eq!(fs::metadata(&sums).unwrap().len(), 0);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn symbols_of_a_block_that_is_whole_are_not_taken_again() {
        // the first of three blocks is whole and waits for the second, which the section from byte
        // 40000 on lies in as well
        let file = bytes(129_950, 15);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 10_000);
        for sequence in 0..434 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        // its symbols come round again, as when the page is begun anew: they tell nothing and are no
        // tests of the block
        for sequence in 0..40 {
            assert!(
                !stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap(),
                "symbol {sequence}"
            );
        }
        // its sums are tests, as many as a block takes, and nothing else is in the file of sums
        let taken =
            (434..470).filter(|sequence| stored.take(0, *sequence, &encoder.packet(0, *sequence)).unwrap());
        assert_eq!(taken.count(), SPARE);
        assert_eq!(fs::metadata(&sums).unwrap().len(), (SPARE * 100) as u64);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn misread_packet_is_found_by_tests_that_were_there_before_anything_looked_wrong() {
        // The first block has a part in the section from byte 40000 on, and in that part a symbol that
        // is misread: the one from byte 41000 on. Nothing shows as long as the second block is missing.
        let file = bytes(129_950, 19);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 10_000);
        for sequence in 0..434 {
            let packet = encoder.packet(0, sequence);
            let packet = if sequence == 410 { misread(packet) } else { packet };
            stored.take(0, sequence, &packet).unwrap();
        }
        assert_eq!((stored.sound(), stored.take_notes()), (Some((4, 13)), vec![]));
        // Meanwhile the block takes the sums that a whole block takes. They are tests of it: those that
        // the symbol is in fail, the others do not.
        for sequence in 434..470 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert_eq!(stored.take_notes(), []);
        // With the second block the section is whole, and wrong. No sum more is needed: the tests that
        // are there tell which packet it was.
        for sequence in 0..433 {
            stored.take(1, sequence, &encoder.packet(1, sequence)).unwrap();
        }
        assert_eq!(
            stored.take_notes(),
            [
                Note::Doubted {
                    section: 4,
                    from: 40_000
                },
                Note::Mended {
                    section: 4,
                    from: 40_000
                }
            ]
        );
        assert_eq!(stored.sound(), Some((8, 13)));
        for sequence in 0..433 {
            stored.take(2, sequence, &encoder.packet(2, sequence)).unwrap();
        }
        assert!(stored.is_complete());
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    #[test]
    fn misread_packets_in_two_blocks_next_to_each_other_are_both_found() {
        // The second block has a misread symbol in the section that it shares with the first, from byte
        // 40000 on, and is whole before the first, which has one in a section of its own, from byte
        // 10000 on.
        let file = bytes(129_950, 41);
        let (mut stored, encoder, path, sums) = checked(&file, 100, 10_000);
        for sequence in 0..433 {
            let packet = encoder.packet(1, sequence);
            let packet = if sequence == 10 { misread(packet) } else { packet };
            stored.take(1, sequence, &packet).unwrap();
        }
        // it waits for the first block, and takes its sums meanwhile
        for sequence in 433..470 {
            stored.take(1, sequence, &encoder.packet(1, sequence)).unwrap();
        }
        assert_eq!(stored.take_notes(), []);
        for sequence in 0..434 {
            let packet = encoder.packet(0, sequence);
            let packet = if sequence == 100 { misread(packet) } else { packet };
            stored.take(0, sequence, &packet).unwrap();
        }
        // Both sections are whole now, and wrong. The second block has the sums that tell which of its
        // packets it was; the first has none yet.
        assert_eq!(
            stored.take_notes(),
            [
                Note::Doubted {
                    section: 1,
                    from: 10_000
                },
                Note::Mended {
                    section: 4,
                    from: 40_000
                }
            ]
        );
        assert_eq!(stored.sound(), Some((7, 13)));
        // That one section was put right does not end the doubt about the other. The sums of the first
        // block are tests of it as they arrive, and tell in their turn; it is not said twice that the
        // section was wrong.
        let mut notes = Vec::new();
        for sequence in 434..470 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
            notes.extend(stored.take_notes());
        }
        assert_eq!(
            notes,
            [Note::Mended {
                section: 1,
                from: 10_000
            }]
        );
        assert_eq!(stored.sound(), Some((8, 13)));
        for sequence in 0..433 {
            stored.take(2, sequence, &encoder.packet(2, sequence)).unwrap();
        }
        assert!(stored.is_complete());
        assert_eq!(fs::read(&path).unwrap(), file);
        fs::remove_file(&path).unwrap();
        fs::remove_file(&sums).unwrap();
    }

    /// As [`send`], to a camera whose run stops after each of the frames `stops`: another one goes on
    /// with what it left, by the index. How many frames it takes until the file is whole.
    fn send_in_runs(
        bytes: &[u8],
        symbol: usize,
        spare: usize,
        lost: u32,
        section: Option<u64>,
        stops: &[u64],
    ) -> u64 {
        let (file, sums) = names();
        let index = file.with_extension("index");
        let checks = section.map(|section| Arc::new(Checks::in_sections("f", bytes, section)));
        let encoder = Encoder::new(bytes, 0xfeed_beef, symbol);
        let schedule = Schedule::new(encoder.plan().clone(), spare);
        let mut stored = Stored::create(&file, &sums, bytes.len() as u64, 0xfeed_beef, symbol)
            .unwrap()
            .indexed(&index)
            .unwrap();
        if let Some(checks) = &checks {
            stored = stored.checked(checks.clone());
        }
        let (mut state, mut frame) = (77u32, 0);
        while !stored.is_complete() {
            if stops.contains(&frame) {
                let before = (stored.held(), stored.received(), stored.sound());
                stored.save().unwrap();
                drop(stored);
                let kept = Kept {
                    length: bytes.len() as u64,
                    checksum: 0xfeed_beef,
                    symbol,
                };
                assert_eq!(Stored::kept(&index).unwrap(), Some(kept));
                stored = Stored::resume(&file, &sums, &index, checks.clone()).unwrap();
                // nothing that had arrived is missing, and every section that had its checksum has it
                assert_eq!((stored.held(), stored.received(), stored.sound()), before);
            }
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
        assert!(fs::read(&file).unwrap() == bytes, "the file as it was sent");
        assert_eq!(stored.take_notes(), []);
        fs::remove_file(&file).unwrap();
        fs::remove_file(&sums).unwrap();
        // there is none if the file was whole before the first stop
        let _ = fs::remove_file(&index);
        frame
    }

    #[test]
    fn file_is_gone_on_with_where_the_run_before_stopped() {
        // Three blocks of 433 symbols and a camera that misses one frame in twenty, or one in three,
        // which takes the short rounds after the first. Wherever the runs stop, in the first block or
        // between two, with sums waiting or without, the file is whole at the frame at which it is for
        // a camera that never stops. The last of these stops after every frame from the 1200th on,
        // which is with one block whole and the others not, and with some sections whole and others
        // not.
        let file = bytes(129_950, 21);
        let every = (1200..3000).collect::<Vec<_>>();
        for (lost, section) in [(5, None), (5, Some(10_000)), (33, None), (33, Some(10_000))] {
            let straight = send(&file, 100, 10, lost, section).0;
            assert!((1200..3000).contains(&straight), "{straight} frames");
            for stops in [
                &[1][..],
                &[217, 433],
                &[700, 701, 1250],
                &[1400],
                &[1500, 1700, 1900],
                &every,
            ] {
                assert_eq!(
                    send_in_runs(&file, 100, 10, lost, section, stops),
                    straight,
                    "{lost} in a hundred lost, sections of {section:?}, stopped after {stops:?}"
                );
            }
        }
    }

    #[test]
    fn packets_that_arrived_after_the_index_was_written_are_only_read_again() {
        // The run is cut off, with no time to write the index: the one it wrote last is some frames
        // old. In those frames the first block became whole and its sections got their checksums.
        let file = bytes(129_950, 22);
        let (path, sums) = names();
        let index = path.with_extension("index");
        let checks = Arc::new(Checks::in_sections("f", &file, 10_000));
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, file.len() as u64, 9, 100)
            .unwrap()
            .indexed(&index)
            .unwrap()
            .checked(checks.clone());
        // every fifth symbol of the first block is missed, and so are some of the sums
        let seen = |sequence: &u32| !sequence.is_multiple_of(5);
        for sequence in (0..460).filter(seen) {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert_eq!((stored.held(), stored.sound()), (368, Some((0, 13))));
        stored.save().unwrap();
        for sequence in (460..560).filter(seen) {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert_eq!((stored.held(), stored.sound()), (434, Some((4, 13))));
        drop(stored);
        // The index knows of none of that, and the file has it: symbols that the index does not name
        // are not looked at, so the block is as it was when the index was written.
        let mut stored = Stored::resume(&path, &sums, &index, Some(checks.clone())).unwrap();
        assert_eq!((stored.held(), stored.sound()), (368, Some((0, 13))));
        // the sums come again, and with them the block
        for sequence in (460..560).filter(seen) {
            assert!(stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap());
            if stored.sound() == Some((4, 13)) {
                break;
            }
        }
        assert_eq!((stored.held(), stored.sound()), (434, Some((4, 13))));
        for block in 1..3 {
            for sequence in 0..433 {
                stored
                    .take(block, sequence, &encoder.packet(block, sequence))
                    .unwrap();
            }
        }
        assert!(stored.is_complete());
        assert_eq!(stored.take_notes(), []);
        assert_eq!(fs::read(&path).unwrap(), file);
        for path in [&path, &sums, &index] {
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn index_is_written_every_few_seconds_while_packets_arrive() {
        let file = bytes(30_000, 23);
        let (path, sums) = names();
        let index = path.with_extension("index");
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, 30_000, 9, 100)
            .unwrap()
            .indexed(&index)
            .unwrap();
        for sequence in 0..10 {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        assert!(!index.exists(), "not with every packet");
        // a packet that tells nothing new is no reason to write it, however long ago the last time was
        stored.indexed_at = Instant::now().checked_sub(INDEX_EVERY).unwrap();
        assert!(!stored.take(0, 3, &encoder.packet(0, 3)).unwrap());
        assert!(!index.exists());
        assert!(stored.take(0, 10, &encoder.packet(0, 10)).unwrap());
        assert!(index.exists() && !path.with_extension("index.new").exists());
        drop(stored);
        let stored = Stored::resume(&path, &sums, &index, None).unwrap();
        assert_eq!((stored.held(), stored.received()), (11, 1100));
        // a file that is begun anew has nothing to do with the index of the one before it
        drop(stored);
        let stored = Stored::create(&path, &sums, 30_000, 9, 100).unwrap();
        assert!(index.exists());
        let mut stored = stored.indexed(&index).unwrap();
        assert!(!index.exists());
        // and one without an index writes none
        let mut plain = Stored::create(&path.with_extension("plain"), &sums, 30_000, 9, 100).unwrap();
        plain.indexed_at = Instant::now().checked_sub(INDEX_EVERY).unwrap();
        assert!(plain.take(0, 0, &encoder.packet(0, 0)).unwrap());
        plain.save().unwrap();
        // nothing has arrived of the first: nothing to write down
        stored.save().unwrap();
        assert!(!index.exists());
        for path in [&path, &sums, &path.with_extension("plain")] {
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn index_that_is_not_what_was_written_is_not_gone_on_with() {
        let file = bytes(30_000, 24);
        let (path, sums) = names();
        let index = path.with_extension("index");
        assert_eq!(Stored::kept(&index).unwrap(), None);
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, 30_000, 9, 100)
            .unwrap()
            .indexed(&index)
            .unwrap();
        // symbols and sums, so that the index has both
        for sequence in (0..40).chain(300..310) {
            stored.take(0, sequence, &encoder.packet(0, sequence)).unwrap();
        }
        stored.save().unwrap();
        drop(stored);
        let written = fs::read(&index).unwrap();
        let refused = |bytes: &[u8]| {
            fs::write(&index, bytes).unwrap();
            let kept = Stored::kept(&index).map(|_| ());
            let resumed = Stored::resume(&path, &sums, &index, None).map(|_| ());
            (
                kept.map_err(|error| error.kind()),
                resumed.map_err(|error| error.kind()),
            )
        };
        let no_index = Err(io::ErrorKind::InvalidData);
        // a byte that is off, anywhere; an index that is cut short; one of another layout
        for at in [0, 14, 20, 40, written.len() / 2, written.len() - 1] {
            let mut bytes = written.clone();
            bytes[at] ^= 1;
            assert_eq!(refused(&bytes), (no_index, no_index), "byte {at}");
        }
        for length in [0, 10, 30, written.len() - 1] {
            assert_eq!(
                refused(&written[..length]),
                (no_index, no_index),
                "{length} bytes"
            );
        }
        // One that is whole and says what the files do not hold: the checksum of an index is no proof
        // of where it comes from.
        let signed = |mut bytes: Vec<u8>| {
            let end = bytes.len() - 4;
            let checksum = crc::checksum(&bytes[..end]);
            bytes[end..].copy_from_slice(&checksum.to_be_bytes());
            bytes
        };
        let header = INDEX.len() + 28;
        // a symbol beyond the 300 that the block has
        let mut bytes = written.clone();
        bytes[header + 1 + 4 * 8 + 1] |= 0x80;
        assert_eq!(refused(&signed(bytes)), (Ok(()), no_index));
        // a sum that lies beyond the sums that are written
        let mut bytes = written.clone();
        let sum = header + 1 + 5 * 8 + 4;
        bytes[sum + 4..sum + 12].copy_from_slice(&1000u64.to_be_bytes());
        assert_eq!(refused(&signed(bytes)), (Ok(()), no_index));
        // more blocks than the file has
        let mut bytes = written.clone();
        bytes.splice(written.len() - 4..written.len() - 4, [0; 5]);
        assert_eq!(refused(&signed(bytes)), (Ok(()), no_index));
        // the index as it was written, with sums that end before the last one it names
        fs::write(&sums, [0; 950]).unwrap();
        assert_eq!(refused(&written), (Ok(()), Err(io::ErrorKind::UnexpectedEof)));
        // and with the checksums of a file that is not as long
        let other = Arc::new(Checks::in_sections("f", &file[1..], 4000));
        fs::write(&sums, [0; 1000]).unwrap();
        let resumed = Stored::resume(&path, &sums, &index, Some(other)).map(|_| ());
        assert_eq!(
            resumed.map_err(|error| error.kind()),
            Err(io::ErrorKind::InvalidInput)
        );
        assert_eq!(refused(&written), (Ok(()), Ok(())));
        for path in [&path, &sums, &index] {
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn packet_that_was_misread_in_one_run_is_found_in_the_next() {
        // as in `symbol_that_was_misread_is_found_by_the_sums_that_follow`, with a run that stops when
        // the block is whole and wrong, and before the sums have told which packet it was
        let file = bytes(30_000, 8);
        let (path, sums) = names();
        let index = path.with_extension("index");
        let checks = Arc::new(Checks::in_sections("f", &file, 4000));
        let encoder = Encoder::new(&file, 9, 100);
        let mut stored = Stored::create(&path, &sums, 30_000, 9, 100)
            .unwrap()
            .indexed(&index)
            .unwrap()
            .checked(checks.clone());
        for sequence in (0..300).filter(|sequence| sequence % 7 != 0) {
            let packet = encoder.packet(0, sequence);
            let packet = if sequence == 40 { misread(packet) } else { packet };
            stored.take(0, sequence, &packet).unwrap();
        }
        let mut next = 300;
        while stored.take_notes().is_empty() {
            stored.take(0, next, &encoder.packet(0, next)).unwrap();
            next += 1;
        }
        assert!(!stored.is_complete());
        let sound = stored.sound();
        stored.save().unwrap();
        drop(stored);
        // The next run finds the same sections wrong, says so, and goes on looking with the sums that
        // the first one kept.
        let mut stored = Stored::resume(&path, &sums, &index, Some(checks)).unwrap();
        assert_eq!(stored.sound(), sound);
        let notes = stored.take_notes();
        assert!(
            matches!(notes[..], [Note::Doubted { section: 0 | 1, .. }]),
            "{notes:?}"
        );
        let taken = sums_until_whole(&mut stored, &encoder, 0, next);
        assert!(next - 300 + taken < 43 + SPARE as u32, "{taken} sums more");
        let notes = stored.take_notes();
        assert!(
            matches!(notes[..], [Note::Mended { section: 0 | 1, .. }]),
            "{notes:?}"
        );
        assert_eq!(stored.sound(), Some((8, 8)));
        assert_eq!(fs::read(&path).unwrap(), file);
        for path in [&path, &sums, &index] {
            fs::remove_file(path).unwrap();
        }
    }
}
