//! Reads the frames of a camera that is pointed at the page of `qqcode-show`, up to the file that is on
//! show there.
//!
//! The show is a matrix of tiles on a web page, each frame with a packet of which any that are enough make
//! the file ([`crate::packet`], [`crate::fountain`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use opencv::core::Mat;

use crate::checks::{Checks, Note};
use crate::crc;
use crate::fountain::Decoder;
use crate::packet::{Frame, Origin};
use crate::scan::{Found, Outline, Scanner};
use crate::stored::Stored;

/// What became of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No matrix is made out
    NoCode,
    /// There is a matrix, which can't be read
    Unread,
    /// There is a matrix, which names a file that is longer than any that is taken
    Other,
    /// The start screen of `qqcode-show`: the file is still to come
    Ready,
    /// A matrix with a packet of the file
    Packet {
        /// Whether it told anything that the packets before it had not
        new: bool,
    },
    /// The file is whole, [`Pipeline::take_file`] has it. What comes of the same file after it says the
    /// same.
    Complete,
    /// All of the file has arrived and its checksum is wrong, or those of some of its sections are
    /// ([`Pipeline::checked`]), so something was misread. What was collected is dropped, and the file
    /// counts from the beginning.
    Corrupt,
}

/// What a code in a frame is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// It can't be read
    Unread,
    Start,
    /// A packet of the file
    Piece,
}

/// A code in a frame, for showing somebody who points a camera what it sees.
#[derive(Clone, Debug, PartialEq)]
pub struct Sighting {
    pub outline: Outline,
    pub kind: Kind,
    /// Tiles along the longer side of the matrix
    pub modules: usize,
    /// Pixels of the frame to a tile
    pub module_size: f32,
    /// The share of its bytes that were misread and put right
    pub wrong: f32,
}

/// The codes that were made out in a frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Survey {
    pub codes: Vec<Sighting>,
}

impl Survey {
    /// The code that the frame is taken by: one that is read if there is one, and one that can't be
    /// read if not.
    pub fn leading(&self) -> Option<&Sighting> {
        let rank = |sighting: &&Sighting| match sighting.kind {
            Kind::Start | Kind::Piece => 0,
            Kind::Unread => 1,
        };
        self.codes.iter().min_by_key(rank)
    }
}

/// A file is not taken to be longer than this, whatever a frame says: what has to be kept for it would
/// not fit anywhere.
const MAX_LENGTH: u64 = 1 << 40;

/// What has arrived of the file, where it is kept.
enum Transfer {
    Packets(Decoder),
    Stored(Stored),
}

#[derive(Default)]
pub struct Pipeline {
    scanner: Scanner,
    /// The file that is on show, going by the code read last
    current: Option<Origin>,
    /// What has arrived of that file
    transfer: Option<Transfer>,
    /// What the start screen calls the file
    name: Option<String>,
    /// How many files have been begun
    files: u64,
    /// Whether the file that is on show has been put together
    whole: bool,
    file: Option<Vec<u8>>,
    /// Where the file is put together on disk, and where its sums are kept meanwhile, if it is not to
    /// be kept in memory, see [`Pipeline::storing`]
    store: Option<(PathBuf, PathBuf)>,
    /// Where it is kept which packets are in the two, see [`Pipeline::resuming`]
    index: Option<PathBuf>,
    /// Whether a file has been begun on disk. What is there after that is what this run put there, and
    /// not what another one left.
    begun: bool,
    /// How many bytes there were of a file that is gone on with, until [`Pipeline::take_resumed`] asks
    resumed: Option<u64>,
    /// The blocks of a web page that are solved and kept in memory, as far as they reach
    kept: Vec<u8>,
    /// The checksums of the one file that is taken, see [`Pipeline::checked`]
    checks: Option<Arc<Checks>>,
    notes: Vec<Note>,
    survey: Survey,
}

impl Pipeline {
    /// A pipeline that keeps the file until it is whole and its checksum is right, and has it in
    /// [`Pipeline::take_file`] then.
    pub fn new() -> Self {
        Self::default()
    }

    /// A pipeline for files that are too long to keep in memory. The file is put together in the file
    /// `part` as its frames arrive, with the help of the file `sums`
    /// ([`crate::stored`]); both are made anew for every file that comes on show, unless what another
    /// run left of it is gone on with ([`Pipeline::resuming`]). Once the file is
    /// whole there the outcome is [`Outcome::Complete`], and it is for whoever named the two to see that
    /// `part` has the checksum of [`Pipeline::file`] and to do away with `sums`.
    pub fn storing(part: &Path, sums: &Path) -> Self {
        Self {
            store: Some((part.to_owned(), sums.to_owned())),
            ..Self::default()
        }
    }

    /// Goes on with a file that a run before this one did not finish, and leaves what this run does not
    /// finish for the next. This is for a pipeline that puts the file together on disk
    /// ([`Pipeline::storing`]): `index` is a third file, in which it is kept which packets are in the
    /// other two ([`crate::stored`]). It is written every few seconds while packets arrive, and by
    /// [`Pipeline::save`].
    ///
    /// If the three are there when the first packet of this run is read, and they are of the file on
    /// show in packets of the size on show, nothing is made anew: what had arrived counts, and
    /// [`Pipeline::take_resumed`] says how much that is. If they are of another file, or of packets of
    /// another size, that packet is an error, and they stay as they are: what took hours to read is not
    /// thrown away for a page that shows something else. What this run began itself is made anew for
    /// another file or another matrix, as without an index.
    pub fn resuming(mut self, index: &Path) -> Self {
        self.index = Some(index.to_owned());
        self
    }

    /// Takes no other file than the one that `checks` are of, which makes a frame of another file an
    /// error, and holds what arrives against them: a file that is put together on disk section by
    /// section as soon as each is whole, where a packet that was misread is looked for and left out
    /// ([`crate::stored`]); a file that is kept in memory when all of it has arrived, where sections
    /// that are wrong make the outcome [`Outcome::Corrupt`]. [`Pipeline::take_notes`] says what was
    /// found.
    pub fn checked(mut self, checks: Checks) -> Self {
        self.checks = Some(Arc::new(checks));
        self
    }

    /// Reads one frame, with blue, green and red for every pixel: the matrix is not read from any other.
    pub fn process(&mut self, frame: &Mat) -> Result<Outcome> {
        let found = self.scanner.scan(frame)?;
        self.take(found)
    }

    /// Takes in what a [`Scanner`] found in a frame. The frames may be scanned by several at a time, as
    /// long as what they find is taken in here in the order of the frames.
    pub fn take(&mut self, found: Vec<Found>) -> Result<Outcome> {
        let mut codes = Vec::with_capacity(found.len());
        let mut said = None;
        for code in found {
            let kind = match &code.frame {
                None => Kind::Unread,
                Some(Frame::Start { .. }) => Kind::Start,
                Some(Frame::Packet { .. }) => Kind::Piece,
            };
            codes.push(Sighting {
                outline: code.outline,
                kind,
                modules: code.modules,
                module_size: code.module_size(),
                wrong: code.wrong,
            });
            // one show at a time: the first of two is the one that counts
            said = said.or(code.frame);
        }
        self.survey = Survey { codes };
        let on_show = said.as_ref().map(Frame::origin);
        if let (Some(checks), Some(on_show)) = (&self.checks, on_show) {
            ensure!(
                checks.origin == on_show,
                "the file on show has {} bytes and the CRC-32 {:08x}, and the checksums are those of {:?}, a file \
                 of {} bytes with the CRC-32 {:08x}",
                on_show.length,
                on_show.checksum,
                checks.name,
                checks.origin.length,
                checks.origin.checksum
            );
        }
        Ok(match said {
            Some(frame) => self.take_frame(frame)?,
            None if self.survey.codes.is_empty() => Outcome::NoCode,
            None => Outcome::Unread,
        })
    }

    /// Makes `origin` the file that is on show. What was collected of another one is of no use for it.
    fn turn_to(&mut self, origin: Origin) {
        if self.current != Some(origin) {
            self.current = Some(origin);
            self.files += 1;
            self.whole = false;
            self.transfer = None;
            self.name = None;
            self.kept.clear();
        }
    }

    /// Takes in what a frame of the web page says.
    fn take_frame(&mut self, frame: Frame) -> Result<Outcome> {
        let origin = frame.origin();
        if origin.length > MAX_LENGTH {
            return Ok(Outcome::Other);
        }
        self.turn_to(origin);
        if self.whole {
            return Ok(Outcome::Complete);
        }
        let (block, sequence, bytes) = match frame {
            Frame::Start { name, .. } => {
                self.name = Some(name).filter(|name| !name.is_empty());
                if origin.length > 0 {
                    return Ok(Outcome::Ready);
                }
                // a file without a byte has no packets to wait for
                self.transfer = None;
                if self.store.is_some() {
                    self.begin(origin, 1)?;
                    self.whole = true;
                    return Ok(Outcome::Complete);
                }
                return Ok(self.finish(Some(Vec::new())));
            }
            Frame::Packet {
                block,
                sequence,
                bytes,
                ..
            } => (block, sequence, bytes),
        };
        // packets of another size are those of another matrix, with which the page begins anew
        let symbol = match &self.transfer {
            Some(Transfer::Packets(decoder)) => Some(decoder.plan().symbol),
            Some(Transfer::Stored(stored)) => Some(stored.plan().symbol),
            None => None,
        };
        if symbol != Some(bytes.len()) {
            self.kept.clear();
            self.transfer = Some(if self.store.is_some() {
                Transfer::Stored(self.begin(origin, bytes.len())?)
            } else {
                Transfer::Packets(Decoder::new(origin.length, origin.checksum, bytes.len()))
            });
        }
        let (new, whole) = match &mut self.transfer {
            Some(Transfer::Stored(stored)) => {
                let new = stored
                    .take(block, sequence, &bytes)
                    .context("can't write the file as it arrives")?;
                self.notes.extend(stored.take_notes());
                (new, stored.is_complete())
            }
            Some(Transfer::Packets(decoder)) => {
                let taken = decoder.take(block, sequence, &bytes);
                if let Some(solved) = taken.solved {
                    let (from, to) = (
                        solved.offset as usize,
                        solved.offset as usize + solved.bytes.len(),
                    );
                    if self.kept.len() < to {
                        self.kept.resize(to, 0);
                    }
                    self.kept[from..to].copy_from_slice(&solved.bytes);
                }
                (taken.new, decoder.is_complete())
            }
            None => unreachable!("just set"),
        };
        if !whole {
            return Ok(Outcome::Packet { new });
        }
        self.transfer = None;
        if self.store.is_some() {
            // the file is on disk, and whoever said where sees to its checksum
            self.whole = true;
            return Ok(Outcome::Complete);
        }
        let file = std::mem::take(&mut self.kept);
        let checked = Some(file).filter(|bytes| crc::checksum(bytes) == origin.checksum);
        Ok(self.finish(checked))
    }

    /// Begins the file `origin` on disk, in packets of `symbol` bytes: anew, or with what a run before
    /// this one left of it, see [`Pipeline::resuming`].
    fn begin(&mut self, origin: Origin, symbol: usize) -> Result<Stored> {
        let (part, sums) = self.store.clone().expect("a place for the file");
        let name = part.display();
        let left = match &self.index {
            // an index without the file it is of is of nothing
            Some(index) if !self.begun && part.exists() => {
                Stored::kept(index).with_context(|| format!("can't read {}", index.display()))?
            }
            _ => None,
        };
        self.begun = true;
        if let (Some(left), Some(index)) = (left, &self.index) {
            if (left.length, left.checksum) != (origin.length, origin.checksum) {
                bail!(
                    "{name} has what arrived of another file, one of {} bytes with the CRC-32 {:08x}, and is not \
                     thrown away for the file on show: it is to be moved or removed first",
                    left.length,
                    left.checksum
                );
            }
            if left.symbol != symbol {
                bail!(
                    "{name} has what arrived of this file in packets of {} bytes, and the page shows packets of \
                     {symbol} bytes now. The file is gone on with by the matrix that it was begun with; for \
                     another one, {name} is to be removed first, and the file begins anew",
                    left.symbol
                );
            }
            let stored = Stored::resume(&part, &sums, index, self.checks.clone())
                .with_context(|| format!("can't go on with {name} by {}", index.display()))?;
            self.resumed = Some(stored.received());
            return Ok(stored);
        }
        let mut stored = Stored::create(&part, &sums, origin.length, origin.checksum, symbol)
            .with_context(|| format!("can't write {name}"))?;
        if let Some(index) = &self.index {
            stored = stored
                .indexed(index)
                .with_context(|| format!("can't remove {}", index.display()))?;
        }
        Ok(match &self.checks {
            Some(checks) => stored.checked(checks.clone()),
            None => stored,
        })
    }

    /// Writes down what has arrived of a file that is put together on disk, for a run that goes on with
    /// it ([`Pipeline::resuming`]). Nothing is done without an index, or for a file that is kept in
    /// memory.
    pub fn save(&mut self) -> Result<()> {
        if let (Some(Transfer::Stored(stored)), Some(index)) = (&mut self.transfer, &self.index) {
            stored
                .save()
                .with_context(|| format!("can't write {}", index.display()))?;
        }
        Ok(())
    }

    /// Once after a file was gone on with rather than begun: how many bytes of it had arrived in the
    /// runs before this one.
    pub fn take_resumed(&mut self) -> Option<u64> {
        self.resumed.take()
    }

    /// Ends the collecting of the file: with the file, or with nothing if its checksum is wrong or those
    /// of its sections are.
    fn finish(&mut self, file: Option<Vec<u8>>) -> Outcome {
        let sound = |bytes: &Vec<u8>| {
            let Some(checks) = &self.checks else {
                return true;
            };
            let wrong = checks.wrong(bytes);
            if let Some(first) = wrong.first() {
                self.notes.push(Note::Wrong {
                    sections: wrong.len(),
                    from: checks.range(*first).start,
                });
            }
            wrong.is_empty()
        };
        match file.filter(sound) {
            Some(bytes) => {
                self.whole = true;
                self.file = Some(bytes);
                Outcome::Complete
            }
            None => Outcome::Corrupt,
        }
    }

    /// The file, once after [`Outcome::Complete`]; `None` for a file that was put together on disk.
    pub fn take_file(&mut self) -> Option<Vec<u8>> {
        self.file.take()
    }

    /// How many packets of the file have told something and how many it takes; `None` before its first
    /// packet and once it is whole.
    pub fn packets(&self) -> Option<(usize, usize)> {
        match &self.transfer {
            Some(Transfer::Packets(decoder)) => Some((decoder.held(), decoder.plan().symbols())),
            Some(Transfer::Stored(stored)) => Some((stored.held(), stored.plan().symbols())),
            None => None,
        }
    }

    /// Of a file that is put together on disk and held against its checksums, how many of its sections
    /// have theirs and how many it has.
    pub fn sound(&self) -> Option<(usize, usize)> {
        match &self.transfer {
            Some(Transfer::Stored(stored)) => stored.sound(),
            _ => None,
        }
    }

    /// What holding the file against its checksums came to since this was last asked.
    pub fn take_notes(&mut self) -> Vec<Note> {
        std::mem::take(&mut self.notes)
    }

    /// The file that is on show, going by the code read last.
    pub fn file(&self) -> Option<Origin> {
        self.current
    }

    /// How many bytes of the file that is on show have arrived, if any of its codes has been read.
    pub fn received(&self) -> Option<u64> {
        let file = self.current?;
        Some(match &self.transfer {
            Some(Transfer::Packets(decoder)) => decoder.received(),
            Some(Transfer::Stored(stored)) => stored.received(),
            None if self.whole => file.length,
            None => 0,
        })
    }

    /// What the start screen calls the file, if it was read.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// How many files have been begun: more than one if the codes of another file came into view before
    /// the first was whole, at which what had arrived of the first was dropped.
    pub fn files(&self) -> u64 {
        self.files
    }

    /// The codes that were made out in the frame taken in last.
    pub fn last_survey(&self) -> &Survey {
        &self.survey
    }
}
