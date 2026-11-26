//! Reads the frames of a camera that is pointed at `qqcode-show`, up to the file that is on show there.
//!
//! The show is either of two: QR codes on a terminal, each with a piece of the file and its place
//! ([`crate::protocol`], [`crate::assembly`]), or a matrix of tiles on a web page, each frame with a packet
//! of which any that are enough make the file ([`crate::packet`], [`crate::fountain`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use opencv::core::Mat;

use crate::assembly::Assembly;
use crate::checks::{Checks, Note};
use crate::fountain::Decoder;
use crate::packet::{Frame, Origin};
use crate::protocol::{self, Message};
use crate::scan::{Content, Found, Outline, Scanner};
use crate::stored::Stored;

/// What became of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Neither a QR code nor a matrix is made out
    NoCode,
    /// There is a QR code or a matrix, which can't be read
    Unread,
    /// There is a QR code, which is not one of `qqcode-show`
    Other,
    /// The start screen of `qqcode-show`: the file is still to come
    Ready,
    /// A QR code with a piece of the file
    Piece {
        /// How many of its bytes had not arrived before; none for a piece that is read again
        new: usize,
        /// How many bytes between the piece before and this one are missing: those went by unread
        skipped: usize,
    },
    /// A matrix with a packet of the file
    Packet {
        /// Whether it told anything that the packets before it had not
        new: bool,
    },
    /// The end screen of `qqcode-show`, with parts of the file still missing
    Done,
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
    /// Not a code of `qqcode-show`
    Other,
    Start,
    /// A piece or a packet of the file
    Piece,
    End,
}

/// A code in a frame, for showing somebody who points a camera what it sees.
#[derive(Clone, Debug, PartialEq)]
pub struct Sighting {
    pub outline: Outline,
    pub kind: Kind,
    /// Modules along a side of a QR code, tiles along the longer side of a matrix
    pub modules: usize,
    /// Pixels of the frame to a module or tile
    pub module_size: f32,
    /// Of a matrix, the share of its bytes that were misread and put right; `None` for a QR code
    pub wrong: Option<f32>,
}

impl Sighting {
    /// Whether this is a matrix of tiles rather than a QR code.
    pub fn is_matrix(&self) -> bool {
        self.wrong.is_some()
    }
}

/// The codes that were made out in a frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Survey {
    pub codes: Vec<Sighting>,
}

impl Survey {
    /// The code that the frame is taken by: one of `qqcode-show` if there is one, any other one that is
    /// decoded if not, and one that can't be decoded last.
    pub fn leading(&self) -> Option<&Sighting> {
        let rank = |sighting: &&Sighting| match sighting.kind {
            Kind::Start | Kind::Piece | Kind::End => 0,
            Kind::Other => 1,
            Kind::Unread => 2,
        };
        self.codes.iter().min_by_key(rank)
    }
}

/// A file is not taken to be longer than this, whatever a frame says: what has to be kept for it would
/// not fit anywhere.
const MAX_LENGTH: u64 = 1 << 40;

/// What has arrived of the file, in the way it is shown.
enum Transfer {
    Pieces(Assembly),
    Packets(Decoder),
    Stored(Stored),
}

/// What a code of `qqcode-show` says, in either of its two ways.
enum Said {
    Code(Message),
    Matrix(Frame),
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
    /// Where the piece read last ends, or 0 after the start screen: where the next one begins if none is
    /// left out
    reached: Option<usize>,
    /// How many files have been begun
    files: u64,
    /// Whether the file that is on show has been put together
    whole: bool,
    file: Option<Vec<u8>>,
    /// Where a file that a web page shows is put together on disk, and where its sums are kept
    /// meanwhile, if it is not to be kept in memory, see [`Pipeline::storing`]
    store: Option<(PathBuf, PathBuf)>,
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

    /// A pipeline for files that are too long to keep in memory. The file that a web page shows is put
    /// together in the file `part` as its frames arrive, with the help of the file `sums`
    /// ([`crate::stored`]); both are made anew for every file that comes on show. Once the file is
    /// whole there the outcome is [`Outcome::Complete`], and it is for whoever named the two to see that
    /// `part` has the checksum of [`Pipeline::file`] and to do away with `sums`. A file that is shown in
    /// QR codes is short, and kept in memory as ever.
    pub fn storing(part: &Path, sums: &Path) -> Self {
        Self {
            store: Some((part.to_owned(), sums.to_owned())),
            ..Self::default()
        }
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

    /// Reads one frame, in grey or with blue, green and red.
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
            let (modules, module_size, outline) = (code.modules, code.module_size(), code.outline);
            let (kind, wrong, says) = match code.content {
                Content::Code(None) => (Kind::Unread, None, None),
                Content::Code(Some(bytes)) => match Message::parse(&bytes) {
                    None => (Kind::Other, None, None),
                    Some(message) => {
                        let kind = match message {
                            Message::Start { .. } => Kind::Start,
                            Message::Piece { .. } => Kind::Piece,
                            Message::End { .. } => Kind::End,
                        };
                        (kind, None, Some(Said::Code(message)))
                    }
                },
                Content::Matrix { frame: None, wrong } => (Kind::Unread, Some(wrong), None),
                Content::Matrix {
                    frame: Some(frame),
                    wrong,
                } => {
                    let kind = match frame {
                        Frame::Start { .. } => Kind::Start,
                        Frame::Packet { .. } => Kind::Piece,
                    };
                    (kind, Some(wrong), Some(Said::Matrix(frame)))
                }
            };
            codes.push(Sighting {
                outline,
                kind,
                modules,
                module_size,
                wrong,
            });
            // one show at a time: the first of two is the one that counts
            said = said.or(says);
        }
        self.survey = Survey { codes };
        let on_show = match &said {
            Some(Said::Code(message)) => Some(Origin {
                checksum: message.file().checksum,
                length: message.file().length as u64,
            }),
            Some(Said::Matrix(frame)) => Some(frame.origin()),
            None => None,
        };
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
            Some(Said::Code(message)) => self.take_message(message),
            Some(Said::Matrix(frame)) => self.take_frame(frame)?,
            None => match self.survey.leading().map(|code| code.kind) {
                Some(Kind::Unread) => Outcome::Unread,
                Some(_) => Outcome::Other,
                None => Outcome::NoCode,
            },
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
            self.reached = None;
            self.kept.clear();
        }
    }

    /// Takes in what a QR code of `qqcode-show` says.
    fn take_message(&mut self, message: Message) -> Outcome {
        let file = message.file();
        self.turn_to(Origin {
            checksum: file.checksum,
            length: file.length as u64,
        });
        if self.whole {
            // nothing left to find out about this file
            return Outcome::Complete;
        }
        if !matches!(self.transfer, Some(Transfer::Pieces(_))) {
            self.transfer = Some(Transfer::Pieces(Assembly::new(file)));
        }
        let Some(Transfer::Pieces(assembly)) = &mut self.transfer else {
            unreachable!("just set");
        };
        let outcome = match message {
            Message::Start { name, .. } => {
                self.name = Some(name).filter(|name| !name.is_empty());
                self.reached = Some(0);
                Outcome::Ready
            }
            Message::Piece { offset, bytes, .. } => {
                let offset = offset as usize;
                let skipped = match self.reached {
                    Some(reached) if reached < offset => assembly.missing_in(reached..offset),
                    _ => 0,
                };
                self.reached = Some(offset + bytes.len());
                Outcome::Piece {
                    new: assembly.add(offset, &bytes),
                    skipped,
                }
            }
            Message::End { .. } => Outcome::Done,
        };
        if !assembly.is_complete() {
            return outcome;
        }
        let Some(Transfer::Pieces(assembly)) = self.transfer.take() else {
            unreachable!("it is complete");
        };
        self.finish(assembly.into_bytes())
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
                if let Some((part, sums)) = &self.store {
                    Stored::create(part, sums, 0, origin.checksum, 1)
                        .with_context(|| format!("can't write {}", part.display()))?;
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
            _ => None,
        };
        if symbol != Some(bytes.len()) {
            self.kept.clear();
            self.transfer = Some(match &self.store {
                Some((part, sums)) => {
                    let stored = Stored::create(part, sums, origin.length, origin.checksum, bytes.len())
                        .with_context(|| format!("can't write {}", part.display()))?;
                    Transfer::Stored(match &self.checks {
                        Some(checks) => stored.checked(checks.clone()),
                        None => stored,
                    })
                }
                None => Transfer::Packets(Decoder::new(origin.length, origin.checksum, bytes.len())),
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
            _ => unreachable!("just set"),
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
        let checked = Some(file).filter(|bytes| protocol::checksum(bytes) == origin.checksum);
        Ok(self.finish(checked))
    }

    /// Ends the collecting of the file: with the file, or with nothing if its checksum is wrong or those
    /// of its sections are.
    fn finish(&mut self, file: Option<Vec<u8>>) -> Outcome {
        self.reached = None;
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

    /// What has arrived of a file that is shown in QR codes; `None` before its first code, once it is
    /// whole, and for a file that is shown as a matrix.
    pub fn assembly(&self) -> Option<&Assembly> {
        match &self.transfer {
            Some(Transfer::Pieces(assembly)) => Some(assembly),
            _ => None,
        }
    }

    /// Of a file that is shown as a matrix, how many packets have told something and how many it takes.
    pub fn packets(&self) -> Option<(usize, usize)> {
        match &self.transfer {
            Some(Transfer::Packets(decoder)) => Some((decoder.held(), decoder.plan().symbols())),
            Some(Transfer::Stored(stored)) => Some((stored.held(), stored.plan().symbols())),
            _ => None,
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
            Some(Transfer::Pieces(assembly)) => assembly.received() as u64,
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
