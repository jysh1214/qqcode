//! Reads the frames of a camera that is pointed at `qqcode-show`, up to the file that is on show there.

use anyhow::Result;
use opencv::core::Mat;

use crate::assembly::Assembly;
use crate::protocol::{FileId, Message};
use crate::scan::{Found, Outline, Scanner};

/// What became of one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// No QR code is made out
    NoCode,
    /// There is a QR code, which can't be decoded
    Unread,
    /// There is a QR code, which is not one of `qqcode-show`
    Other,
    /// The start screen of `qqcode-show`: the pieces of the file are still to come
    Ready,
    /// A piece of the file
    Piece {
        /// How many of its bytes had not arrived before; none for a piece that is read again
        new: usize,
        /// How many bytes between the piece before and this one are missing: those went by unread
        skipped: usize,
    },
    /// The end screen of `qqcode-show`, with parts of the file still missing
    Done,
    /// The file is whole, [`Pipeline::take_file`] has it. Codes of the same file that come after it say
    /// the same.
    Complete,
    /// All of the file has arrived and its checksum is wrong, so a code was misread. What was collected is
    /// dropped, and the pieces count from the beginning.
    Corrupt,
}

/// What a code in a frame is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// It can't be decoded
    Unread,
    /// Not a code of `qqcode-show`
    Other,
    Start,
    Piece,
    End,
}

/// A code in a frame, for showing somebody who points a camera what it sees.
#[derive(Clone, Debug, PartialEq)]
pub struct Sighting {
    pub outline: Outline,
    pub kind: Kind,
    /// Modules along a side of the code
    pub modules: usize,
    /// Pixels of the frame to a module
    pub module_size: f32,
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

#[derive(Default)]
pub struct Pipeline {
    scanner: Scanner,
    /// The file that is on show, going by the code read last
    current: Option<FileId>,
    /// What has arrived of that file
    assembly: Option<Assembly>,
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
    survey: Survey,
}

impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }

    /// Reads one frame, in grey or with blue, green and red.
    pub fn process(&mut self, frame: &Mat) -> Result<Outcome> {
        let found = self.scanner.scan(frame)?;
        let mut codes = Vec::with_capacity(found.len());
        let mut message = None;
        for code in found {
            let parsed = code.bytes.as_deref().map(Message::parse);
            let kind = match &parsed {
                None => Kind::Unread,
                Some(None) => Kind::Other,
                Some(Some(Message::Start { .. })) => Kind::Start,
                Some(Some(Message::Piece { .. })) => Kind::Piece,
                Some(Some(Message::End { .. })) => Kind::End,
            };
            codes.push(sighting(&code, kind));
            // one show at a time: the first of two is the one that counts
            message = message.or(parsed.flatten());
        }
        self.survey = Survey { codes };
        Ok(match message {
            Some(message) => self.take(message),
            None => match self.survey.leading().map(|code| code.kind) {
                Some(Kind::Unread) => Outcome::Unread,
                Some(_) => Outcome::Other,
                None => Outcome::NoCode,
            },
        })
    }

    /// Takes in what a code of `qqcode-show` says.
    fn take(&mut self, message: Message) -> Outcome {
        let file = message.file();
        if self.current != Some(file) {
            // the first code, or one of another file: what was collected so far is of no use for it
            self.current = Some(file);
            self.files += 1;
            self.whole = false;
            self.assembly = None;
            self.name = None;
            self.reached = None;
        }
        if self.whole {
            // nothing left to find out about this file
            return Outcome::Complete;
        }
        let assembly = self.assembly.get_or_insert_with(|| Assembly::new(file));
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
        let assembly = self.assembly.take().expect("it is complete");
        self.reached = None;
        match assembly.into_bytes() {
            Some(bytes) => {
                self.whole = true;
                self.file = Some(bytes);
                Outcome::Complete
            }
            None => Outcome::Corrupt,
        }
    }

    /// The file, once after [`Outcome::Complete`].
    pub fn take_file(&mut self) -> Option<Vec<u8>> {
        self.file.take()
    }

    /// What has arrived of the file that is on show; `None` before its first code, and again once the
    /// file is whole.
    pub fn assembly(&self) -> Option<&Assembly> {
        self.assembly.as_ref()
    }

    /// The file that is on show, going by the code read last.
    pub fn file(&self) -> Option<FileId> {
        self.current
    }

    /// How many bytes of the file that is on show have arrived, if any of its codes has been read.
    pub fn received(&self) -> Option<usize> {
        let file = self.current?;
        Some(match &self.assembly {
            Some(assembly) => assembly.received(),
            None if self.whole => file.length as usize,
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

    /// The codes that were made out in the frame processed last.
    pub fn last_survey(&self) -> &Survey {
        &self.survey
    }
}

fn sighting(code: &Found, kind: Kind) -> Sighting {
    Sighting {
        outline: code.outline,
        kind,
        modules: code.modules,
        module_size: code.module_size(),
    }
}
