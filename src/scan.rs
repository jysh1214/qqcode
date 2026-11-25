//! Finds the codes in a frame and reads what they hold: the QR codes that `qqcode-show` draws on a terminal,
//! and the matrix of tiles of its web page.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, PoisonError};

use anyhow::{Result, bail, ensure};
use opencv::core::{Mat, Point2f};
use opencv::imgproc;
use opencv::prelude::*;
use rxing::common::{DetectorRXingResult, HybridBinarizer};
use rxing::qrcode::cpp_port::decoder::Decode;
use rxing::qrcode::cpp_port::detector::{FindFinderPatterns, GenerateFinderPatternSets, SampleQR};
use rxing::{BinaryBitmap, Luma8Source};

use crate::packet::Frame;
use crate::sight::{self, Finder, Lock};

/// The corners of a code in a frame: top-left, top-right, bottom-right, bottom-left, as the code has them
/// and not as the frame shows them, so that a code on its head has its top-left corner at the bottom right.
pub type Outline = [Point2f; 4];

/// What is made out in a frame: a QR code, or the matrix of tiles of the web page.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub outline: Outline,
    /// Modules along a side of a QR code, going by the distance between its finder patterns; of a
    /// matrix, the tiles along its longer side
    pub modules: usize,
    pub content: Content,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Content {
    /// A QR code with the bytes that it holds, or `None` if it could not be decoded: too small in the
    /// frame, blurred, outshone, or caught while the screen changed
    Code(Option<Vec<u8>>),
    /// A matrix of tiles with what its frame says, or `None` if it could not be read
    Matrix {
        frame: Option<Frame>,
        /// The share of its bytes that were misread and put right
        wrong: f32,
    },
}

impl Found {
    /// How many pixels of the frame a module of the code or a tile of the matrix is wide, along its
    /// longest side. What is seen at an angle is smaller on the sides that are further away.
    pub fn module_size(&self) -> f32 {
        let outline = &self.outline;
        let side = |corner: usize| {
            let (from, to) = (outline[corner], outline[(corner + 1) % 4]);
            (to.x - from.x).hypot(to.y - from.y)
        };
        let longest = (0..4).map(side).fold(0.0, f32::max);
        longest / self.modules as f32
    }
}

/// Looks for codes, one frame after the other, from as many threads at a time as there are.
#[derive(Default)]
pub struct Scanner {
    /// The matrix that the frames show, once one of them has been read
    lock: Mutex<Option<Lock>>,
}

impl Scanner {
    /// What a frame of one channel (grey) or three or four (blue, green, red first) shows.
    pub fn scan(&self, frame: &Mat) -> Result<Vec<Found>> {
        let mut grey = Mat::default();
        match frame.channels() {
            1 => frame.copy_to(&mut grey)?,
            3 => imgproc::cvt_color_def(frame, &mut grey, imgproc::COLOR_BGR2GRAY)?,
            4 => imgproc::cvt_color_def(frame, &mut grey, imgproc::COLOR_BGRA2GRAY)?,
            channels => bail!("a frame is expected in one, three or four channels, not {channels}"),
        }
        let (width, height) = (grey.cols() as usize, grey.rows() as usize);
        ensure!(
            grey.is_continuous() && grey.depth() == opencv::core::CV_8U,
            "a frame is expected in bytes"
        );
        let pixels = grey.data_bytes()?;
        let hint = *self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        // The detector is made for pictures of anything, and a frame it trips over is a frame without a
        // code: nothing that should end a run which has collected most of a file.
        let found = panic::catch_unwind(AssertUnwindSafe(|| codes(pixels, width, height, frame, hint)));
        let (found, lock) = found.unwrap_or_else(|_| Ok((Vec::new(), None)))?;
        if lock.is_some() {
            *self.lock.lock().unwrap_or_else(PoisonError::into_inner) = lock;
        }
        Ok(found)
    }
}

/// The codes in a frame, of which `pixels` are the grey values, and the matrix that was read if one was.
fn codes(
    pixels: &[u8],
    width: usize,
    height: usize,
    frame: &Mat,
    hint: Option<Lock>,
) -> Result<(Vec<Found>, Option<Lock>)> {
    let Ok(source) = Luma8Source::new_with_slice(pixels, width as u32, height as u32) else {
        return Ok((Vec::new(), None));
    };
    let bitmap = BinaryBitmap::new(HybridBinarizer::new(source));
    let image = bitmap.get_black_matrix();
    // every three finder patterns that may belong to one code, the likeliest first
    let mut patterns = FindFinderPatterns(image, true, 0);

    // The matrix of the web page has four of them, and comes first: it is what a frame is full of.
    let finders = patterns
        .iter()
        // the detector counts from the corner of a pixel, the frame from its middle
        .map(|pattern| Finder {
            x: pattern.p.x - 0.5,
            y: pattern.p.y - 0.5,
            size: pattern.size as f32,
        })
        .collect::<Vec<_>>();
    let matrix = if frame.channels() == 3 {
        sight::read(frame, &finders, hint)?
    } else {
        None
    };
    let matrix = matrix.map(|sighted| {
        let longest = sighted
            .lock
            .map_or(0, |lock| lock.grid.columns.max(lock.grid.rows));
        let outline = sighted.outline;
        let side = (0..4)
            .map(|corner| {
                let (from, to) = (outline[corner], outline[(corner + 1) % 4]);
                (to.x - from.x).hypot(to.y - from.y)
            })
            .fold(0.0, f32::max);
        let found = Found {
            outline,
            // of a matrix that is not read, going by the size of its finder patterns
            modules: if longest > 0 {
                longest
            } else {
                (side / sighted.tile_size).round().max(1.0) as usize
            },
            content: Content::Matrix {
                frame: sighted.frame,
                wrong: sighted.wrong,
            },
        };
        (found, sighted.lock)
    });
    if let Some((found, Some(lock))) = matrix {
        return Ok((vec![found], Some(lock)));
    }
    let mut taken = Vec::new();
    let mut read = Vec::new();
    let mut unread = Vec::new();
    for set in GenerateFinderPatternSets(&mut patterns) {
        let corners = [set.tl, set.tr, set.bl];
        if corners.iter().any(|pattern| taken.contains(pattern)) {
            continue;
        }
        let Ok(sampled) = SampleQR(image, &set) else {
            continue;
        };
        // the corners come row by row
        let &[top_left, top_right, bottom_left, bottom_right] = sampled.getPoints() else {
            continue;
        };
        let outline =
            [top_left, top_right, bottom_right, bottom_left].map(|corner| Point2f::new(corner.x, corner.y));
        let modules = sampled.getBits().getWidth() as usize;
        let decoded = Decode(sampled.getBits()).ok().filter(|decoded| decoded.isValid());
        match decoded {
            Some(decoded) => {
                taken.extend(corners);
                read.push(Found {
                    outline,
                    modules,
                    content: Content::Code(Some(decoded.content().bytes().to_vec())),
                });
            }
            None => unread.push((corners, outline, modules)),
        }
    }
    // What can't be decoded is a code as far as its finder patterns belong to no code that can, and each
    // of them to one code only.
    for (corners, outline, modules) in unread {
        if corners.iter().any(|pattern| taken.contains(pattern)) {
            continue;
        }
        taken.extend(corners);
        read.push(Found {
            outline,
            modules,
            content: Content::Code(None),
        });
    }
    // Four finder patterns that are no matrix that can be read: a matrix all the same, if the frames
    // before showed one and there is no QR code that is read instead.
    let decoded = read
        .iter()
        .any(|found| matches!(found.content, Content::Code(Some(_))));
    if let (Some((found, _)), Some(_), false) = (matrix, hint, decoded) {
        return Ok((vec![found], None));
    }
    Ok((read, None))
}
