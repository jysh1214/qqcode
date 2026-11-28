//! Finds the matrix of tiles that the page of `qqcode-show` shows in a frame, and reads what it holds.

use std::panic::{self, AssertUnwindSafe};
use std::sync::{Mutex, PoisonError};

use anyhow::{Result, bail, ensure};
use opencv::core::{Mat, Point2f};
use opencv::imgproc;
use opencv::prelude::*;
use rxing::common::HybridBinarizer;
use rxing::qrcode::cpp_port::detector::FindFinderPatterns;
use rxing::{BinaryBitmap, Luma8Source};

use crate::packet::Frame;
use crate::sight::{self, Finder, Lock};

/// The corners of a matrix in a frame: top-left, top-right, bottom-right, bottom-left, as the matrix has
/// them and not as the frame shows them, so that a matrix on its head has its top-left corner at the bottom
/// right. Those of a matrix that could not be read are as the frame shows them.
pub type Outline = [Point2f; 4];

/// The matrix of tiles that is made out in a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub outline: Outline,
    /// The tiles along its longer side; of a matrix that is not read, going by the size of its finder
    /// patterns
    pub modules: usize,
    /// What its frame says, or `None` if it could not be read: too small in the frame, blurred,
    /// outshone, or caught while the page changed
    pub frame: Option<Frame>,
    /// The share of its bytes that were misread and put right
    pub wrong: f32,
}

impl Found {
    /// How many pixels of the frame a tile of the matrix is wide, along its longest side. What is seen
    /// at an angle is smaller on the sides that are further away.
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

/// Looks for the matrix, one frame after the other, from as many threads at a time as there are.
#[derive(Default)]
pub struct Scanner {
    /// The matrix that the frames show, once one of them has been read
    lock: Mutex<Option<Lock>>,
}

impl Scanner {
    /// What a frame of one channel (grey) or three or four (blue, green, red first) shows. The matrix is
    /// told by its colours and read from a frame of three channels only.
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

/// The matrix in a frame, of which `pixels` are the grey values, and what the frames after it are tried
/// by if it was read.
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
    // the matrix has one in each of its four corners
    let finders = FindFinderPatterns(bitmap.get_black_matrix(), true, 0)
        .iter()
        // the detector counts from the corner of a pixel, the frame from its middle
        .map(|pattern| Finder {
            x: pattern.p.x - 0.5,
            y: pattern.p.y - 0.5,
            size: pattern.size as f32,
        })
        .collect::<Vec<_>>();
    let sighted = if frame.channels() == 3 {
        sight::read(frame, &finders, hint)?
    } else {
        None
    };
    let Some(sighted) = sighted else {
        return Ok((Vec::new(), None));
    };
    // Four finder patterns that are no matrix that can be read: a matrix all the same if the frames
    // before showed one, and nothing if not, since four of them can be anything.
    if sighted.lock.is_none() && hint.is_none() {
        return Ok((Vec::new(), None));
    }
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
        modules: if longest > 0 {
            longest
        } else {
            (side / sighted.tile_size).round().max(1.0) as usize
        },
        frame: sighted.frame,
        wrong: sighted.wrong,
    };
    Ok((vec![found], sighted.lock))
}
