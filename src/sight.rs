//! Reads the matrix of tiles ([`crate::tiles`]) out of a frame of the camera.
//!
//! The four finder patterns in the corners of the matrix say where it is. The frame is laid flat by them,
//! at the size the matrix has on the page, and then read tile by tile: how like each of the sixteen symbols
//! a tile looks, and how near its light is to each of the four colours. What comes of that is not a symbol
//! and a colour for the tile but how sure each of its six bits is, which is what the check bits of a frame
//! ([`crate::ldpc`]) put it right from.
//!
//! A lens bends what the four corners can't account for, by a few pixels in the middle of a large matrix.
//! So no tile is read where it ought to be, but where it is found. One tile can't tell: a symbol that is
//! moved by a sub-cell looks much like another symbol. Some dozens of them together can, so the offset is
//! found for areas of eight by eight tiles, starting with those next to the finder patterns and each
//! around where its neighbours turned out to be. What bends a picture bends all of it smoothly, so the
//! areas are then taken together: an area that was found where it is not is put right by the others, and
//! a tile is as far off as the bend of the picture has it where it is.

use anyhow::Result;
use opencv::core::{self, Mat, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use crate::packet::Frame;
use crate::tiles::{self, ANCHOR, Grid, MODULE, PITCH, SCALES, SUB, Shape, Sure, TILE};

/// A finder pattern in a frame: its middle and how many pixels it is across.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Finder {
    pub x: f32,
    pub y: f32,
    pub size: f32,
}

/// Which matrix the frames show and which way up, once one of them has been read: the next ones are tried
/// that way first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lock {
    pub grid: Grid,
    /// How many quarter turns the camera sees the matrix turned by
    turn: usize,
}

/// A matrix that is made out in a frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Sighted {
    /// The corners of the matrix in the frame: top-left, top-right, bottom-right, bottom-left as the
    /// matrix has them if it was read, and as the frame shows them if not
    pub outline: [Point2f; 4],
    /// What the frame says, if its tiles could be read
    pub frame: Option<Frame>,
    pub lock: Option<Lock>,
    /// Tiles along the longer side of the matrix; of a matrix that is not read, going by the size of its
    /// finder patterns
    pub tiles: usize,
    /// The share of the tiles of a frame that were misread and put right
    pub wrong: f32,
}

/// How far a tile may be from where the corners put it, in pixels of the flat picture.
const MAX_DRIFT: i32 = 12;

/// Tiles along a side of the areas for which an offset is found, at the least, and how many pixels either
/// way of where its neighbours are an area is looked for.
const AREA: usize = 8;
const SEARCH: i32 = 2;

/// The four finder patterns that are the corners of a matrix, if `finders` has such: top-left, top-right,
/// bottom-right, bottom-left as the frame shows them.
fn corners(finders: &[Finder]) -> Option<[Finder; 4]> {
    // the patterns of a matrix are all of a size, and larger than what else may look like one
    let largest = finders.iter().map(|finder| finder.size).fold(0.0, f32::max);
    let likely = finders
        .iter()
        .filter(|finder| finder.size >= 0.6 * largest)
        .copied()
        .collect::<Vec<_>>();
    if likely.len() < 4 {
        return None;
    }
    let extreme = |key: fn(&Finder) -> f32| {
        likely
            .iter()
            .copied()
            .min_by(|a, b| key(a).total_cmp(&key(b)))
            .expect("four at least")
    };
    let corners = [
        extreme(|finder| finder.x + finder.y),
        extreme(|finder| finder.y - finder.x),
        extreme(|finder| -finder.x - finder.y),
        extreme(|finder| finder.x - finder.y),
    ];
    let distinct = (0..4).all(|a| (0..a).all(|b| corners[a] != corners[b]));
    distinct.then_some(corners)
}

fn distance(a: &Finder, b: &Finder) -> f32 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// The matrices that the corners may belong to, the likeliest first: going by how far the finder
/// patterns are apart in units of their own size.
fn candidates(corners: &[Finder; 4]) -> Vec<Lock> {
    let [top_left, top_right, bottom_right, bottom_left] = corners;
    let across = (distance(top_left, top_right) + distance(bottom_left, bottom_right)) / 2.0;
    let down = (distance(top_left, bottom_left) + distance(top_right, bottom_right)) / 2.0;
    // a pattern is seven modules across; this is the frame's pixels to a tile
    let pattern = corners.iter().map(|finder| finder.size).sum::<f32>() / 4.0;
    let pitch = pattern / (7 * MODULE) as f32 * PITCH as f32;
    if pitch <= 0.0 {
        return Vec::new();
    }
    let seen = (across / pitch + ANCHOR as f32, down / pitch + ANCHOR as f32);
    let mut ranked = Vec::new();
    for turn in 0..4 {
        // a matrix on its side has its columns down the frame
        let (columns, rows) = if turn % 2 == 0 { seen } else { (seen.1, seen.0) };
        let mut grids = Shape::ALL
            .into_iter()
            .flat_map(|shape| SCALES.map(move |scale| Grid::new(shape, scale)))
            .map(|grid| {
                let off =
                    (grid.columns as f32 - columns).abs() / columns + (grid.rows as f32 - rows).abs() / rows;
                (off, grid)
            })
            .filter(|(off, _)| *off < 0.3)
            .collect::<Vec<_>>();
        grids.sort_by(|a, b| a.0.total_cmp(&b.0));
        // upright is what a camera nearly always is: more guesses for that
        let guesses = if turn == 0 { 3 } else { 1 };
        ranked.extend(
            grids
                .into_iter()
                .take(guesses)
                .map(|(off, grid)| (off + turn as f32, Lock { grid, turn })),
        );
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked.into_iter().map(|(_, lock)| lock).collect()
}

/// Looks for a matrix among the finder patterns of a frame, which has blue, green and red for every
/// pixel, and reads it. `hint` is what earlier frames showed.
pub fn read(frame: &Mat, finders: &[Finder], hint: Option<Lock>) -> Result<Option<Sighted>> {
    let Some(corners) = corners(finders) else {
        return Ok(None);
    };
    let seen = corners.map(|finder| Point2f::new(finder.x, finder.y));
    let mut tries = candidates(&corners);
    if let Some(hint) = hint {
        tries.retain(|lock| *lock != hint);
        tries.insert(0, hint);
    }
    for lock in tries {
        // the corner that the frame shows top-left is the matrix's own top-left after so many turns
        let own = std::array::from_fn(|corner| seen[(corner + lock.turn) % 4]);
        let Some(reading) = attempt(frame, &own, lock.grid)? else {
            continue;
        };
        return Ok(Some(Sighted {
            outline: reading.outline,
            frame: Some(reading.frame),
            lock: Some(lock),
            tiles: lock.grid.columns.max(lock.grid.rows),
            wrong: reading.wrong,
        }));
    }
    // a matrix all the same, as far as its corners say: one that can't be read
    let pattern = corners.iter().map(|finder| finder.size).sum::<f32>() / 4.0;
    let pitch = pattern / (7 * MODULE) as f32 * PITCH as f32;
    let longest = (0..4)
        .map(|corner| distance(&corners[corner], &corners[(corner + 1) % 4]))
        .fold(0.0, f32::max);
    Ok(Some(Sighted {
        outline: seen,
        frame: None,
        lock: None,
        tiles: (longest / pitch).round().max(1.0) as usize,
        wrong: 0.0,
    }))
}

struct Reading {
    frame: Frame,
    outline: [Point2f; 4],
    wrong: f32,
}

/// Reads the frame as a matrix of `grid` whose finder patterns are at `anchors`.
fn attempt(frame: &Mat, anchors: &[Point2f; 4], grid: Grid) -> Result<Option<Reading>> {
    let flat_anchors = grid.anchors().map(|(x, y)| Point2f::new(x, y));
    let seen = Vector::<Point2f>::from_slice(anchors);
    let own = Vector::<Point2f>::from_slice(&flat_anchors);
    let transform = imgproc::get_perspective_transform_def(&seen, &own)?;
    let size = Size::new(grid.width() as i32, grid.height() as i32);
    let mut flat = Mat::default();
    imgproc::warp_perspective(
        frame,
        &mut flat,
        &transform,
        size,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;
    if flat.channels() != 3 || !flat.is_continuous() {
        return Ok(None);
    }
    let sure = read_tiles(flat.data_bytes()?, grid);
    let Some(unpacked) = tiles::unpack(grid, &sure) else {
        return Ok(None);
    };
    let Some(frame) = Frame::parse(&unpacked.payload) else {
        return Ok(None);
    };

    // where the edges of the matrix are in the frame
    let (width, height) = (grid.width() as f32 - 1.0, grid.height() as f32 - 1.0);
    let edges = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)].map(|(x, y)| Point2f::new(x, y));
    let back = imgproc::get_perspective_transform_def(&own, &seen)?;
    let mut outline = Vector::<Point2f>::new();
    core::perspective_transform(&Vector::<Point2f>::from_slice(&edges), &mut outline, &back)?;
    Ok(Some(Reading {
        frame,
        outline: std::array::from_fn(|corner| outline.get(corner).expect("four corners")),
        wrong: unpacked.wrong as f32 / unpacked.tiles as f32,
    }))
}

/// For each of the eight patterns, whether each of the sixteen sub-cells is lit (1) or dark (-1).
fn signs() -> [[i32; 16]; 8] {
    std::array::from_fn(|pattern| {
        std::array::from_fn(|cell| {
            if tiles::lit(pattern as u8, cell / 4, cell % 4) {
                1
            } else {
                -1
            }
        })
    })
}

/// What a tile looks like at one place: how much like each of the eight patterns it is, below zero for a
/// pattern with lit and dark exchanged, and how much like the one it is most like.
#[derive(Clone, Copy, Default)]
struct Match {
    likeness: [i32; 8],
    best: i32,
}

/// How sure a bit may be at most: a tile that is misread for all that it looks right, under a reflection
/// or between two frames, must not outweigh the checks that say so.
const SUREST: f32 = 12.0;

/// The tiles of the flat picture of a matrix, which has blue, green and red for every pixel: how sure each
/// of the six bits is that a cell has, as [`tiles::pack`] has them.
pub fn read_tiles(flat: &[u8], grid: Grid) -> Vec<Sure> {
    let (width, height) = (grid.width(), grid.height());
    assert_eq!(
        flat.len(),
        3 * width * height,
        "a picture of the size of the matrix"
    );
    // how bright a pixel is whatever its colour: all four colours have a channel that is full
    let light = flat
        .chunks_exact(3)
        .map(|pixel| pixel[0].max(pixel[1]).max(pixel[2]))
        .collect::<Vec<_>>();
    let signs = signs();

    let look = |left: i32, top: i32| -> Option<Match> {
        if left < 0 || top < 0 || left as usize + TILE > width || top as usize + TILE > height {
            return None;
        }
        let (left, top) = (left as usize, top as usize);
        let mut cells = [0i32; 16];
        for (cell, sum) in cells.iter_mut().enumerate() {
            let (x, y) = (left + SUB * (cell % 4), top + SUB * (cell / 4));
            for row in 0..SUB {
                let at = (y + row) * width + x;
                *sum += light[at..at + SUB].iter().map(|value| *value as i32).sum::<i32>();
            }
        }
        let mut found = Match::default();
        for (pattern, signs) in signs.iter().enumerate() {
            let likeness = cells
                .iter()
                .zip(signs)
                .map(|(cell, sign)| cell * sign)
                .sum::<i32>();
            found.likeness[pattern] = likeness;
            found.best = found.best.max(likeness.abs());
        }
        Some(found)
    };

    // Where the tiles are found, in areas of some tiles a side: an offset for each, the one at which its
    // tiles are most like symbols taken together and the gaps between them darkest. The gaps are what
    // tells an offset from one that is a sub-cell further, at which the tiles of a blurred picture look
    // like symbols nearly as much: an area that takes such an offset draws the next ones after it, a
    // sub-cell at a time, until they are a whole tile off and read their neighbours' tiles. The areas are
    // gone through outwards from the corners, which are where they ought to be, and each is looked for
    // around where its neighbours were found.
    // The areas are as equal as they come, none smaller than `AREA` tiles a side: a strip of a row or two
    // along an edge has too few tiles to be found by, and goes wherever the area before it went.
    let areas = ((grid.columns / AREA).max(1), (grid.rows / AREA).max(1));
    let area = |column: usize, row: usize| row * areas.0 + column;
    let span = |index: usize, count: usize, tiles: usize| index * tiles / count..(index + 1) * tiles / count;
    let mut order = (0..areas.1)
        .flat_map(|row| (0..areas.0).map(move |column| (column, row)))
        .collect::<Vec<_>>();
    order.sort_by_key(|(column, row)| {
        let across = (*column).min(areas.0 - 1 - column);
        let down = (*row).min(areas.1 - 1 - row);
        across * across + down * down
    });
    // the tiles that an area is looked for by: every other one tells as much as all of them
    let samples = (0..areas.1)
        .flat_map(|row| (0..areas.0).map(move |column| (column, row)))
        .map(|(column, row)| {
            span(row, areas.1, grid.rows)
                .flat_map(|tile_row| {
                    span(column, areas.0, grid.columns).map(move |tile_column| (tile_column, tile_row))
                })
                .filter(|(tile_column, tile_row)| {
                    (tile_column + tile_row) % 2 == 0 && !grid.is_anchor(*tile_column, *tile_row)
                })
                .map(|(tile_column, tile_row)| grid.origin(tile_column, tile_row))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // How like symbols the tiles of a sample are at an offset, which may be below nothing where the gaps
    // are light; `None` for an offset that is further than a tile may be.
    let likeness = |sample: &[(usize, usize)], offset: (i32, i32)| -> Option<i64> {
        if offset.0.abs() > MAX_DRIFT || offset.1.abs() > MAX_DRIFT {
            return None;
        }
        let like = sample
            .iter()
            .filter_map(|(left, top)| {
                let (left, top) = (*left as i32 + offset.0, *top as i32 + offset.1);
                let found = look(left, top)?;
                // the gap to the right of the tile and the one below it, which are dark where the tile is
                let (left, top) = (left as usize, top as usize);
                let mut gap = 0;
                if left + TILE < width && top + TILE < height {
                    let beside = (0..=TILE).map(|down| light[(top + down) * width + left + TILE] as i64);
                    let below = (0..TILE).map(|across| light[(top + TILE) * width + left + across] as i64);
                    gap = beside.chain(below).sum();
                }
                Some(found.best as i64 - gap)
            })
            .sum();
        Some(like)
    };
    // The offset at which the tiles of a sample are most like symbols, of those up to `reach` pixels
    // either way of `around`. Between whole pixels, it is where a parabola through the likeness at the
    // best pixel and at the two next to it has its top. What the sample is like at an offset is kept in
    // `known`, as it is asked for again: by the parabola, and when the area is looked for a second time.
    let find = |sample: &[(usize, usize)],
                known: &mut Vec<((i32, i32), Option<i64>)>,
                around: (i32, i32),
                reach: i32|
     -> (f32, f32) {
        let mut like = |offset: (i32, i32)| -> Option<i64> {
            if let Some((_, like)) = known.iter().find(|(at, _)| *at == offset) {
                return *like;
            }
            let like = likeness(sample, offset);
            known.push((offset, like));
            like
        };
        let mut chosen = (around, like(around));
        for beside in -reach..=reach {
            for below in -reach..=reach {
                let offset = (around.0 + beside, around.1 + below);
                let here = like(offset);
                if here > chosen.1 {
                    chosen = (offset, here);
                }
            }
        }
        let ((x, y), best) = chosen;
        let between = |before: Option<i64>, after: Option<i64>| {
            let (Some(before), Some(best), Some(after)) = (before, best, after) else {
                return 0.0;
            };
            let bend = (before - 2 * best + after) as f32;
            if bend < 0.0 {
                (0.5 * (before - after) as f32 / bend).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        };
        (
            x as f32 + between(like((x - 1, y)), like((x + 1, y))),
            y as f32 + between(like((x, y - 1)), like((x, y + 1))),
        )
    };
    let mut known = vec![Vec::new(); areas.0 * areas.1];
    let mut drift: Vec<Option<(f32, f32)>> = vec![None; areas.0 * areas.1];
    for (column, row) in order {
        let (mut across, mut down, mut count) = (0.0, 0.0, 0);
        for (beside, below) in [
            (-1i32, 0i32),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ] {
            let (near_column, near_row) = (column as i32 + beside, row as i32 + below);
            if near_column < 0 || near_row < 0 || near_column >= areas.0 as i32 || near_row >= areas.1 as i32
            {
                continue;
            }
            if let Some((x, y)) = drift[area(near_column as usize, near_row as usize)] {
                across += x;
                down += y;
                count += 1;
            }
        }
        let expected = if count > 0 {
            (
                (across / count as f32).round() as i32,
                (down / count as f32).round() as i32,
            )
        } else {
            (0, 0)
        };
        let index = area(column, row);
        drift[index] = Some(find(&samples[index], &mut known[index], expected, SEARCH));
    }

    // What bends a picture bends all of it, a lens or a screen that is not flat, and smoothly. So the areas
    // taken together say more of where one of them is than it says itself: an area whose tiles look as
    // much like symbols a sub-cell off, which a blurred one often does, is put right by the others, and
    // so are the areas that followed it there. Each area is then looked for once more, next to where the
    // others put it, and the bend is taken from the areas again.
    let middle_of = |left: f64, top: f64| {
        (
            2.0 * (left + TILE as f64 / 2.0) / width as f64 - 1.0,
            2.0 * (top + TILE as f64 / 2.0) / height as f64 - 1.0,
        )
    };
    let spots = samples
        .iter()
        .map(|sample| {
            let count = sample.len().max(1) as f64;
            let left = sample.iter().map(|(left, _)| *left as f64).sum::<f64>() / count;
            let top = sample.iter().map(|(_, top)| *top as f64).sum::<f64>() / count;
            middle_of(left, top)
        })
        .collect::<Vec<_>>();
    let weights = samples
        .iter()
        .map(|sample| sample.len() as f64)
        .collect::<Vec<_>>();
    let mut found = drift
        .iter()
        .map(|drift| drift.unwrap_or((0.0, 0.0)))
        .collect::<Vec<_>>();
    let mut bend = Bend::fit(&spots, &found, &weights);
    for (index, sample) in samples.iter().enumerate() {
        let (x, y) = bend.at(spots[index]);
        found[index] = find(sample, &mut known[index], (x.round() as i32, y.round() as i32), 1);
    }
    bend = Bend::fit(&spots, &found, &weights);
    // What an area is off by beyond the bend, evened out with the areas around it: what bends the picture
    // in a way that the bend has no words for. An area that is far off is taken to be misread.
    let rest = found
        .iter()
        .zip(&spots)
        .zip(&weights)
        .map(|((found, spot), weight)| {
            let (x, y) = bend.at(*spot);
            let rest = (found.0 - x as f32, found.1 - y as f32);
            (*weight > 0.0 && rest.0.abs() <= ASTRAY && rest.1.abs() <= ASTRAY).then_some(rest)
        })
        .collect::<Vec<_>>();
    let evened = (0..areas.0 * areas.1)
        .map(|index| {
            let (column, row) = ((index % areas.0) as i32, (index / areas.0) as i32);
            let (mut across, mut down, mut count) = (0.0, 0.0, 0);
            for near_row in (row - 1).max(0)..=(row + 1).min(areas.1 as i32 - 1) {
                for near_column in (column - 1).max(0)..=(column + 1).min(areas.0 as i32 - 1) {
                    if let Some((x, y)) = rest[area(near_column as usize, near_row as usize)] {
                        across += x;
                        down += y;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                (across / count as f32, down / count as f32)
            } else {
                (0.0, 0.0)
            }
        })
        .collect::<Vec<_>>();
    // a tile is as far off as the bend has it where it is, and as the areas around it are beyond that, by
    // how near it is to the middle of each
    let offset_of = |column: usize, row: usize| -> (i32, i32) {
        let place = |tile: usize, count: usize, tiles: usize| {
            let at = ((tile as f32 + 0.5) * count as f32 / tiles as f32 - 0.5).clamp(0.0, count as f32 - 1.0);
            let first = (at.floor() as usize).min(count.saturating_sub(2));
            (first, (first + 1).min(count - 1), at - first as f32)
        };
        let (left, right, across) = place(column, areas.0, grid.columns);
        let (top, bottom, down) = place(row, areas.1, grid.rows);
        let at = |column: usize, row: usize| evened[area(column, row)];
        let blend = |pick: fn((f32, f32)) -> f32| {
            let upper = pick(at(left, top)) * (1.0 - across) + pick(at(right, top)) * across;
            let lower = pick(at(left, bottom)) * (1.0 - across) + pick(at(right, bottom)) * across;
            upper * (1.0 - down) + lower * down
        };
        let (own_left, own_top) = grid.origin(column, row);
        let (x, y) = bend.at(middle_of(own_left as f64, own_top as f64));
        (
            (x as f32 + blend(|offset| offset.0)).round() as i32,
            (y as f32 + blend(|offset| offset.1)).round() as i32,
        )
    };

    let cell = |column: usize, row: usize| row * grid.columns + column;
    let mut offsets = vec![(0i32, 0i32); grid.columns * grid.rows];
    let mut seen = Vec::with_capacity(grid.tiles());
    for (column, row) in grid.cells() {
        let offset = offset_of(column, row);
        let (left, top) = grid.origin(column, row);
        offsets[cell(column, row)] = offset;
        seen.push(look(left as i32 + offset.0, top as i32 + offset.1).unwrap_or_default());
    }
    // A blurred picture keeps more of a pattern of two halves than of one with a lit middle, so a tile is
    // not simply the pattern it is most like: each likeness is weighed by how strong that pattern comes
    // out in this picture, going by the tiles that are taken for it at first.
    let mut strength = [1f32; 8];
    for _ in 0..2 {
        let (mut sums, mut counts) = ([0f32; 8], [0usize; 8]);
        for found in &seen {
            let pattern = pattern_of(&found.likeness, &strength);
            sums[pattern] += found.likeness[pattern].abs() as f32;
            counts[pattern] += 1;
        }
        for pattern in 0..8 {
            // a pattern that hardly any tile has keeps its weight
            if counts[pattern] >= 16 && sums[pattern] > 0.0 {
                strength[pattern] = sums[pattern] / counts[pattern] as f32;
            }
        }
    }
    // How like each of the eight patterns a tile is, with a like tile at about 1, or at -1 with lit and dark
    // exchanged. The patterns that a tile is not are then as far from 0 as the picture is off, and how far
    // that is in this picture says what a likeness is worth: a tile that is 0.2 nearer to one pattern than
    // to the next is sure of it in a sharp picture and not in a blurred one.
    let scores = seen
        .iter()
        .map(|found| {
            std::array::from_fn::<f32, 8, _>(|pattern| found.likeness[pattern] as f32 / strength[pattern])
        })
        .collect::<Vec<_>>();
    let (mut off, mut count) = (0f64, 0usize);
    for score in &scores {
        let taken = pattern_of_scores(score);
        for (pattern, value) in score.iter().enumerate() {
            if pattern != taken {
                off += (*value as f64) * (*value as f64);
                count += 1;
            }
        }
    }
    let symbol_weight = 1.0 / (off / count.max(1) as f64).max(1e-6) as f32;

    // The colour of each tile: all the light there is in it, with its brightest channel as 1. The dark half
    // of a tile adds nothing to that on the page, and in a frame it has what a camera lets the colour of
    // the lit half run into: one that packs its frames keeps colour at half the sharpness of light.
    let mut tints = Vec::with_capacity(grid.tiles());
    for (column, row) in grid.cells() {
        let (left, top) = grid.origin(column, row);
        let offset = offsets[cell(column, row)];
        let (left, top) = (left as i32 + offset.0, top as i32 + offset.1);
        let mut light = [0f32; 3];
        if left >= 0 && top >= 0 && left as usize + TILE <= width && top as usize + TILE <= height {
            for y in 0..TILE {
                let row = &flat[3 * ((top as usize + y) * width + left as usize)..][..3 * TILE];
                for pixel in row.chunks_exact(3) {
                    for channel in 0..3 {
                        light[channel] += pixel[channel] as f32;
                    }
                }
            }
        }
        let brightest = light.iter().copied().fold(1.0, f32::max);
        tints.push(light.map(|channel| channel / brightest));
    }
    // How far the light of a tile is from each of the four colours as this picture has them, and what
    // that is worth, which again is by how far the tiles are from the colours that they are.
    let centres = centres(&tints);
    let apart = tints
        .iter()
        .map(|tint| centres.map(|centre| distance_of(&centre, tint)))
        .collect::<Vec<_>>();
    let nearest = apart
        .iter()
        .map(|apart| apart.iter().copied().fold(f32::MAX, f32::min) as f64)
        .sum::<f64>();
    let colour_weight = 1.0 / (nearest / apart.len().max(1) as f64).max(1e-6) as f32;

    scores
        .iter()
        .zip(&apart)
        .map(|(score, apart)| {
            let symbol = symbol_bits(score).map(|bit| symbol_weight * bit);
            let colour = [
                apart[2].min(apart[3]) - apart[0].min(apart[1]),
                apart[1].min(apart[3]) - apart[0].min(apart[2]),
            ]
            .map(|bit| colour_weight * bit);
            let bits = [symbol[0], symbol[1], symbol[2], symbol[3], colour[0], colour[1]];
            bits.map(|bit| bit.clamp(-SUREST, SUREST))
        })
        .collect()
}

/// How far an area may be from where the bend of the picture has it, in pixels, to count as found.
const ASTRAY: f32 = 1.0;

/// How far the tiles of a picture are from where its corners put them, as something that changes smoothly
/// from one side of the picture to the other: a sum of powers of how far across a place is, of how far
/// down, and of the two multiplied, up to `degree` of them taken together. A lens bends a picture by the
/// third power of how far a place is from the middle.
struct Bend {
    degree: usize,
    across: Vec<f64>,
    down: Vec<f64>,
}

impl Bend {
    /// The powers of a place, which is from -1 to 1 across and down the picture.
    fn powers(degree: usize, (x, y): (f64, f64)) -> Vec<f64> {
        let mut powers = Vec::new();
        for together in 0..=degree {
            for of_down in 0..=together {
                powers.push(x.powi((together - of_down) as i32) * y.powi(of_down as i32));
            }
        }
        powers
    }

    /// The bend that the offsets `found` at `spots` are nearest to, each counted by its `weight`, with as
    /// many powers as there are spots to tell them by. A spot that is far from where the others have it
    /// counts for less, the further the far less: it is taken to be misread, and the bend is of the others,
    /// which takes some turns to come out.
    fn fit(spots: &[(f64, f64)], found: &[(f32, f32)], weights: &[f64]) -> Self {
        let degree = match weights.iter().filter(|weight| **weight > 0.0).count() {
            0..6 => 0,
            6..15 => 1,
            15..30 => 2,
            30..60 => 3,
            _ => 4,
        };
        let terms = (degree + 1) * (degree + 2) / 2;
        let mut bend = Self {
            degree,
            across: vec![0.0; terms],
            down: vec![0.0; terms],
        };
        let mut trust = weights.to_vec();
        for _ in 0..6 {
            // the squares of what is left are least where these sums come out even
            let mut sums = vec![vec![0.0; terms + 2]; terms];
            for ((spot, found), trust) in spots.iter().zip(found).zip(&trust) {
                let powers = Self::powers(degree, *spot);
                for (row, power) in sums.iter_mut().zip(&powers) {
                    for (sum, other) in row.iter_mut().zip(&powers) {
                        *sum += trust * power * other;
                    }
                    row[terms] += trust * power * found.0 as f64;
                    row[terms + 1] += trust * power * found.1 as f64;
                }
            }
            for term in 0..terms {
                // a power that no spot tells apart from the others is left at nothing
                sums[term][term] += 1e-9;
                let pivot = (term..terms)
                    .max_by(|a, b| sums[*a][term].abs().total_cmp(&sums[*b][term].abs()))
                    .expect("a row");
                sums.swap(term, pivot);
                let row = sums[term].clone();
                for (other, sum) in sums.iter_mut().enumerate() {
                    if other != term {
                        let times = sum[term] / row[term];
                        for (value, from) in sum.iter_mut().zip(&row) {
                            *value -= times * from;
                        }
                    }
                }
            }
            for (term, row) in sums.iter().enumerate() {
                bend.across[term] = row[terms] / row[term];
                bend.down[term] = row[terms + 1] / row[term];
            }
            for (((spot, found), weight), trust) in spots.iter().zip(found).zip(weights).zip(&mut trust) {
                let (x, y) = bend.at(*spot);
                let off = (found.0 as f64 - x).powi(2) + (found.1 as f64 - y).powi(2);
                *trust = weight / (1.0 + off / (ASTRAY as f64 * ASTRAY as f64)).powi(2);
            }
        }
        bend
    }

    /// How far off a place is, across and down.
    fn at(&self, spot: (f64, f64)) -> (f64, f64) {
        let powers = Self::powers(self.degree, spot);
        let sum = |of: &[f64]| of.iter().zip(&powers).map(|(a, b)| a * b).sum::<f64>();
        (sum(&self.across), sum(&self.down))
    }
}

/// The pattern that a tile is most like, going by its likenesses in units of their strengths.
fn pattern_of_scores(score: &[f32; 8]) -> usize {
    (0..8)
        .max_by(|a, b| score[*a].abs().total_cmp(&score[*b].abs()))
        .expect("eight patterns")
}

/// What the likenesses of a tile say of the four bits of its symbol, the first of which is whether lit and
/// dark are exchanged and the others the number of the pattern: for each bit, how much more like the tile
/// is to the likeliest symbol that has a 0 there than to the likeliest that has a 1.
fn symbol_bits(score: &[f32; 8]) -> [f32; 4] {
    let most = score.iter().copied().fold(f32::MIN, f32::max);
    let least = score.iter().copied().fold(f32::MAX, f32::min);
    let mut bits = [most + least, 0.0, 0.0, 0.0];
    for bit in 0..3 {
        let side = |one: bool| {
            (0..8)
                .filter(|pattern| (pattern >> (2 - bit) & 1 == 1) == one)
                .map(|pattern| score[pattern].abs())
                .fold(0.0, f32::max)
        };
        bits[1 + bit] = side(false) - side(true);
    }
    bits
}

/// The pattern that a tile is most like, each likeness taken by the strength of its pattern.
fn pattern_of(likeness: &[i32; 8], strength: &[f32; 8]) -> usize {
    let weighed = |pattern: usize| likeness[pattern].abs() as f32 / strength[pattern];
    (0..8)
        .max_by(|a, b| weighed(*a).total_cmp(&weighed(*b)))
        .expect("eight patterns")
}

/// How much less of a channel the tiles of one colour have than the others, at the least, for a frame to
/// be taken at its word about where its colours are.
const APART: f32 = 0.2;

/// How far a tint is from a colour, squared.
fn distance_of(centre: &[f32; 3], tint: &[f32; 3]) -> f32 {
    centre.iter().zip(tint).map(|(a, b)| (a - b) * (a - b)).sum()
}

/// Which of the four colours each tint is nearest to, as [`centres`] has them.
#[cfg(test)]
fn classify(tints: &[[f32; 3]]) -> Vec<u8> {
    let centres = centres(tints);
    tints
        .iter()
        .map(|tint| {
            (0..4u8)
                .min_by(|a, b| {
                    distance_of(&centres[*a as usize], tint)
                        .total_cmp(&distance_of(&centres[*b as usize], tint))
                })
                .expect("four colours")
        })
        .collect()
}

/// Where the four colours are among these tints. The colours are those of the page as a camera sees them,
/// which is not as they are on the page: one camera has two thirds of the red in a cyan tile that it has
/// in the white tile next to it. So the four begin where this frame has them, as far as it shows that: a
/// channel that the tiles have in two [`groups`] says how much of it white has, and how much the colour
/// that lacks it. Where a channel says nothing, they begin as they ought to be. They are then moved, a few
/// times over, to the middle of the tints that are nearest to each.
fn centres(tints: &[[f32; 3]]) -> [[f32; 3]; 4] {
    // blue, green, red, as the frame has them
    let pure: [[f32; 3]; 4] =
        tiles::COLOURS.map(|[red, green, blue]| [blue, green, red].map(|c| c as f32 / 255.0));
    let nearest = |centres: &[[f32; 3]; 4], tint: &[f32; 3]| {
        (0..4u8)
            .min_by(|a, b| {
                distance_of(&centres[*a as usize], tint).total_cmp(&distance_of(&centres[*b as usize], tint))
            })
            .expect("four colours")
    };
    let mut begin = pure;
    for channel in 0..3 {
        if let Some((faint, full)) = groups(tints.iter().map(|tint| tint[channel]).collect()) {
            for colour in 0..4 {
                begin[colour][channel] = if pure[colour][channel] == 0.0 { faint } else { full };
            }
        }
    }
    let mut centres = begin;
    for _ in 0..4 {
        let mut sums = [[0f32; 3]; 4];
        let mut counts = [0usize; 4];
        for tint in tints {
            let colour = nearest(&centres, tint) as usize;
            counts[colour] += 1;
            for channel in 0..3 {
                sums[colour][channel] += tint[channel];
            }
        }
        for colour in 0..4 {
            // a colour that hardly any tile has stays where it was
            if counts[colour] >= 16 {
                let moved = sums[colour].map(|sum| sum / counts[colour] as f32);
                // and none goes over to where another one belongs
                if distance_of(&moved, &begin[colour]) < 0.6 * 0.6 {
                    centres[colour] = moved;
                }
            }
        }
    }
    centres
}

/// The two groups that the tiles are in by how much they have of one channel, if they are in two: the
/// middle of those that have little of it and of those that have all of it, at least [`APART`] from each
/// other. The groups are those on either side of the middle between the ends, less a twentieth of the
/// tiles at each end, which may be anything; a colour that hardly any tile has is no group.
fn groups(mut values: Vec<f32>) -> Option<(f32, f32)> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f32::total_cmp);
    let at = |share: f32| values[((values.len() - 1) as f32 * share).round() as usize];
    let between = (at(0.05) + at(0.95)) / 2.0;
    let (faint, full) = values.split_at(values.partition_point(|value| *value < between));
    if faint.len() < 16 || full.len() < 16 {
        return None;
    }
    let middle = |of: &[f32]| of.iter().sum::<f32>() / of.len() as f32;
    let (faint, full) = (middle(faint), middle(full));
    (full - faint >= APART).then_some((faint, full))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finder(x: f32, y: f32) -> Finder {
        Finder { x, y, size: 42.0 }
    }

    /// The picture of a frame as the page shows it, in blue, green, red.
    fn picture(grid: Grid, values: &[u8]) -> Vec<u8> {
        let mut picture = tiles::render(grid, values);
        for pixel in picture.chunks_exact_mut(3) {
            pixel.swap(0, 2);
        }
        picture
    }

    fn values(grid: Grid, seed: u8) -> Vec<u8> {
        let payload = (0..grid.capacity())
            .map(|at| (at as u8).wrapping_mul(31).wrapping_add(seed))
            .collect::<Vec<_>>();
        tiles::pack(grid, &payload)
    }

    #[test]
    fn corners_are_the_four_patterns_furthest_out() {
        let four = [
            finder(900.0, 100.0),
            finder(100.0, 110.0),
            finder(110.0, 600.0),
            finder(910.0, 590.0),
        ];
        let expected = [four[1], four[0], four[3], four[2]];
        assert_eq!(corners(&four), Some(expected));
        // what looks like a pattern among the tiles is smaller and further in
        let mut more = four.to_vec();
        more.push(Finder {
            x: 500.0,
            y: 300.0,
            size: 12.0,
        });
        more.push(finder(480.0, 320.0));
        assert_eq!(corners(&more), Some(expected));
        assert_eq!(corners(&four[..3]), None);
    }

    #[test]
    fn matrix_is_guessed_from_its_corners() {
        // 176 by 99 tiles of 9 pixels at their own size: the patterns are 170 and 93 cells apart
        let seen = [
            finder(30.0, 30.0),
            finder(1560.0, 30.0),
            finder(1560.0, 867.0),
            finder(30.0, 867.0),
        ];
        let guesses = candidates(&seen);
        assert_eq!(
            guesses[0],
            Lock {
                grid: Grid::new(Shape::Wide, 11),
                turn: 0
            }
        );
        // the next sizes and the other ways up come after
        assert!(
            guesses.len() >= 4 && guesses[1].turn == 0 && guesses[1].grid.rows != 99,
            "{guesses:?}"
        );
        assert!(
            guesses
                .iter()
                .any(|lock| lock.turn == 2 && lock.grid == Grid::new(Shape::Wide, 11))
        );
        // at two thirds of the size, with patterns that are measured a little too large
        let far = seen.map(|finder| Finder {
            x: finder.x * 0.66,
            y: finder.y * 0.66,
            size: 29.0,
        });
        assert_eq!(candidates(&far)[0].grid, Grid::new(Shape::Wide, 11));
        // four by three
        let classic = Grid::new(Shape::Classic, 8);
        let [a, _, c, _] = classic.anchors();
        let seen = [
            finder(a.0, a.1),
            finder(c.0, a.1),
            finder(c.0, c.1),
            finder(a.0, c.1),
        ];
        assert_eq!(candidates(&seen)[0].grid, classic);
    }

    #[test]
    fn tiles_are_read_from_the_picture_of_the_page() {
        for grid in [Grid::new(Shape::Square, 4), Grid::new(Shape::Wide, 11)] {
            let values = values(grid, 3);
            assert_eq!(
                tiles::likeliest(&read_tiles(&picture(grid, &values), grid)),
                values
            );
        }
    }

    #[test]
    fn tiles_are_read_where_they_are_found() -> Result<()> {
        // What a lens does: the picture is where the corners put it at its edges, and bulges towards its
        // middle, by up to 6 pixels across and 4 down.
        let grid = Grid::new(Shape::Wide, 8);
        let values = values(grid, 5);
        let straight = picture(grid, &values);
        let (width, height) = (grid.width() as i32, grid.height() as i32);
        let mut flat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, Scalar::all(0.0))?;
        flat.data_bytes_mut()?.copy_from_slice(&straight);
        let mut from_x = Mat::new_rows_cols_with_default(height, width, core::CV_32FC1, Scalar::all(0.0))?;
        let mut from_y = from_x.clone();
        let bulge = |at: i32, of: i32, most: f32| most * (std::f32::consts::PI * at as f32 / of as f32).sin();
        for y in 0..height {
            for x in 0..width {
                *from_x.at_2d_mut::<f32>(y, x)? = x as f32 + bulge(y, height, 6.0);
                *from_y.at_2d_mut::<f32>(y, x)? = y as f32 + bulge(x, width, 4.0);
            }
        }
        let mut bent = Mat::default();
        imgproc::remap(
            &flat,
            &mut bent,
            &from_x,
            &from_y,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        let sure = read_tiles(bent.data_bytes()?, grid);
        let read = tiles::likeliest(&sure);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        // a tile here and there, which the check bits make up for
        assert!(wrong * 100 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &sure).is_some());
        Ok(())
    }

    #[test]
    fn colours_are_told_apart_as_a_camera_sees_them() {
        // washed out and tinted: what should be nothing of a channel is 40 %, and blue is weak
        let seen = |colour: usize| {
            let [red, green, blue] = tiles::COLOURS[colour].map(|channel| 0.4 + 0.6 * channel as f32 / 255.0);
            [0.7 * blue, green, red]
        };
        let tints = (0..400).map(|tile| seen(tile % 4)).collect::<Vec<_>>();
        let colours = classify(&tints);
        assert!(
            colours
                .iter()
                .enumerate()
                .all(|(tile, colour)| *colour as usize == tile % 4)
        );
        // a frame that happens to have one colour only is read as that
        assert_eq!(classify(&vec![[1.0, 1.0, 0.1]; 50]), vec![1; 50]);
    }

    #[test]
    fn tiles_are_followed_from_the_corners_to_where_they_are_a_whole_tile_off() -> Result<()> {
        // A picture that sags: where the corners put it at its left and right edges, and up to 9 pixels
        // lower towards the middle, which is a whole tile. Only from the corners inwards, an area after
        // the one before, can the tiles be followed that far; looked for where they ought to be, the
        // tiles in the middle are those of the row above.
        let grid = Grid::new(Shape::Wide, 8);
        let values = values(grid, 6);
        let straight = picture(grid, &values);
        let (width, height) = (grid.width() as i32, grid.height() as i32);
        let mut flat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, Scalar::all(0.0))?;
        flat.data_bytes_mut()?.copy_from_slice(&straight);
        let mut from_x = Mat::new_rows_cols_with_default(height, width, core::CV_32FC1, Scalar::all(0.0))?;
        let mut from_y = from_x.clone();
        let bulge = |at: i32, of: i32, most: f32| most * (std::f32::consts::PI * at as f32 / of as f32).sin();
        for y in 0..height {
            for x in 0..width {
                *from_x.at_2d_mut::<f32>(y, x)? = x as f32 + bulge(y, height, 2.0);
                *from_y.at_2d_mut::<f32>(y, x)? = y as f32 + bulge(x, width, 9.0);
            }
        }
        let mut bent = Mat::default();
        imgproc::remap(
            &flat,
            &mut bent,
            &from_x,
            &from_y,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        let sure = read_tiles(bent.data_bytes()?, grid);
        let read = tiles::likeliest(&sure);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        assert!(wrong * 50 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &sure).is_some());
        Ok(())
    }

    #[test]
    fn tiles_are_read_where_they_are_found_to_the_other_side_as_well() -> Result<()> {
        // the bulge of a lens the other way round: towards the middle of the picture the tiles are up to
        // 6 pixels further to the right than the corners put them, and up to 4 further down
        let grid = Grid::new(Shape::Wide, 8);
        let values = values(grid, 6);
        let straight = picture(grid, &values);
        let (width, height) = (grid.width() as i32, grid.height() as i32);
        let mut flat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, Scalar::all(0.0))?;
        flat.data_bytes_mut()?.copy_from_slice(&straight);
        let mut from_x = Mat::new_rows_cols_with_default(height, width, core::CV_32FC1, Scalar::all(0.0))?;
        let mut from_y = from_x.clone();
        let bulge = |at: i32, of: i32, most: f32| most * (std::f32::consts::PI * at as f32 / of as f32).sin();
        for y in 0..height {
            for x in 0..width {
                *from_x.at_2d_mut::<f32>(y, x)? = x as f32 - bulge(y, height, 6.0);
                *from_y.at_2d_mut::<f32>(y, x)? = y as f32 - bulge(x, width, 4.0);
            }
        }
        let mut bent = Mat::default();
        imgproc::remap(
            &flat,
            &mut bent,
            &from_x,
            &from_y,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(0.0),
        )?;
        let sure = read_tiles(bent.data_bytes()?, grid);
        let read = tiles::likeliest(&sure);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        assert!(wrong * 100 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &sure).is_some());
        Ok(())
    }

    #[test]
    fn colours_are_told_apart_where_the_camera_has_them() {
        // A camera that lets blue through where the page has none: the yellow tiles have between a
        // third of it and somewhat more than half. For three in ten of them that is nearer to white
        // as the page has it than to yellow, and nearer to yellow as this camera sees it.
        let tints = (0..400)
            .map(|tile| match tile % 4 {
                0 => [0.95, 1.0, 1.0],
                1 => [1.0, 1.0, 0.05],
                2 => [1.0, 0.05, 1.0],
                _ => [0.36 + 0.002 * (tile / 4) as f32, 1.0, 1.0],
            })
            .collect::<Vec<_>>();
        let colours = classify(&tints);
        let wrong = (0..400)
            .filter(|tile| colours[*tile] as usize != tile % 4)
            .count();
        assert_eq!(wrong, 0);
    }

    #[test]
    fn colours_are_told_apart_that_are_all_nearer_to_white_than_to_what_they_are() {
        // As a webcam had them off a laptop's screen: two thirds of the red in a cyan tile that a white
        // one has, and half of what magenta and yellow lack. Each has its tiles around that, cyan from
        // 0.54 to 0.78 of red, which is nearer to white as the page has it for every one of them.
        let seen = |tile: usize| {
            let around = 0.12 * ((tile / 4 % 25) as f32 / 24.0 - 0.5);
            match tile % 4 {
                0 => [0.94 + around, 0.95, 0.97 - around.abs()],
                1 => [0.96, 0.97, 0.66 + 2.0 * around],
                2 => [0.97, 0.53 + 2.0 * around, 0.94],
                _ => [0.53 + 2.0 * around, 0.94, 0.99],
            }
        };
        let tints = (0..2000).map(seen).collect::<Vec<_>>();
        let colours = classify(&tints);
        let wrong = (0..2000)
            .filter(|tile| colours[*tile] as usize != tile % 4)
            .count();
        assert_eq!(wrong, 0);
        // with a colour that the frame has no tile of, the others are found all the same
        let three = (0..2000)
            .filter(|tile| tile % 4 != 3)
            .map(|tile| (tile % 4, seen(tile)))
            .collect::<Vec<_>>();
        let colours = classify(&three.iter().map(|(_, tint)| *tint).collect::<Vec<_>>());
        assert!(
            three
                .iter()
                .zip(&colours)
                .all(|((is, _), read)| *is == *read as usize)
        );
    }

    #[test]
    fn channel_is_in_two_groups_only_if_they_are_apart() {
        let spread = |middle: f32, wide: f32, count: usize| {
            (0..count).map(move |at| middle + wide * (at as f32 / (count - 1) as f32 - 0.5))
        };
        // a quarter of the tiles lack a third of the channel
        let two = spread(0.66, 0.2, 100)
            .chain(spread(0.97, 0.1, 300))
            .collect::<Vec<_>>();
        let (faint, full) = groups(two).expect("two groups");
        assert!(
            (faint - 0.66).abs() < 0.01 && (full - 0.97).abs() < 0.01,
            "{faint} {full}"
        );
        // tiles that all have the channel, some more of it and some less, are one group
        assert_eq!(groups(spread(0.9, 0.3, 400).collect()), None);
        assert_eq!(groups(vec![0.1; 50]), None);
        assert_eq!(groups(Vec::new()), None);
        // and a dozen tiles of garbage are none
        let few = spread(0.1, 0.1, 12)
            .chain(spread(0.97, 0.1, 388))
            .collect::<Vec<_>>();
        assert_eq!(groups(few), None);
    }

    #[test]
    fn small_pattern_further_out_than_the_matrix_is_no_corner() {
        let four = [
            finder(900.0, 100.0),
            finder(100.0, 110.0),
            finder(110.0, 600.0),
            finder(910.0, 590.0),
        ];
        let expected = [four[1], four[0], four[3], four[2]];
        // something in the corner of the frame that looks like a finder pattern, half the size of those
        // of the matrix: it lies further out than any of them, and is none of the matrix all the same
        let mut more = four.to_vec();
        more.push(Finder {
            x: 20.0,
            y: 15.0,
            size: 21.0,
        });
        assert_eq!(corners(&more), Some(expected));
        // one of nearly their size is taken for a corner, as it has to be: a matrix seen at an angle
        // has patterns of different sizes
        more[4].size = 30.0;
        assert_eq!(corners(&more).map(|corners| corners[0]), Some(more[4]));
    }

    #[test]
    fn finder_patterns_in_a_row_are_no_matrix() -> Result<()> {
        // four of them and no four corners: the two at the ends are each the nearest to two corners of
        // the frame
        let row = [100.0, 300.0, 500.0, 700.0].map(|x| finder(x, 200.0));
        assert_eq!(corners(&row), None);
        let frame = Mat::new_rows_cols_with_default(400, 800, core::CV_8UC3, Scalar::all(0.0))?;
        assert_eq!(read(&frame, &row, None)?, None);
        Ok(())
    }

    #[test]
    fn colour_that_hardly_any_tile_has_is_no_group() {
        // Ten tiles in a hundred lack the channel: more than the twentieth at either end that may be
        // anything, and fewer than a group is taken to have. The same of ten that have it.
        assert_eq!(groups([vec![0.2; 10], vec![0.97; 90]].concat()), None);
        assert_eq!(groups([vec![0.2; 90], vec![0.97; 10]].concat()), None);
        // sixteen are a group
        let (faint, full) = groups([vec![0.2; 16], vec![0.97; 84]].concat()).expect("two groups");
        assert!(
            (faint - 0.2).abs() < 0.001 && (full - 0.97).abs() < 0.001,
            "{faint} {full}"
        );
    }

    /// A picture as the page shows it, out of focus by so many pixels.
    fn blurred(grid: Grid, values: &[u8], sigma: f64) -> Result<Mat> {
        let (width, height) = (grid.width() as i32, grid.height() as i32);
        let mut flat = Mat::new_rows_cols_with_default(height, width, core::CV_8UC3, Scalar::all(0.0))?;
        flat.data_bytes_mut()?.copy_from_slice(&picture(grid, values));
        let mut blurred = Mat::default();
        imgproc::gaussian_blur_def(&flat, &mut blurred, Size::new(0, 0), sigma)?;
        Ok(blurred)
    }

    #[test]
    fn tiles_of_a_blurred_picture_are_read_where_they_are() -> Result<()> {
        // A picture that is where its corners put it, and out of focus. Its tiles look like symbols a
        // sub-cell further nearly as much as where they are, and an area that takes them there draws the
        // next ones after it until they are a whole tile off. The matrices are ones whose tiles do not
        // come out in areas of eight: the strip that is left over is no area of its own, in which a
        // handful of tiles would decide where it is.
        for (shape, scale) in [(Shape::Wide, 9), (Shape::Classic, 7)] {
            let grid = Grid::new(shape, scale);
            assert!(!grid.rows.is_multiple_of(AREA) || !grid.columns.is_multiple_of(AREA));
            for seed in 0..3 {
                let values = values(grid, seed);
                let read = tiles::likeliest(&read_tiles(blurred(grid, &values, 1.6)?.data_bytes()?, grid));
                let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
                assert!(wrong * 1000 < values.len(), "{shape:?}, picture {seed}: {wrong}");
                // more out of focus, some tiles are misread, and none because it is looked for elsewhere
                let read = tiles::likeliest(&read_tiles(blurred(grid, &values, 2.0)?.data_bytes()?, grid));
                let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
                assert!(
                    wrong * 100 < 6 * values.len(),
                    "{shape:?}, picture {seed}: {wrong}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn bend_is_what_the_areas_say_together() {
        // what a lens does to a picture that is held by its corners, up to three pixels
        let lens = |(x, y): (f64, f64)| {
            let far = x * x + y * y;
            (2.5 * x * (far - 1.2), 2.5 * y * (far - 1.2))
        };
        let spots = (0..96)
            .map(|spot| {
                (
                    (2 * (spot % 12) + 1) as f64 / 12.0 - 1.0,
                    (2 * (spot / 12) + 1) as f64 / 8.0 - 1.0,
                )
            })
            .collect::<Vec<_>>();
        // every area is found a fifth of a pixel from where it is at most, in no pattern
        let mut state = 0x2545_f491u32;
        let mut wobble = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            0.4 * (state >> 8) as f32 / (1 << 24) as f32 - 0.2
        };
        let found = spots
            .iter()
            .map(|spot| (lens(*spot).0 as f32 + wobble(), lens(*spot).1 as f32 + wobble()))
            .collect::<Vec<_>>();
        let weights = vec![32.0; 96];
        let off = |bend: &Bend| {
            spots
                .iter()
                .map(|spot| {
                    let ((x, y), (is_x, is_y)) = (bend.at(*spot), lens(*spot));
                    (x - is_x).abs().max((y - is_y).abs())
                })
                .fold(0.0, f64::max)
        };
        let bend = Bend::fit(&spots, &found, &weights);
        assert_eq!((bend.degree, bend.across.len()), (4, 15));
        assert!(off(&bend) < 0.12, "{}", off(&bend));
        // one area in eleven is found a sub-cell or two from where it is: the bend is that of the others
        let mut astray = found.clone();
        for found in astray.iter_mut().step_by(11) {
            *found = (found.0 + 2.0, found.1 - 4.0);
        }
        let bend = Bend::fit(&spots, &astray, &weights);
        assert!(off(&bend) < 0.3, "{}", off(&bend));
        // an area without a tile to find it by says nothing, wherever it is put
        let mut some = weights.clone();
        for (weight, found) in some.iter_mut().zip(&mut astray).step_by(11) {
            (*weight, *found) = (0.0, (40.0, 40.0));
        }
        assert!(off(&Bend::fit(&spots, &astray, &some)) < 0.3);
        // the fewer areas there are, the fewer powers the bend has: from thirty a third, from fifteen a
        // second, from six a first, and with fewer than that all tiles are off by the same
        for (areas, degree) in [
            (60, 4),
            (59, 3),
            (30, 3),
            (29, 2),
            (15, 2),
            (14, 1),
            (6, 1),
            (5, 0),
            (0, 0),
        ] {
            let bend = Bend::fit(&spots[..areas], &found[..areas], &weights[..areas]);
            assert_eq!(bend.degree, degree, "{areas} areas");
        }
        let flat = [(1.5f32, -0.5f32); 5];
        let (x, y) = Bend::fit(&spots[..5], &flat, &weights[..5]).at((0.3, -0.8));
        assert!((x - 1.5).abs() < 1e-6 && (y + 0.5).abs() < 1e-6, "{x} {y}");
        let nothing = Bend::fit(&[], &[], &[]);
        assert_eq!(nothing.at((0.0, 0.0)), (0.0, 0.0));
    }
}
