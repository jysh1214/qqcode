//! Reads the matrix of tiles ([`crate::tiles`]) out of a frame of the camera.
//!
//! The four finder patterns in the corners of the matrix say where it is. The frame is laid flat by them,
//! at the size the matrix has on the page, and then read tile by tile: which of the sixteen symbols a tile
//! looks most like, and which of the four colours its light has.
//!
//! A lens bends what the four corners can't account for, by a few pixels in the middle of a large matrix.
//! So no tile is read where it ought to be, but where it is found. One tile can't tell: a symbol that is
//! moved by a sub-cell looks much like another symbol. Some dozens of them together can, so the offset is
//! found for areas of eight by eight tiles, starting with those next to the finder patterns and each
//! around where its neighbours turned out to be, and a tile is as far off as the areas around it.

use anyhow::Result;
use opencv::core::{self, Mat, Point2f, Scalar, Size, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use crate::packet::Frame;
use crate::tiles::{self, ANCHOR, Grid, MODULE, PITCH, SCALES, SUB, Shape, TILE};

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
    /// The share of the bytes of a frame that were misread and put right
    pub wrong: f32,
}

/// How far a tile may be from where the corners put it, in pixels of the flat picture.
const MAX_DRIFT: i32 = 12;

/// Tiles along a side of the areas for which an offset is found, and how many pixels either way of where
/// its neighbours are an area is looked for.
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
    let values = read_tiles(flat.data_bytes()?, grid);
    let Some(unpacked) = tiles::unpack(grid, &values) else {
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
        wrong: unpacked.wrong as f32 / unpacked.bytes as f32,
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

/// The tiles of the flat picture of a matrix, which has blue, green and red for every pixel: a value
/// from 0 to 63 for every cell, as [`tiles::pack`] has them.
pub fn read_tiles(flat: &[u8], grid: Grid) -> Vec<u8> {
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
    // tiles are most like symbols taken together. The areas are gone through outwards from the corners,
    // which are where they ought to be, and each is looked for around where its neighbours were found.
    let areas = (grid.columns.div_ceil(AREA), grid.rows.div_ceil(AREA));
    let area = |column: usize, row: usize| row * areas.0 + column;
    let mut order = (0..areas.1)
        .flat_map(|row| (0..areas.0).map(move |column| (column, row)))
        .collect::<Vec<_>>();
    order.sort_by_key(|(column, row)| {
        let across = (*column).min(areas.0 - 1 - column);
        let down = (*row).min(areas.1 - 1 - row);
        across * across + down * down
    });
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
        // every other tile of the area tells as much as all of them
        let sample = (AREA * row..(AREA * (row + 1)).min(grid.rows))
            .flat_map(|tile_row| {
                (AREA * column..(AREA * (column + 1)).min(grid.columns))
                    .map(move |tile_column| (tile_column, tile_row))
            })
            .filter(|(tile_column, tile_row)| {
                (tile_column + tile_row) % 2 == 0 && !grid.is_anchor(*tile_column, *tile_row)
            })
            .map(|(tile_column, tile_row)| grid.origin(tile_column, tile_row))
            .collect::<Vec<_>>();
        let likeness = |offset: (i32, i32)| -> i64 {
            if offset.0.abs() > MAX_DRIFT || offset.1.abs() > MAX_DRIFT {
                return 0;
            }
            sample
                .iter()
                .filter_map(|(left, top)| look(*left as i32 + offset.0, *top as i32 + offset.1))
                .map(|found| found.best as i64)
                .sum()
        };
        let mut chosen = (expected, likeness(expected));
        for beside in -SEARCH..=SEARCH {
            for below in -SEARCH..=SEARCH {
                let offset = (expected.0 + beside, expected.1 + below);
                let like = likeness(offset);
                if like > chosen.1 {
                    chosen = (offset, like);
                }
            }
        }
        // Between whole pixels, the offset is where a parabola through the likeness at the best pixel and
        // at the two next to it has its top.
        let ((x, y), best) = chosen;
        let between = |before: i64, after: i64| {
            let bend = (before - 2 * best + after) as f32;
            if bend < 0.0 {
                (0.5 * (before - after) as f32 / bend).clamp(-0.5, 0.5)
            } else {
                0.0
            }
        };
        let fine = (
            x as f32 + between(likeness((x - 1, y)), likeness((x + 1, y))),
            y as f32 + between(likeness((x, y - 1)), likeness((x, y + 1))),
        );
        drift[area(column, row)] = Some(fine);
    }
    // a tile is as far off as the areas around it are, by how near it is to the middle of each
    let offset_of = |column: usize, row: usize| -> (i32, i32) {
        let place = |tile: usize, count: usize| {
            let at = ((tile as f32 + 0.5) / AREA as f32 - 0.5).clamp(0.0, count as f32 - 1.0);
            let first = (at.floor() as usize).min(count.saturating_sub(2));
            (first, (first + 1).min(count - 1), at - first as f32)
        };
        let (left, right, across) = place(column, areas.0);
        let (top, bottom, down) = place(row, areas.1);
        let at = |column: usize, row: usize| drift[area(column, row)].unwrap_or((0.0, 0.0));
        let blend = |pick: fn((f32, f32)) -> f32| {
            let upper = pick(at(left, top)) * (1.0 - across) + pick(at(right, top)) * across;
            let lower = pick(at(left, bottom)) * (1.0 - across) + pick(at(right, bottom)) * across;
            (upper * (1.0 - down) + lower * down).round() as i32
        };
        (blend(|offset| offset.0), blend(|offset| offset.1))
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
    let mut symbols = vec![0u8; grid.columns * grid.rows];
    for ((column, row), found) in grid.cells().zip(&seen) {
        let pattern = pattern_of(&found.likeness, &strength);
        let exchanged = if found.likeness[pattern] < 0 { 8 } else { 0 };
        symbols[cell(column, row)] = pattern as u8 | exchanged;
    }

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
    let colours = classify(&tints);

    grid.cells()
        .zip(colours)
        .map(|((column, row), colour)| symbols[cell(column, row)] << 2 | colour)
        .collect()
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

/// Which of the four colours each tint is. The colours are those of the page as a camera sees them, which
/// is not as they are on the page: one camera has two thirds of the red in a cyan tile that it has in the
/// white tile next to it. So the four begin where this frame has them, as far as it shows that: a channel
/// that the tiles have in two [`groups`] says how much of it white has, and how much the colour that
/// lacks it. Where a channel says nothing, they begin as they ought to be. They are then moved, a few
/// times over, to the middle of the tints that are nearest to each.
fn classify(tints: &[[f32; 3]]) -> Vec<u8> {
    // blue, green, red, as the frame has them
    let pure: [[f32; 3]; 4] =
        tiles::COLOURS.map(|[red, green, blue]| [blue, green, red].map(|c| c as f32 / 255.0));
    let nearest = |centres: &[[f32; 3]; 4], tint: &[f32; 3]| {
        let off = |centre: &[f32; 3]| {
            centre
                .iter()
                .zip(tint)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
        };
        (0..4u8)
            .min_by(|a, b| off(&centres[*a as usize]).total_cmp(&off(&centres[*b as usize])))
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
                let off = moved
                    .iter()
                    .zip(&begin[colour])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f32>();
                // and none goes over to where another one belongs
                if off < 0.6 * 0.6 {
                    centres[colour] = moved;
                }
            }
        }
    }
    tints.iter().map(|tint| nearest(&centres, tint)).collect()
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
            assert_eq!(read_tiles(&picture(grid, &values), grid), values);
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
        let read = read_tiles(bent.data_bytes()?, grid);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        // a tile here and there, which the check bytes make up for
        assert!(wrong * 100 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &read).is_some());
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
        let read = read_tiles(bent.data_bytes()?, grid);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        assert!(wrong * 50 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &read).is_some());
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
        let read = read_tiles(bent.data_bytes()?, grid);
        let wrong = read.iter().zip(&values).filter(|(a, b)| a != b).count();
        assert!(wrong * 100 < values.len(), "{wrong} of {}", values.len());
        assert!(tiles::unpack(grid, &read).is_some());
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
}
