//! The picture that the web page of `qqcode-show` puts on a screen: a matrix of coloured tiles.
//!
//! A tile is a square of 8 pixels on a dark ground, one of sixteen symbols in one of four colours, which
//! makes six bits. The symbols are made of sub-cells of 2 pixels, lit or dark, in patterns that are as
//! unlike each other as sixteen patterns of four by four sub-cells can be and that have no feature finer
//! than a sub-cell: a camera that blurs them still tells them apart. Half of every symbol is lit, so every
//! tile shows its colour. In the four corners of the matrix sit the finder patterns of a QR code, by which
//! the picture is found in a frame and laid flat.
//!
//! The bytes of a frame are the data of one code word ([`crate::ldpc`]) whose bits are spread over the
//! tiles in no order, four to the symbol of a tile and two to its colour, so that what a camera gets wrong
//! in one place of the picture is shared out among all the checks. The page computes all of this in
//! JavaScript, so nothing here can change without it.

use std::sync::{Arc, Mutex, PoisonError};

use crate::ldpc::{Code, Draw};

/// Pixels along a side of a tile, and of a sub-cell of it.
pub const TILE: usize = 8;
pub const SUB: usize = 2;
/// Pixels from one tile to the next: a tile and a dark gap.
pub const PITCH: usize = 9;
/// Dark pixels around the matrix.
pub const BORDER: usize = 4;
/// Cells along a side of the light square that a corner gives to its finder pattern.
pub const ANCHOR: usize = 6;
/// Pixels to a module of a finder pattern, which is seven of them wide.
pub const MODULE: usize = 6;

/// Bits to a tile: four that say which symbol it is and two that say which colour.
pub const BITS: usize = 6;

/// How many of the bits of a frame are data: so many in so many. The others are check bits.
pub const DATA: (usize, usize) = (3, 4);

/// Rounds through the checks that the reader gives a frame.
const ROUNDS: usize = 50;

/// The colours of the tiles as red, green, blue: white, and white less one of the three.
pub const COLOURS: [[u8; 3]; 4] = [[255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]];

/// The shape of the matrix, best that of the camera's frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum Shape {
    /// 16:9
    Wide,
    /// 4:3
    Classic,
    /// 1:1
    Square,
}

impl Shape {
    pub const ALL: [Shape; 3] = [Shape::Wide, Shape::Classic, Shape::Square];

    /// Columns for every 9 rows.
    fn columns(self) -> usize {
        match self {
            Shape::Wide => 16,
            Shape::Classic => 12,
            Shape::Square => 9,
        }
    }
}

/// The scales that a matrix comes in: it has 9 rows for each.
pub const SCALES: std::ops::RangeInclusive<usize> = 4..=24;

/// How many tiles a matrix has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub columns: usize,
    pub rows: usize,
}

impl Grid {
    pub fn new(shape: Shape, scale: usize) -> Self {
        assert!(
            SCALES.contains(&scale),
            "a matrix has the scales 4 to 24, not {scale}"
        );
        Self {
            columns: shape.columns() * scale,
            rows: 9 * scale,
        }
    }

    /// Pixels across and down.
    pub fn width(&self) -> usize {
        2 * BORDER + self.columns * PITCH
    }

    pub fn height(&self) -> usize {
        2 * BORDER + self.rows * PITCH
    }

    /// Whether the cell belongs to the finder pattern of a corner.
    pub fn is_anchor(&self, column: usize, row: usize) -> bool {
        (column < ANCHOR || column >= self.columns - ANCHOR) && (row < ANCHOR || row >= self.rows - ANCHOR)
    }

    /// The cells that hold tiles, row by row, as column and row.
    pub fn cells(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.rows)
            .flat_map(move |row| (0..self.columns).map(move |column| (column, row)))
            .filter(|(column, row)| !self.is_anchor(*column, *row))
    }

    /// How many cells hold tiles.
    pub fn tiles(&self) -> usize {
        self.columns * self.rows - 4 * ANCHOR * ANCHOR
    }

    /// The pixel at the top-left of a cell.
    pub fn origin(&self, column: usize, row: usize) -> (usize, usize) {
        (BORDER + column * PITCH, BORDER + row * PITCH)
    }

    /// The middles of the four finder patterns, top-left, top-right, bottom-right, bottom-left, in pixels
    /// counted from the middle of the first one.
    pub fn anchors(&self) -> [(f32, f32); 4] {
        let middle = (ANCHOR * PITCH) as f32 / 2.0 - 0.5;
        let near = BORDER as f32 + middle;
        let right = (BORDER + (self.columns - ANCHOR) * PITCH) as f32 + middle;
        let low = (BORDER + (self.rows - ANCHOR) * PITCH) as f32 + middle;
        [(near, near), (right, near), (right, low), (near, low)]
    }

    /// How many bytes a frame holds once the check bits are taken off.
    pub fn capacity(&self) -> usize {
        self.tiles() * BITS * DATA.0 / DATA.1 / 8
    }
}

/// What the bits of a frame are made with: the code, and the place among the bits of the tiles that each
/// bit of a code word has.
struct Layout {
    code: Code,
    places: Vec<u32>,
}

/// Where the sequence of numbers begins that the places of the bits are drawn from.
const PLACES: u32 = 0x6c07_8965;

/// The layout of a matrix, which follows from how many tiles it has. It is made once.
fn layout(grid: Grid) -> Arc<Layout> {
    static MADE: Mutex<Vec<(usize, Arc<Layout>)>> = Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, layout)) = made.iter().find(|(tiles, _)| *tiles == grid.tiles()) {
        return layout.clone();
    }
    let bits = grid.tiles() * BITS;
    let layout = Arc::new(Layout {
        code: Code::new(bits, 8 * grid.capacity()),
        places: Draw(PLACES).order(bits),
    });
    made.push((grid.tiles(), layout.clone()));
    layout
}

/// The numbers that the tiles are mixed with, so that the tiles of a frame look alike in no two places
/// whatever its bytes are: this many of a sequence that goes on from `state`.
fn mix(bytes: &mut [u8], state: &mut u32) {
    for byte in bytes {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        *byte ^= *state as u8;
    }
}

const MIX: u32 = 0x9e37_79b9;

/// The tiles of the frame that holds `payload`, which is as long as the grid's capacity: a value from 0 to
/// 63 for every cell, its symbol times four plus its colour.
pub fn pack(grid: Grid, payload: &[u8]) -> Vec<u8> {
    assert_eq!(payload.len(), grid.capacity(), "a payload that fills the frame");
    let layout = layout(grid);
    let data = payload
        .iter()
        .flat_map(|byte| (0..8).rev().map(move |bit| byte >> bit & 1))
        .collect::<Vec<_>>();
    let mut tiles = vec![0u8; grid.tiles()];
    for (bit, place) in layout.code.encode(&data).iter().zip(&layout.places) {
        let place = *place as usize;
        tiles[place / BITS] |= bit << (BITS - 1 - place % BITS);
    }
    mix(&mut tiles, &mut { MIX });
    tiles.iter_mut().for_each(|tile| *tile &= 63);
    tiles
}

/// How sure a reader is of the six bits of a tile, the first of its symbol first: above zero for a 0 and
/// below it for a 1, the further from zero the surer.
pub type Sure = [f32; BITS];

/// A reader that has no doubt of these tiles.
pub fn certain(tiles: &[u8]) -> Vec<Sure> {
    tiles
        .iter()
        .map(|tile| {
            std::array::from_fn(|bit| {
                if tile >> (BITS - 1 - bit) & 1 == 1 {
                    -1.0
                } else {
                    1.0
                }
            })
        })
        .collect()
}

/// The tiles as a reader takes them to be that has to say one thing of each.
pub fn likeliest(sure: &[Sure]) -> Vec<u8> {
    sure.iter()
        .map(|tile| {
            tile.iter()
                .fold(0, |value, bit| value << 1 | u8::from(*bit < 0.0))
        })
        .collect()
}

/// What a frame holds, read from its tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unpacked {
    pub payload: Vec<u8>,
    /// How many tiles were misread and put right
    pub wrong: usize,
    /// How many tiles the frame has
    pub tiles: usize,
}

/// The payload of the frame with these tiles, or `None` if too many of them are misread.
pub fn unpack(grid: Grid, sure: &[Sure]) -> Option<Unpacked> {
    if sure.len() != grid.tiles() {
        return None;
    }
    let layout = layout(grid);
    let mut mixed = vec![0u8; grid.tiles()];
    mix(&mut mixed, &mut { MIX });
    // the bits of the code word as they are before they are mixed: a bit that the mixing turns over is as
    // sure as it was, of the other value
    let seen = layout
        .places
        .iter()
        .map(|place| {
            let (tile, bit) = (*place as usize / BITS, *place as usize % BITS);
            let value = sure[tile][bit];
            if mixed[tile] >> (BITS - 1 - bit) & 1 == 1 {
                -value
            } else {
                value
            }
        })
        .collect::<Vec<_>>();
    let word = layout.code.decode(&seen, ROUNDS)?;
    let payload = word[..layout.code.data()]
        .chunks_exact(8)
        .map(|byte| byte.iter().fold(0, |value, bit| value << 1 | bit))
        .collect();
    let mut misread = vec![false; grid.tiles()];
    for ((bit, seen), place) in word.iter().zip(&seen).zip(&layout.places) {
        misread[*place as usize / BITS] |= (*bit == 1) != (*seen < 0.0);
    }
    Some(Unpacked {
        payload,
        wrong: misread.iter().filter(|misread| **misread).count(),
        tiles: grid.tiles(),
    })
}

/// The waves that a symbol is made of along either axis: flat, one half lit, the middle lit.
const WAVES: [[bool; 4]; 3] = [
    [false, false, false, false],
    [false, false, true, true],
    [false, true, true, false],
];

/// The wave down and the wave across of the eight patterns; the flat one in both is left out, as nothing
/// of it would be lit.
const PATTERNS: [(usize, usize); 8] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2)];

/// Whether the sub-cell in `row` and `column` (0 to 3) of a symbol (0 to 15) is lit. A sub-cell is lit
/// where the two waves of the pattern differ, and the symbols 8 to 15 are the first eight with lit and
/// dark exchanged.
pub fn lit(symbol: u8, row: usize, column: usize) -> bool {
    let (down, across) = PATTERNS[(symbol & 7) as usize];
    WAVES[down][row] ^ WAVES[across][column] ^ (symbol & 8 != 0)
}

/// The picture of a frame with these tiles: red, green and blue for every pixel, row by row.
pub fn render(grid: Grid, tiles: &[u8]) -> Vec<u8> {
    assert_eq!(tiles.len(), grid.tiles());
    let (width, height) = (grid.width(), grid.height());
    let mut picture = vec![0u8; 3 * width * height];
    let mut fill = |left: usize, top: usize, side: usize, colour: [u8; 3]| {
        for y in top..top + side {
            for x in left..left + side {
                picture[3 * (y * width + x)..][..3].copy_from_slice(&colour);
            }
        }
    };
    // the finder patterns: dark, light, dark squares of 7, 5 and 3 modules in a light square
    let (far_x, far_y) = (grid.columns - ANCHOR, grid.rows - ANCHOR);
    for (column, row) in [(0, 0), (far_x, 0), (far_x, far_y), (0, far_y)] {
        let (left, top) = grid.origin(column, row);
        let (light, dark) = ([255; 3], [0; 3]);
        fill(left, top, ANCHOR * PITCH, light);
        let inset = (ANCHOR * PITCH - 7 * MODULE) / 2;
        fill(left + inset, top + inset, 7 * MODULE, dark);
        fill(left + inset + MODULE, top + inset + MODULE, 5 * MODULE, light);
        fill(
            left + inset + 2 * MODULE,
            top + inset + 2 * MODULE,
            3 * MODULE,
            dark,
        );
    }
    for ((column, row), value) in grid.cells().zip(tiles) {
        let (left, top) = grid.origin(column, row);
        let colour = COLOURS[(value & 3) as usize];
        for sub_row in 0..TILE / SUB {
            for sub_column in 0..TILE / SUB {
                if lit(value >> 2, sub_row, sub_column) {
                    fill(left + SUB * sub_column, top + SUB * sub_row, SUB, colour);
                }
            }
        }
    }
    picture
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(count: usize, seed: u32) -> Vec<u8> {
        let mut state = seed | 1;
        let mut bytes = vec![0u8; count];
        mix(&mut bytes, &mut state);
        bytes
    }

    const WIDE: Grid = Grid {
        columns: 176,
        rows: 99,
    };

    #[test]
    fn grid_has_the_shape_and_scale_asked_for() {
        assert_eq!(Grid::new(Shape::Wide, 11), WIDE);
        assert_eq!((WIDE.width(), WIDE.height()), (1592, 899));
        assert_eq!(WIDE.tiles(), 176 * 99 - 144);
        assert_eq!(WIDE.cells().count(), WIDE.tiles());
        let classic = Grid::new(Shape::Classic, 8);
        assert_eq!((classic.columns, classic.rows), (96, 72));
        assert_eq!(Grid::new(Shape::Square, 4).tiles(), 36 * 36 - 144);
        // the corners are left out, and nothing else
        let cells = WIDE.cells().collect::<Vec<_>>();
        assert_eq!(cells[0], (6, 0));
        assert_eq!(cells[163], (169, 0));
        assert_eq!(cells[164], (6, 1));
        assert!(cells.contains(&(0, 6)) && cells.contains(&(175, 92)) && !cells.contains(&(175, 93)));
        assert_eq!(WIDE.anchors()[0], (30.5, 30.5));
        assert_eq!(WIDE.anchors()[2], (1560.5, 867.5));
    }

    #[test]
    fn frame_holds_what_the_check_bits_leave() {
        // 17280 tiles of 6 bits, three in four of them data
        assert_eq!(WIDE.capacity(), 17280 * 6 * 3 / 4 / 8);
        assert_eq!(WIDE.capacity(), 9720);
        let small = Grid::new(Shape::Square, 4);
        assert_eq!(small.capacity(), 648);
        // a matrix whose bits do not come out in whole bytes: the bytes that are whole are the data, and
        // what is left over are check bits
        let odd = Grid::new(Shape::Classic, 7);
        assert_eq!((odd.tiles(), odd.capacity()), (5148, 2895));
        for shape in Shape::ALL {
            for scale in SCALES {
                let grid = Grid::new(shape, scale);
                let (bits, data) = (grid.tiles() * BITS, 8 * grid.capacity());
                assert!(data * DATA.1 <= bits * DATA.0 && bits * DATA.0 < (data + 8) * DATA.1);
            }
        }
    }

    #[test]
    fn bits_of_a_frame_each_have_a_place_of_their_own() {
        let grid = Grid::new(Shape::Square, 4);
        let layout = layout(grid);
        assert_eq!((layout.code.bits(), layout.code.data()), (6912, 5184));
        let mut places = layout.places.clone();
        places.sort_unstable();
        assert!(places.iter().copied().eq(0..6912));
        // in no order: the first bits of the code word are not the first tiles
        assert!(layout.places[..60].iter().any(|place| *place >= 60));
        // and the layout is made once
        assert!(Arc::ptr_eq(&layout, &super::layout(grid)));
    }

    #[test]
    fn symbols_are_half_lit_and_far_from_each_other() {
        let pattern = |symbol: u8| {
            (0..16)
                .map(|cell| lit(symbol, cell / 4, cell % 4))
                .collect::<Vec<_>>()
        };
        for symbol in 0..16 {
            assert_eq!(
                pattern(symbol).iter().filter(|lit| **lit).count(),
                8,
                "symbol {symbol}"
            );
            for other in 0..symbol {
                let differ = pattern(symbol)
                    .iter()
                    .zip(pattern(other))
                    .filter(|(a, b)| **a != *b)
                    .count();
                // exchanged, or half of the sub-cells
                let expected = if symbol == other + 8 { 16 } else { 8 };
                assert_eq!(differ, expected, "symbols {other} and {symbol}");
            }
        }
    }

    #[test]
    fn tiles_come_back_as_the_payload() {
        for grid in [WIDE, Grid::new(Shape::Square, 4), Grid::new(Shape::Classic, 7)] {
            let payload = bytes(grid.capacity(), 7);
            let tiles = pack(grid, &payload);
            assert_eq!(tiles.len(), grid.tiles());
            assert!(tiles.iter().all(|tile| *tile < 64));
            assert_eq!(likeliest(&certain(&tiles)), tiles);
            let unpacked = unpack(grid, &certain(&tiles)).unwrap();
            assert!(unpacked.payload == payload);
            assert_eq!((unpacked.wrong, unpacked.tiles), (0, grid.tiles()));
        }
        assert!(unpack(WIDE, &certain(&[0; 100])).is_none());
    }

    #[test]
    fn tiles_of_an_empty_payload_are_mixed_all_the_same() {
        let tiles = pack(WIDE, &vec![0; WIDE.capacity()]);
        let mut counts = [0; 64];
        for tile in &tiles {
            counts[*tile as usize] += 1;
        }
        // 270 of each on average
        assert!(
            counts.iter().all(|count| (180..360).contains(count)),
            "{counts:?}"
        );
    }

    #[test]
    fn misread_tiles_are_put_right_up_to_a_point() {
        let payload = bytes(WIDE.capacity(), 9);
        let tiles = pack(WIDE, &payload);
        // a patch of 20 by 20 tiles is read as anything and as surely as the others, as between two frames
        let mut damaged = tiles.clone();
        for ((column, row), tile) in WIDE.cells().zip(&mut damaged) {
            if (60..80).contains(&column) && (40..60).contains(&row) {
                *tile = (*tile + 21) % 64;
            }
        }
        let unpacked = unpack(WIDE, &certain(&damaged)).unwrap();
        assert!(unpacked.payload == payload);
        assert_eq!(unpacked.wrong, 400);
        // a patch four times as large that the reader makes nothing of, as under a reflection: nothing is
        // known of its bits
        let mut blind = certain(&tiles);
        for ((column, row), tile) in WIDE.cells().zip(&mut blind) {
            if (50..90).contains(&column) && (30..70).contains(&row) {
                *tile = [0.0; BITS];
            }
        }
        assert!(unpack(WIDE, &blind).unwrap().payload == payload);
        // one tile in ten all over the picture is another, each less sure than the tiles that are right
        let mut doubtful = certain(&tiles);
        for tile in doubtful.iter_mut().step_by(10) {
            *tile = tile.map(|bit| -0.25 * bit);
        }
        let unpacked = unpack(WIDE, &doubtful).unwrap();
        assert!(unpacked.payload == payload);
        assert_eq!((unpacked.wrong, unpacked.tiles), (1728, 17280));
        // half the picture is beyond it
        let mut lost = tiles;
        lost[..8000].fill(0);
        assert!(unpack(WIDE, &certain(&lost)).is_none());
    }

    #[test]
    fn picture_has_the_finder_patterns_and_the_tiles() {
        let grid = Grid::new(Shape::Square, 4);
        let mut tiles = vec![0u8; grid.tiles()];
        // symbol 9, the lower half lit, in yellow, as the first tile, which is right of a finder pattern
        tiles[0] = 9 << 2 | 3;
        let picture = render(grid, &tiles);
        let at = |x: usize, y: usize| {
            let pixel = &picture[3 * (y * grid.width() + x)..][..3];
            [pixel[0], pixel[1], pixel[2]]
        };
        assert_eq!(picture.len(), 3 * 332 * 332);
        let (white, black, yellow) = ([255; 3], [0; 3], [255, 255, 0]);
        // the border, the light square, and the rings of the pattern along its middle row
        assert_eq!(
            [at(0, 0), at(3, 30), at(4, 4), at(9, 30)],
            [black, black, white, white]
        );
        assert_eq!(
            [at(10, 30), at(16, 30), at(22, 30), at(39, 30)],
            [black, white, black, black]
        );
        assert_eq!(
            [at(40, 30), at(46, 30), at(52, 30), at(57, 30)],
            [white, black, white, white]
        );
        // symbol 9 is the pattern (1, 0) exchanged: dark in the lower half
        assert_eq!(
            [at(58, 4), at(65, 7), at(58, 8), at(65, 11)],
            [yellow, yellow, black, black]
        );
        // the gap after it, and the next tile: symbol 0 in white, lit on the right
        assert_eq!(
            [at(66, 4), at(67, 4), at(71, 4), at(74, 11)],
            [black, black, white, white]
        );
        // the bottom-right finder pattern
        assert_eq!(
            [at(297, 297), at(301, 301), at(327, 327), at(328, 328)],
            [black, black, white, black]
        );
    }

    #[test]
    fn tiles_of_another_count_than_the_grid_has_are_no_frame() {
        let grid = Grid::new(Shape::Square, 4);
        let payload = (0..grid.capacity()).map(|at| at as u8).collect::<Vec<_>>();
        let mut tiles = pack(grid, &payload);
        assert!(unpack(grid, &certain(&tiles)).map(|unpacked| unpacked.payload) == Some(payload));
        // read with the wrong guess of what the matrix is: a tile too many, and one too few
        tiles.push(0);
        assert!(unpack(grid, &certain(&tiles)).is_none());
        tiles.truncate(tiles.len() - 2);
        assert!(unpack(grid, &certain(&tiles)).is_none());
    }
}
