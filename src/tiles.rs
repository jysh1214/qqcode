//! The picture that the web page of `qqcode-show` puts on a screen: a matrix of coloured tiles.
//!
//! A tile is a square of 8 pixels on a dark ground, one of sixteen symbols in one of four colours, which
//! makes six bits. The symbols are made of sub-cells of 2 pixels, lit or dark, in patterns that are as
//! unlike each other as sixteen patterns of four by four sub-cells can be and that have no feature finer
//! than a sub-cell: a camera that blurs them still tells them apart. Half of every symbol is lit, so every
//! tile shows its colour. In the four corners of the matrix sit the finder patterns of a QR code, by which
//! the picture is found in a frame and laid flat.
//!
//! The bytes of a frame are spread over two planes, the symbols and the colours, each with Reed-Solomon
//! check bytes of its own ([`crate::rs`]), more for the symbols, which a camera that is not sharp or that
//! packs its frames gets wrong before it gets a colour wrong. The page computes
//! all of this in JavaScript, so nothing here can change without it.

use crate::rs;

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

/// Check bytes to a code word of at most 255 bytes, in the plane of the symbols and in that of the colours.
pub const SYMBOL_PARITY: usize = 64;
pub const COLOUR_PARITY: usize = 40;

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

    fn planes(&self) -> (Plane, Plane) {
        let symbols = Plane {
            bytes: self.tiles() / 2,
            parity: SYMBOL_PARITY,
        };
        let colours = Plane {
            bytes: self.tiles() / 4,
            parity: COLOUR_PARITY,
        };
        (symbols, colours)
    }

    /// How many bytes a frame holds once the check bytes are taken off.
    pub fn capacity(&self) -> usize {
        let (symbols, colours) = self.planes();
        symbols.data() + colours.data()
    }
}

/// The bytes that the tiles spell in their symbols or in their colours, in code words that are interleaved:
/// the first byte of every word, then the second of every word, and so on. Damage in one place of the
/// picture is then shared out among all the words.
struct Plane {
    bytes: usize,
    parity: usize,
}

impl Plane {
    /// The lengths of the code words: as few words as hold the bytes, as equal as they come.
    fn words(&self) -> Vec<usize> {
        let count = self.bytes.div_ceil(255);
        (0..count)
            .map(|word| self.bytes / count + usize::from(word < self.bytes % count))
            .collect()
    }

    fn data(&self) -> usize {
        self.bytes - self.words().len() * self.parity
    }

    fn encode(&self, data: &[u8]) -> Vec<u8> {
        assert_eq!(data.len(), self.data());
        let mut rest = data;
        let mut words = Vec::new();
        for length in self.words() {
            let (word, tail) = rest.split_at(length - self.parity);
            let mut word = word.to_vec();
            word.extend(rs::encode(&word, self.parity));
            words.push(word);
            rest = tail;
        }
        let mut plane = Vec::with_capacity(self.bytes);
        for place in 0..words.first().map_or(0, Vec::len) {
            plane.extend(words.iter().filter_map(|word| word.get(place)));
        }
        plane
    }

    /// The data in a plane as it was read, and how many of its bytes had to be put right; `None` if a
    /// word has more wrong bytes than its check bytes make up for.
    fn decode(&self, plane: &[u8]) -> Option<(Vec<u8>, usize)> {
        let lengths = self.words();
        let mut words = lengths
            .iter()
            .map(|length| Vec::with_capacity(*length))
            .collect::<Vec<_>>();
        let mut bytes = plane.iter();
        for place in 0..lengths.first().copied().unwrap_or(0) {
            for (word, length) in words.iter_mut().zip(&lengths) {
                if place < *length {
                    word.push(*bytes.next()?);
                }
            }
        }
        let mut data = Vec::with_capacity(self.data());
        let mut wrong = 0;
        for word in &mut words {
            wrong += rs::correct(word, self.parity)?;
            data.extend(&word[..word.len() - self.parity]);
        }
        Some((data, wrong))
    }
}

/// The numbers that the planes are mixed with, so that the tiles of a frame look alike in no two places
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
    let (symbols, colours) = grid.planes();
    assert_eq!(payload.len(), grid.capacity(), "a payload that fills the frame");
    let (first, second) = payload.split_at(symbols.data());
    let (mut symbols, mut colours) = (symbols.encode(first), colours.encode(second));
    let mut state = MIX;
    mix(&mut symbols, &mut state);
    mix(&mut colours, &mut state);
    (0..grid.tiles())
        .map(|tile| {
            // what is left over of the cells when the bytes are full stays 0
            let symbol = symbols
                .get(tile / 2)
                .map_or(0, |byte| (byte >> (4 - 4 * (tile % 2))) & 15);
            let colour = colours
                .get(tile / 4)
                .map_or(0, |byte| (byte >> (6 - 2 * (tile % 4))) & 3);
            symbol << 2 | colour
        })
        .collect()
}

/// What a frame holds, read from its tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unpacked {
    pub payload: Vec<u8>,
    /// How many bytes of the two planes were misread and put right
    pub wrong: usize,
    /// How many bytes the two planes have
    pub bytes: usize,
}

/// The payload of the frame with these tiles, or `None` if too many of them are misread.
pub fn unpack(grid: Grid, tiles: &[u8]) -> Option<Unpacked> {
    if tiles.len() != grid.tiles() {
        return None;
    }
    let (symbols, colours) = grid.planes();
    let mut symbol_bytes = vec![0u8; symbols.bytes];
    let mut colour_bytes = vec![0u8; colours.bytes];
    for (tile, value) in tiles.iter().enumerate() {
        if let Some(byte) = symbol_bytes.get_mut(tile / 2) {
            *byte |= (value >> 2 & 15) << (4 - 4 * (tile % 2));
        }
        if let Some(byte) = colour_bytes.get_mut(tile / 4) {
            *byte |= (value & 3) << (6 - 2 * (tile % 4));
        }
    }
    let mut state = MIX;
    mix(&mut symbol_bytes, &mut state);
    mix(&mut colour_bytes, &mut state);
    let (mut payload, wrong_symbols) = symbols.decode(&symbol_bytes)?;
    let (rest, wrong_colours) = colours.decode(&colour_bytes)?;
    payload.extend(rest);
    Some(Unpacked {
        payload,
        wrong: wrong_symbols + wrong_colours,
        bytes: symbols.bytes + colours.bytes,
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
    fn frame_holds_what_the_check_bytes_leave() {
        // 17280 tiles: 8640 bytes of symbols in 34 words with 64 check bytes each, 4320 bytes of colours in
        // 17 words with 40
        assert_eq!(WIDE.capacity(), 8640 - 34 * 64 + 4320 - 17 * 40);
        assert_eq!(WIDE.capacity(), 10104);
        let small = Grid::new(Shape::Square, 4);
        assert_eq!(small.capacity(), 576 - 3 * 64 + 288 - 2 * 40);
        for shape in Shape::ALL {
            for scale in SCALES {
                let grid = Grid::new(shape, scale);
                let (symbols, colours) = grid.planes();
                for plane in [symbols, colours] {
                    let words = plane.words();
                    assert_eq!(words.iter().sum::<usize>(), plane.bytes);
                    assert!(
                        words
                            .iter()
                            .all(|length| (plane.parity + 1..=255).contains(length))
                    );
                }
            }
        }
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
            let unpacked = unpack(grid, &tiles).unwrap();
            assert_eq!((unpacked.payload, unpacked.wrong), (payload, 0));
            assert_eq!(unpacked.bytes, grid.tiles() / 2 + grid.tiles() / 4);
        }
        assert_eq!(unpack(WIDE, &[0; 100]), None);
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
        // a patch of 30 by 30 tiles is read as anything, as under a reflection
        let mut damaged = tiles.clone();
        for ((column, row), tile) in WIDE.cells().zip(&mut damaged) {
            if (60..90).contains(&column) && (40..70).contains(&row) {
                *tile = (*tile + 21) % 64;
            }
        }
        let unpacked = unpack(WIDE, &damaged).unwrap();
        assert_eq!(unpacked.payload, payload);
        assert!((600..=720).contains(&unpacked.wrong), "{}", unpacked.wrong);
        // one tile in sixty all over the picture, each in a byte of colours of its own
        let mut speckled = tiles.clone();
        for tile in speckled.iter_mut().step_by(60) {
            *tile ^= 0b10_1101;
        }
        assert_eq!(unpack(WIDE, &speckled).unwrap().payload, payload);
        // half the picture is beyond it
        let mut lost = tiles;
        lost[..8000].fill(0);
        assert_eq!(unpack(WIDE, &lost), None);
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
        assert_eq!(
            unpack(grid, &tiles).map(|unpacked| unpacked.payload),
            Some(payload)
        );
        // read with the wrong guess of what the matrix is: a tile too many, and one too few
        tiles.push(0);
        assert_eq!(unpack(grid, &tiles), None);
        tiles.truncate(tiles.len() - 2);
        assert_eq!(unpack(grid, &tiles), None);
    }

    #[test]
    fn planes_that_are_a_whole_number_of_words_have_no_word_more() {
        // 256 by 144 tiles less the corners: 18360 bytes of symbols, which are 72 words of 255 bytes, and
        // 9180 bytes of colours, which are 36
        let grid = Grid::new(Shape::Wide, 16);
        let (symbols, colours) = grid.planes();
        assert_eq!((symbols.words(), colours.words()), (vec![255; 72], vec![255; 36]));
        assert_eq!(grid.capacity(), 18360 - 72 * 64 + 9180 - 36 * 40);
    }
}
