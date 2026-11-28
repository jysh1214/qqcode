//! The CRC-32 by which a file is named in the frames of the page, together with its length, and which
//! closes a frame and the index of a file on disk.

/// The remainders of the bytes 0 to 255 for [`checksum`].
const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut byte = 0;
    while byte < 256 {
        let mut remainder = byte as u32;
        let mut bit = 0;
        while bit < 8 {
            remainder = if remainder & 1 == 1 {
                0xedb8_8320 ^ (remainder >> 1)
            } else {
                remainder >> 1
            };
            bit += 1;
        }
        table[byte] = remainder;
        byte += 1;
    }
    table
};

/// CRC-32 as zip, gzip and PNG compute it, which is also what `crc32` of the shell and of Python print.
pub fn checksum(bytes: &[u8]) -> u32 {
    let mut checksum = Checksum::new();
    checksum.update(bytes);
    checksum.finish()
}

/// The [`checksum`] of bytes that come piece by piece.
pub struct Checksum(u32);

impl Checksum {
    pub fn new() -> Self {
        Self(!0)
    }

    pub fn update(&mut self, bytes: &[u8]) {
        let step = |crc: u32, byte: &u8| CRC_TABLE[((crc ^ *byte as u32) & 0xff) as usize] ^ (crc >> 8);
        self.0 = bytes.iter().fold(self.0, step);
    }

    pub fn finish(&self) -> u32 {
        !self.0
    }
}

impl Default for Checksum {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_is_the_crc_32_of_zip() {
        assert_eq!(checksum(b"123456789"), 0xcbf4_3926);
        assert_eq!(checksum(b""), 0);
        // piece by piece
        let mut pieces = Checksum::new();
        pieces.update(b"1234");
        pieces.update(b"");
        pieces.update(b"56789");
        assert_eq!(pieces.finish(), 0xcbf4_3926);
        assert_eq!(
            checksum(b"The quick brown fox jumps over the lazy dog"),
            0x414f_a339
        );
    }
}
