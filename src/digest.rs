//! The SHA-256 and the MD5 of a file, which both programs print when they end: `qqcode-show` those of the
//! file it showed, `qqcode` those of the file it wrote, to be compared with each other or with what
//! `sha256sum` and `md5sum` say. They are not in the codes, where a file is named by its CRC-32.

use md5::Md5;
use sha2::{Digest, Sha256};

/// The SHA-256 of `bytes` in hexadecimal, as `sha256sum` prints it.
pub fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// The MD5 of `bytes` in hexadecimal, as `md5sum` prints it.
pub fn md5(bytes: &[u8]) -> String {
    hex(&Md5::digest(bytes))
}

/// The SHA-256 and the MD5 of bytes that come piece by piece.
#[derive(Default)]
pub struct Digests {
    sha256: Sha256,
    md5: Md5,
}

impl Digests {
    pub fn update(&mut self, bytes: &[u8]) {
        self.sha256.update(bytes);
        self.md5.update(bytes);
    }

    /// The two in hexadecimal, the SHA-256 first.
    pub fn finish(self) -> (String, String) {
        (hex(&self.sha256.finalize()), hex(&self.md5.finalize()))
    }
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOX: &[u8] = b"The quick brown fox jumps over the lazy dog";

    #[test]
    fn sha_256_is_that_of_sha256sum() {
        assert_eq!(
            sha256(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256(FOX),
            "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
        );
        // more than one block of 64 bytes
        assert_eq!(
            sha256(&[b'a'; 1000]),
            "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
        );
    }

    #[test]
    fn digests_of_pieces_are_those_of_the_whole() {
        let mut digests = Digests::default();
        for piece in FOX.chunks(7) {
            digests.update(piece);
        }
        assert_eq!(digests.finish(), (sha256(FOX), md5(FOX)));
        assert_eq!(Digests::default().finish(), (sha256(b""), md5(b"")));
    }

    #[test]
    fn md5_is_that_of_md5sum() {
        assert_eq!(md5(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5(FOX), "9e107d9d372bb6826bd81d3542a419d6");
        // more than one block of 64 bytes
        assert_eq!(md5(&[b'a'; 1000]), "cabe45dcc9ae5b66ba86600cca6b8ba8");
    }
}
