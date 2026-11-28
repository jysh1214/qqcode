//! Carries a file from a screen to a camera.
//!
//! `qqcode-show` writes a web page that shows the file. The page cuts it into packets of which any that
//! are enough make the file ([`fountain`]), puts each into a frame ([`packet`]) and shows that as a matrix
//! of coloured tiles ([`tiles`]) with check bytes for what a camera misreads ([`rs`]). A file is named in
//! the frames by its length and its CRC-32 ([`crc`]).
//!
//! A file that the page shows may be far longer than the memory: [`stored`] puts it together on disk.
//! [`checks`] are the checksums of a file section by section, which go ahead of it as a text and against
//! which it is held as it arrives.
//!
//! `qqcode` looks for the matrix in the frames of a camera ([`scan`], [`sight`]), and [`Pipeline`] takes
//! in what is found until the file is whole. Either program ends by printing the SHA-256 and the MD5 of
//! the file ([`digest`]).

pub mod checks;
pub mod crc;
pub mod digest;
pub mod fountain;
pub mod packet;
pub mod reader;
pub mod rs;
pub mod scan;
pub mod sight;
pub mod stored;
pub mod tiles;

pub use reader::{Kind, Outcome, Pipeline, Sighting, Survey};
