//! Carries a file from a screen to a camera.
//!
//! `qqcode-show` shows the file in one of two ways. On a terminal it cuts it into pieces and puts each
//! into a QR code ([`code`]) as a message that says which file it belongs to and where in it its bytes go
//! ([`protocol`]); `qqcode` puts the pieces together ([`assembly`]). On a web page it cuts it into packets
//! of which any that are enough make the file ([`fountain`]), puts each into a frame ([`packet`]) and
//! shows that as a matrix of coloured tiles ([`tiles`]) with check bytes for what a camera misreads
//! ([`rs`]).
//!
//! `qqcode` looks for either in the frames of a camera ([`scan`], and [`sight`] for the matrix), and
//! [`Pipeline`] takes in what is found until the file is whole. Either program ends by printing the
//! SHA-256 and the MD5 of the file ([`digest`]).

pub mod assembly;
pub mod code;
pub mod digest;
pub mod fountain;
pub mod packet;
pub mod protocol;
pub mod reader;
pub mod rs;
pub mod scan;
pub mod sight;
pub mod tiles;

pub use reader::{Kind, Outcome, Pipeline, Sighting, Survey};
