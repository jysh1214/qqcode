//! Carries a file from a screen to a camera as a sequence of QR codes.
//!
//! `qqcode-show` cuts the file into pieces and puts each into a code ([`code`]) as a message that says
//! which file it belongs to and where in it its bytes go ([`protocol`]). `qqcode` looks for the codes in
//! the frames of a camera ([`scan`]) and puts the pieces together until the file is whole ([`assembly`]);
//! [`Pipeline`] chains the two.

pub mod assembly;
pub mod code;
pub mod protocol;
pub mod reader;
pub mod scan;

pub use reader::{Kind, Outcome, Pipeline, Sighting, Survey};
