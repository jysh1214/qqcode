//! End to end on synthetic frames: codes as `qqcode-show` makes them, seen at an angle, and the program
//! `qqcode` on pictures of them.
#![cfg(feature = "reader")]
// a file may well be missing in one part only
#![allow(clippy::single_range_in_vec_init)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use opencv::core::{self, CV_8UC3, Mat, Point2f, Rect, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};

use qqcode::code::{self, Level};
use qqcode::protocol::{FileId, Message};
use qqcode::{Kind, Outcome, Pipeline};

const VERSION: usize = 12;
const LEVEL: Level = Level::M;
/// Bytes of the file in a code: what version 12 holds at level M less the 15 that say where they go.
const PIECE: usize = 272;

/// Bytes of every value, in no pattern.
fn bytes(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 32) as u8
    };
    (0..count).map(|_| next()).collect()
}

/// The messages of a show of `file`: the start screen, the pieces, the end screen.
fn show(file: &[u8], name: &str) -> Vec<Message> {
    let id = FileId::of(file).unwrap();
    let start = Message::Start {
        file: id,
        name: name.to_owned(),
    };
    let pieces = file
        .chunks(PIECE)
        .enumerate()
        .map(|(index, bytes)| Message::Piece {
            file: id,
            offset: (index * PIECE) as u32,
            bytes: bytes.to_vec(),
        });
    let mut messages = vec![start];
    messages.extend(pieces);
    messages.push(Message::End { file: id });
    messages
}

/// A white screen of 800x600 with the code that holds `content` in its middle, 6 pixels a module.
fn screen(content: &[u8]) -> Result<Mat> {
    let code = code::encode(content, VERSION, LEVEL)?;
    let mut screen = Mat::new_size_with_default(Size::new(800, 600), CV_8UC3, Scalar::all(255.0))?;
    let side = code.side() as i32;
    let (left, top) = ((800 - 6 * side) / 2, (600 - 6 * side) / 2);
    for row in 0..side {
        for column in 0..side {
            if code.is_dark(row as usize, column as usize) {
                let module = Rect::new(left + 6 * column, top + 6 * row, 6, 6);
                let black = Scalar::all(0.0);
                imgproc::rectangle(&mut screen, module, black, imgproc::FILLED, imgproc::LINE_8, 0)?;
            }
        }
    }
    Ok(screen)
}

/// A camera frame that shows `screen` off-centre and in perspective in a dim room.
fn photograph(screen: &Mat) -> Result<Mat> {
    let size = screen.size()?;
    let (w, h) = ((size.width - 1) as f32, (size.height - 1) as f32);
    let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
    let seen =
        [(410.0, 130.0), (1550.0, 90.0), (1640.0, 960.0), (350.0, 900.0)].map(|(x, y)| Point2f::new(x, y));
    let transform = imgproc::get_perspective_transform_def(
        &Vector::<Point2f>::from_slice(&flat),
        &Vector::<Point2f>::from_slice(&seen),
    )?;
    let mut frame = Mat::default();
    imgproc::warp_perspective(
        screen,
        &mut frame,
        &transform,
        Size::new(1920, 1080),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(60.0),
    )?;
    Ok(frame)
}

fn frame(message: &Message) -> Result<Mat> {
    photograph(&screen(&message.encode())?)
}

#[test]
fn reads_a_show_from_its_start_screen_to_the_file() -> Result<()> {
    let file = bytes(1000, 1);
    let mut pipeline = Pipeline::new();
    let mut outcomes = Vec::new();
    for message in show(&file, "notes.bin") {
        outcomes.push(pipeline.process(&frame(&message)?)?);
        if outcomes.len() == 1 {
            assert_eq!(pipeline.name(), Some("notes.bin"));
            assert_eq!(pipeline.file(), FileId::of(&file));
        }
    }
    let piece = |new| Outcome::Piece { new, skipped: 0 };
    // the last piece completes the file, and the end screen after it has nothing to add
    assert_eq!(
        outcomes,
        [
            Outcome::Ready,
            piece(272),
            piece(272),
            piece(272),
            Outcome::Complete,
            Outcome::Complete
        ]
    );
    assert_eq!(pipeline.take_file(), Some(file));
    assert_eq!(pipeline.take_file(), None);
    assert_eq!((pipeline.files(), pipeline.received()), (1, Some(1000)));
    Ok(())
}

#[test]
fn code_is_read_whichever_way_up_the_camera_is() -> Result<()> {
    let file = bytes(200, 2);
    let message = &show(&file, "")[1];
    let upright = frame(message)?;
    // a camera on its side takes frames that are higher than wide
    let turns = [
        core::ROTATE_90_CLOCKWISE,
        core::ROTATE_180,
        core::ROTATE_90_COUNTERCLOCKWISE,
    ];
    for turn in turns {
        let mut turned = Mat::default();
        core::rotate(&upright, &mut turned, turn)?;
        let mut pipeline = Pipeline::new();
        assert_eq!(pipeline.process(&turned)?, Outcome::Complete, "turn {turn}");
        assert_eq!(pipeline.take_file().as_ref(), Some(&file));
        // the top-left corner of the code is where the turn takes it
        let corner = pipeline.last_survey().codes[0].outline[0];
        let size = turned.size()?;
        let (right, low) = (
            corner.x > size.width as f32 / 2.0,
            corner.y > size.height as f32 / 2.0,
        );
        let expected = match turn {
            core::ROTATE_90_CLOCKWISE => (true, false),
            core::ROTATE_180 => (true, true),
            _ => (false, true),
        };
        assert_eq!((right, low), expected, "turn {turn}");
    }
    Ok(())
}

#[test]
fn pieces_that_went_by_unread_are_counted_and_come_in_later() -> Result<()> {
    let file = bytes(1500, 3);
    let messages = show(&file, "notes.bin");
    let mut pipeline = Pipeline::new();
    let mut take = |index: usize| pipeline.process(&frame(&messages[index])?);
    assert_eq!(take(0)?, Outcome::Ready);
    // the first and the fourth piece are missed
    assert_eq!(
        take(2)?,
        Outcome::Piece {
            new: 272,
            skipped: 272
        }
    );
    assert_eq!(take(3)?, Outcome::Piece { new: 272, skipped: 0 });
    assert_eq!(
        take(5)?,
        Outcome::Piece {
            new: 272,
            skipped: 272
        }
    );
    assert_eq!(take(6)?, Outcome::Piece { new: 140, skipped: 0 });
    assert_eq!(take(7)?, Outcome::Done);
    let assembly = pipeline.assembly().expect("a file that is begun");
    assert_eq!(assembly.missing(), [0..272, 816..1088]);
    assert_eq!(assembly.received(), 956);

    // the show again from its first code: what was read before brings nothing, the rest the file
    let mut take = |index: usize| pipeline.process(&frame(&messages[index])?);
    assert_eq!(take(1)?, Outcome::Piece { new: 272, skipped: 0 });
    assert_eq!(take(2)?, Outcome::Piece { new: 0, skipped: 0 });
    assert_eq!(take(3)?, Outcome::Piece { new: 0, skipped: 0 });
    assert_eq!(take(4)?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(file));
    Ok(())
}

#[test]
fn frame_between_two_codes_is_no_piece_of_its_own() -> Result<()> {
    let file = bytes(600, 4);
    let messages = show(&file, "");
    let (first, second) = (frame(&messages[1])?, frame(&messages[2])?);
    // what a camera sees while the screen changes: both at once
    let mut between = Mat::default();
    core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;

    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&first)?, Outcome::Piece { new: 272, skipped: 0 });
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    assert_eq!(pipeline.last_survey().codes[0].kind, Kind::Unread);
    assert_eq!(
        pipeline.process(&second)?,
        Outcome::Piece { new: 272, skipped: 0 }
    );
    assert_eq!(pipeline.process(&frame(&messages[3])?)?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(file));
    Ok(())
}

#[test]
fn other_codes_and_no_codes_are_told_apart() -> Result<()> {
    let mut pipeline = Pipeline::new();
    let nothing = Mat::new_size_with_default(Size::new(1920, 1080), CV_8UC3, Scalar::all(128.0))?;
    assert_eq!(pipeline.process(&nothing)?, Outcome::NoCode);
    assert!(pipeline.last_survey().codes.is_empty());
    assert_eq!(pipeline.received(), None);

    let link = photograph(&screen(b"https://example.org/a-link-and-not-a-file")?)?;
    assert_eq!(pipeline.process(&link)?, Outcome::Other);
    let survey = pipeline.last_survey();
    assert_eq!((survey.codes.len(), survey.codes[0].kind), (1, Kind::Other));
    assert_eq!(survey.codes[0].modules, code::side(VERSION));
    assert_eq!((pipeline.files(), pipeline.file()), (0, None));

    // in one channel as well as in three
    let mut grey = Mat::default();
    imgproc::cvt_color_def(&link, &mut grey, imgproc::COLOR_BGR2GRAY)?;
    assert_eq!(pipeline.process(&grey)?, Outcome::Other);
    Ok(())
}

#[test]
fn codes_of_another_file_start_over() -> Result<()> {
    let (first, second) = (bytes(600, 5), bytes(500, 6));
    let (first_show, second_show) = (show(&first, "first"), show(&second, "second"));
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&frame(&first_show[0])?)?, Outcome::Ready);
    assert_eq!(
        pipeline.process(&frame(&first_show[1])?)?,
        Outcome::Piece { new: 272, skipped: 0 }
    );
    assert_eq!((pipeline.files(), pipeline.name()), (1, Some("first")));

    // without a start screen: the second piece of the other file
    assert_eq!(
        pipeline.process(&frame(&second_show[2])?)?,
        Outcome::Piece { new: 228, skipped: 0 }
    );
    assert_eq!((pipeline.files(), pipeline.name()), (2, None));
    assert_eq!(pipeline.file(), FileId::of(&second));
    assert_eq!(pipeline.assembly().unwrap().missing(), [0..272]);
    assert_eq!(pipeline.process(&frame(&second_show[1])?)?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(second));
    Ok(())
}

#[test]
fn file_whose_checksum_is_wrong_is_collected_again() -> Result<()> {
    let file = bytes(500, 7);
    let messages = show(&file, "");
    // a code that was misread in a way its error correction did not notice: a piece with a wrong byte
    let Message::Piece {
        file: id,
        offset,
        mut bytes,
    } = messages[2].clone()
    else {
        panic!("a piece");
    };
    bytes[100] ^= 1;
    let misread = Message::Piece {
        file: id,
        offset,
        bytes,
    };

    let mut pipeline = Pipeline::new();
    assert_eq!(
        pipeline.process(&frame(&messages[1])?)?,
        Outcome::Piece { new: 272, skipped: 0 }
    );
    assert_eq!(pipeline.process(&frame(&misread)?)?, Outcome::Corrupt);
    assert_eq!(pipeline.take_file(), None);
    assert!(pipeline.assembly().is_none());
    assert_eq!(pipeline.received(), Some(0));
    // it is the same file still, and its pieces count from the beginning
    assert_eq!(
        pipeline.process(&frame(&messages[1])?)?,
        Outcome::Piece { new: 272, skipped: 0 }
    );
    assert_eq!(pipeline.process(&frame(&messages[2])?)?, Outcome::Complete);
    assert_eq!((pipeline.files(), pipeline.take_file()), (1, Some(file)));
    Ok(())
}

#[test]
fn empty_file_is_whole_at_its_start_screen() -> Result<()> {
    let messages = show(&[], "empty");
    assert_eq!(messages.len(), 2);
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&frame(&messages[0])?)?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(Vec::new()));
    Ok(())
}

/// A directory of its own for the pictures of one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qqcode-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Saves the frames of `messages` as pictures and returns the name that `qqcode` reads them by.
fn pictures(dir: &Path, messages: &[Message]) -> Result<String> {
    for (index, message) in messages.iter().enumerate() {
        let path = dir.join(format!("frame_{index:03}.png"));
        assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &frame(message)?)?);
    }
    Ok(dir.join("frame_%03d.png").to_str().unwrap().to_owned())
}

#[test]
fn program_writes_the_file_to_the_standard_output() -> Result<()> {
    let file = bytes(700, 8);
    let dir = scratch("whole");
    let input = pictures(&dir, &show(&file, "notes.bin"))?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{said}");
    assert_eq!(run.stdout, file);
    assert!(
        said.contains("frame 1: start screen in view: notes.bin, 700 bytes, waiting for the first code"),
        "{said}"
    );
    assert!(said.contains("frame 3: 544 of 700 bytes"), "{said}");
    // it ends with the piece that completes the file, before the end screen
    assert!(
        said.ends_with(
            "frame 4: the file is whole\n4 frames read, 700 bytes written to the standard output\n"
        ),
        "{said}"
    );

    // and to a file that is named, with nothing on the standard output
    let output = dir.join("read.bin");
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    assert!(run.status.success() && run.stdout.is_empty());
    assert_eq!(fs::read(&output)?, file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_writes_nothing_of_a_file_that_is_not_whole() -> Result<()> {
    let file = bytes(1500, 9);
    let mut messages = show(&file, "notes.bin");
    // the third piece is never seen
    messages.remove(3);
    let dir = scratch("holes");
    let input = pictures(&dir, &messages)?;
    let output = dir.join("read.bin");
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(!output.exists());
    assert!(said.contains("frame 4: 272 bytes went by unread"), "{said}");
    assert!(
        said.contains(
            "frame 7: end screen in view, 272 of 1500 bytes are missing: 544 to 815; g in qqcode-show shows \
             the codes again"
        ),
        "{said}"
    );
    assert!(
        said.contains("7 frames read, nothing written\n")
            && said.contains("the file is not whole: 272 of 1500 bytes are missing: 544 to 815"),
        "{said}"
    );

    // pictures without a code of the show at all
    let link = dir.join("link.png");
    let picture = photograph(&screen(b"https://example.org")?)?;
    assert!(imgcodecs::imwrite_def(link.to_str().unwrap(), &picture)?);
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&link).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty());
    assert!(said.contains("no frame showed a code of qqcode-show"), "{said}");
    fs::remove_dir_all(&dir)?;
    Ok(())
}
