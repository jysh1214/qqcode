//! End to end on synthetic frames: codes as `qqcode-show` makes them, seen at an angle, and the program
//! `qqcode` on pictures of them.
// a file may well be missing in one part only
#![allow(clippy::single_range_in_vec_init)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use opencv::core::{self, CV_8UC3, Mat, Point2f, Rect, Scalar, Size, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};

use qqcode::checks::{Checks, Note};
use qqcode::code::{self, Level};
use qqcode::fountain::{Encoder, Schedule};
use qqcode::packet::{Frame, OVERHEAD, Origin};
use qqcode::protocol::{FileId, Message};
use qqcode::scan::{Content, Found};
use qqcode::tiles::{self, Grid};
use qqcode::{Kind, Outcome, Pipeline, digest};

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
            assert_eq!(pipeline.file(), Some(Origin::of(&file)));
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
    assert_eq!(pipeline.file(), Some(Origin::of(&second)));
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

/// The matrix of the web page that these tests show: 128 by 72 tiles, 5265 bytes of a file in a frame.
const GRID: Grid = Grid {
    columns: 128,
    rows: 72,
};

/// What the web page shows of `file`: its start screen, and after it the packets of the file without end.
fn page_frame(file: &[u8], frame: Option<u64>) -> Frame {
    let origin = Origin::of(file);
    let encoder = Encoder::new(file, origin.checksum, GRID.capacity() - OVERHEAD);
    match frame {
        None => Frame::Start {
            origin,
            name: "notes.bin".to_owned(),
        },
        Some(frame) => {
            let (block, sequence) = Schedule::new(encoder.plan().clone(), 25).turn(frame);
            Frame::Packet {
                origin,
                block,
                sequence,
                bytes: encoder.packet(block, sequence),
            }
        }
    }
}

/// A camera frame that shows the page with `frame` on it: the matrix on a light page, somewhat smaller
/// than it is, a little turned and out of focus.
fn page_seen(frame: &Frame) -> Result<Mat> {
    let picture = tiles::render(GRID, &tiles::pack(GRID, &frame.encode(GRID.capacity())));
    let size = Size::new(GRID.width() as i32, GRID.height() as i32);
    let mut matrix = Mat::new_size_with_default(size, CV_8UC3, Scalar::all(0.0))?;
    for (to, from) in matrix
        .data_bytes_mut()?
        .chunks_exact_mut(3)
        .zip(picture.chunks_exact(3))
    {
        to.copy_from_slice(&[from[2], from[1], from[0]]);
    }
    let middle = Point2f::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
    let mut transform = imgproc::get_rotation_matrix_2d(middle, 1.5, 0.9)?;
    *transform.at_2d_mut::<f64>(0, 2)? += 960.0 - middle.x as f64;
    *transform.at_2d_mut::<f64>(1, 2)? += 540.0 - middle.y as f64;
    let mut seen = Mat::default();
    imgproc::warp_affine(
        &matrix,
        &mut seen,
        &transform,
        Size::new(1920, 1080),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(240.0),
    )?;
    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&seen, &mut blurred, Size::new(0, 0), 1.0)?;
    Ok(blurred)
}

/// Shows the reader a frame of the page of `file`.
fn show_page(pipeline: &mut Pipeline, file: &[u8], frame: Option<u64>) -> Result<Outcome> {
    pipeline.process(&page_seen(&page_frame(file, frame))?)
}

#[test]
fn reads_a_web_page_from_its_start_screen_to_the_file() -> Result<()> {
    let file = bytes(20_000, 11);
    let mut pipeline = Pipeline::new();
    assert_eq!(show_page(&mut pipeline, &file, None)?, Outcome::Ready);
    assert_eq!(
        show_page(&mut pipeline, &file, Some(0))?,
        Outcome::Packet { new: true }
    );
    // the same frame again tells nothing new
    assert_eq!(
        show_page(&mut pipeline, &file, Some(0))?,
        Outcome::Packet { new: false }
    );
    assert_eq!(
        show_page(&mut pipeline, &file, Some(1))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(
        show_page(&mut pipeline, &file, Some(2))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(pipeline.packets(), Some((3, 4)));
    assert_eq!(pipeline.received(), Some(3 * 5265));
    assert_eq!(
        (pipeline.name(), pipeline.file()),
        (Some("notes.bin"), Some(Origin::of(&file)))
    );
    assert!(pipeline.assembly().is_none());
    let seen = pipeline.last_survey().codes[0].clone();
    assert_eq!(
        (seen.kind, seen.modules, seen.is_matrix()),
        (Kind::Piece, 128, true)
    );
    // nine tenths of the 9 pixels that a tile has on the page
    assert!((7.9..8.4).contains(&seen.module_size), "{}", seen.module_size);
    assert_eq!(show_page(&mut pipeline, &file, Some(3))?, Outcome::Complete);
    assert_eq!(pipeline.take_file().as_ref(), Some(&file));
    // what the page shows after that is of a file that is whole
    assert_eq!(show_page(&mut pipeline, &file, Some(9))?, Outcome::Complete);
    assert_eq!(pipeline.received(), Some(20_000));
    Ok(())
}

#[test]
fn file_of_a_web_page_is_put_together_on_disk() -> Result<()> {
    let file = bytes(40_000, 15);
    let dir = scratch("stored");
    let (part, sums) = (dir.join("read.part"), dir.join("read.part.sums"));
    // a pipeline for files that are too long to keep
    let mut pipeline = Pipeline::storing(&part, &sums);
    assert_eq!(show_page(&mut pipeline, &file, None)?, Outcome::Ready);
    // of the eight frames that are the file, the camera misses the third and the sixth
    for frame in [0, 1, 3, 4, 6, 7] {
        assert_eq!(
            show_page(&mut pipeline, &file, Some(frame))?,
            Outcome::Packet { new: true }
        );
    }
    // what arrived is in its place already, and what did not is a hole
    let so_far = fs::read(&part)?;
    assert_eq!(
        (&so_far[..2 * 5265], &so_far[3 * 5265..5 * 5265]),
        (&file[..2 * 5265], &file[3 * 5265..5 * 5265])
    );
    assert!(so_far[2 * 5265..3 * 5265].iter().all(|byte| *byte == 0));
    assert_eq!(pipeline.packets(), Some((6, 8)));
    // sums of the packets follow, and two of them fill the holes, or three
    let mut outcome = Outcome::NoCode;
    for frame in 8..12 {
        outcome = show_page(&mut pipeline, &file, Some(frame))?;
        if outcome == Outcome::Complete {
            break;
        }
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(fs::read(&part)?, file);
    // the file is on disk and nowhere else
    assert_eq!(pipeline.take_file(), None);
    assert_eq!(pipeline.received(), Some(40_000));
    assert_eq!(pipeline.file(), Some(Origin::of(&file)));
    // another file comes on show, one without a byte, which is whole at its start screen: the file on
    // disk is made anew for it
    assert_eq!(show_page(&mut pipeline, &[], None)?, Outcome::Complete);
    assert_eq!(pipeline.file(), Some(Origin::of(&[])));
    assert_eq!(fs::read(&part)?, b"");
    assert_eq!(pipeline.take_file(), None);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// What a scanner makes of a frame of the page that a camera sees without a fault.
fn found(frame: Frame) -> Vec<Found> {
    vec![Found {
        outline: [Point2f::new(0.0, 0.0); 4],
        modules: GRID.columns,
        content: Content::Matrix {
            frame: Some(frame),
            wrong: 0.0,
        },
    }]
}

#[test]
fn file_on_disk_is_held_against_its_checksums_and_a_misread_packet_is_left_out() -> Result<()> {
    // eight packets of 5265 bytes, in eight sections of 5000
    let file = bytes(40_000, 16);
    let dir = scratch("checked");
    let (part, sums) = (dir.join("read.part"), dir.join("read.part.sums"));
    let checks = Checks::in_sections("notes.bin", &file, 5000);
    let mut pipeline = Pipeline::storing(&part, &sums).checked(checks);
    assert_eq!(show_page(&mut pipeline, &file, None)?, Outcome::Ready);
    assert_eq!(pipeline.sound(), None);
    // Every frame arrives, and of the third the camera misreads four bytes, which the check of a frame
    // lets through once in a long while: here it is made up.
    for frame in 0..8 {
        let mut frame = page_frame(&file, Some(frame));
        if let Frame::Packet {
            sequence: 2, bytes, ..
        } = &mut frame
        {
            bytes[10..14].iter_mut().for_each(|byte| *byte ^= 0xff);
        }
        assert_eq!(pipeline.take(found(frame))?, Outcome::Packet { new: true });
    }
    // The file is all there, and the section from byte 10000 on, which has those four bytes, is not
    // what it should be: the file is not whole yet.
    assert_eq!(pipeline.sound(), Some((7, 8)));
    assert_eq!(
        pipeline.take_notes(),
        [Note::Doubted {
            section: 2,
            from: 10_000
        }]
    );
    assert_ne!(fs::read(&part)?, file);
    // the sums that the page shows next tell which packet it was
    let mut outcome = Outcome::NoCode;
    for frame in 8..24 {
        outcome = show_page(&mut pipeline, &file, Some(frame))?;
        if outcome == Outcome::Complete {
            break;
        }
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(
        pipeline.take_notes(),
        [Note::Mended {
            section: 2,
            from: 10_000
        }]
    );
    assert_eq!(fs::read(&part)?, file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn no_other_file_is_taken_than_the_one_that_the_checksums_are_of() -> Result<()> {
    let (file, other) = (bytes(20_000, 17), bytes(20_000, 18));
    let mut pipeline = Pipeline::new().checked(Checks::of("notes.bin", &file));
    // a QR code that is not of qqcode-show is none of any file
    let other_code = photograph(&screen(b"https://example.org")?)?;
    assert_eq!(pipeline.process(&other_code)?, Outcome::Other);
    // the web page of another file, and its QR codes
    let error = show_page(&mut pipeline, &other, None).unwrap_err().to_string();
    let expected = format!(
        "the file on show has 20000 bytes and the CRC-32 {:08x}, and the checksums are those of \"notes.bin\", \
         a file of 20000 bytes with the CRC-32 {:08x}",
        Origin::of(&other).checksum,
        Origin::of(&file).checksum
    );
    assert_eq!(error, expected);
    let error = pipeline.process(&frame(&show(&other, "other")[0])?).unwrap_err();
    assert_eq!(error.to_string(), expected);
    // the file itself is taken, and whole when it has arrived
    assert_eq!(show_page(&mut pipeline, &file, None)?, Outcome::Ready);
    for frame in 0..4 {
        show_page(&mut pipeline, &file, Some(frame))?;
    }
    assert_eq!(pipeline.take_file(), Some(file));
    assert_eq!(pipeline.take_notes(), []);
    Ok(())
}

#[test]
fn file_in_memory_is_held_against_its_checksums_when_it_is_whole() -> Result<()> {
    // The checksums of the first two sections change places: the file that arrives is the one they
    // name, by its length and its CRC-32, and two of its sections are not what they say.
    let file = bytes(20_000, 19);
    let text = Checks::in_sections("notes.bin", &file, 8000).to_string();
    let mut lines = text.lines().map(str::to_owned).collect::<Vec<_>>();
    let (first, second) = (lines[6][..64].to_owned(), lines[7][..64].to_owned());
    lines[6].replace_range(..64, &second);
    lines[7].replace_range(..64, &first);
    let checks = Checks::parse(&lines.join("\n")).unwrap();
    // a web page
    let mut pipeline = Pipeline::new().checked(checks.clone());
    let mut outcome = Outcome::NoCode;
    for frame in 0..4 {
        outcome = show_page(&mut pipeline, &file, Some(frame))?;
    }
    assert_eq!(outcome, Outcome::Corrupt);
    assert_eq!(pipeline.take_file(), None);
    assert_eq!(pipeline.take_notes(), [Note::Wrong { sections: 2, from: 0 }]);
    // QR codes
    let mut pipeline = Pipeline::new().checked(checks);
    let messages = show(&file[..700], "notes.bin");
    assert!(
        pipeline.process(&frame(&messages[1])?).is_err(),
        "700 bytes are another file"
    );
    let mut pipeline = Pipeline::new().checked(Checks::in_sections("notes.bin", &file[..700], 300));
    let mut outcome = Outcome::NoCode;
    for message in &messages {
        outcome = pipeline.process(&frame(message)?)?;
        if outcome == Outcome::Complete {
            break;
        }
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(pipeline.take_file().as_deref(), Some(&file[..700]));
    assert_eq!(pipeline.take_notes(), []);
    Ok(())
}

#[test]
fn web_page_is_read_from_any_frames_that_are_enough() -> Result<()> {
    let file = bytes(30_000, 12);
    let mut pipeline = Pipeline::new();
    // none of the six frames that are the file itself: sums of them only, from wherever the page is
    let mut frames = 0;
    for frame in 1000.. {
        frames += 1;
        match pipeline.process(&page_seen(&page_frame(&file, Some(frame)))?)? {
            Outcome::Complete => break,
            outcome => assert!(matches!(outcome, Outcome::Packet { .. }), "{outcome:?}"),
        }
        assert!(frames < 16, "the file never came together");
    }
    // Six frames that tell something new. Of a file this short most sums do and some don't: there are
    // only 64 sums of six packets.
    assert!((6..=12).contains(&frames), "{frames}");
    assert_eq!(pipeline.take_file(), Some(file));
    Ok(())
}

#[test]
fn frame_between_two_frames_of_a_web_page_is_not_read() -> Result<()> {
    let file = bytes(20_000, 13);
    let (first, second) = (
        page_seen(&page_frame(&file, Some(0)))?,
        page_seen(&page_frame(&file, Some(1)))?,
    );
    let mut between = Mat::default();
    core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&first)?, Outcome::Packet { new: true });
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    let seen = &pipeline.last_survey().codes[0];
    assert_eq!((seen.kind, seen.is_matrix()), (Kind::Unread, true));
    assert_eq!(pipeline.packets(), Some((1, 4)));
    assert_eq!(pipeline.process(&second)?, Outcome::Packet { new: true });
    Ok(())
}

#[test]
fn program_reads_the_pictures_of_a_web_page() -> Result<()> {
    let file = bytes(50_000, 14);
    let dir = scratch("page");
    // the start screen and twelve frames, of which ten make the file
    for (index, frame) in [None].into_iter().chain((0..12).map(Some)).enumerate() {
        let path = dir.join(format!("frame_{index:03}.png"));
        let seen = page_seen(&page_frame(&file, frame))?;
        assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &seen)?);
    }
    let input = dir.join("frame_%03d.png");
    // by two threads, with room for three frames to wait: a file waits for the room it needs
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .args(["--threads", "2", "--buffer", "3"])
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{said}");
    assert_eq!(run.stdout, file);
    assert!(
        said.contains("frame 1: start screen in view: notes.bin, 50000 bytes, waiting for the first code"),
        "{said}"
    );
    let end = format!(
        "frame 11: the file is whole\n11 frames read, 50000 bytes written to the standard output\nSHA-256 {}\nMD5 {}\n",
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert!(said.ends_with(&end), "{said}");

    // To a file that is named it is written as it arrives, under another name until it is whole.
    let (output, part) = (dir.join("read.bin"), dir.join("read.bin.part"));
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success() && run.stdout.is_empty(), "{said}");
    assert_eq!(fs::read(&output)?, file);
    assert!(!part.exists() && !dir.join("read.bin.part.sums").exists());
    let end = format!(
        "11 frames read, 50000 bytes written to {}\nSHA-256 {}\nMD5 {}\n",
        output.display(),
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert!(said.ends_with(&end), "{said}");
    fs::remove_file(&output)?;

    // Held against the checksums that qqcode-show writes next to its page, which the program reads
    // before anything else.
    let list = dir.join("notes.bin.checksums.txt");
    let text = Checks::of("notes.bin", &file).to_string();
    fs::write(&list, &text)?;
    let checked = |list: &Path, to: Option<&Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_qqcode"));
        command.arg(&input).arg("--check").arg(list);
        if let Some(to) = to {
            command.arg("-o").arg(to);
        }
        command.output()
    };
    let run = checked(&list, Some(&output))?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{said}");
    assert_eq!(fs::read(&output)?, file);
    let begin = format!(
        "{}: the checksums of notes.bin, 50000 bytes in 1 section; the SHA-256 of this list is {}\n",
        list.display(),
        digest::sha256(text.as_bytes())
    );
    assert!(said.starts_with(&begin), "{said}");
    let end = format!(
        "11 frames read, 50000 bytes written to {}\nheld against {}: its one section has its checksum\n\
         SHA-256 {}\nMD5 {}\n",
        output.display(),
        list.display(),
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert!(said.ends_with(&end), "{said}");
    fs::remove_file(&output)?;
    // the standard output gets the file as before
    let run = checked(&list, None)?;
    assert!(run.status.success() && run.stdout == file);
    // The checksums of another file: nothing of the one on show is taken.
    let another = dir.join("another.checksums.txt");
    fs::write(
        &another,
        Checks::of("another.bin", &bytes(50_000, 21)).to_string(),
    )?;
    let run = checked(&another, Some(&output))?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(
        !run.status.success() && !output.exists() && !part.exists(),
        "{said}"
    );
    assert!(
        said.contains("frame 1: the file on show has 50000 bytes and the CRC-32")
            && said.contains("and the checksums are those of \"another.bin\", a file of 50000 bytes"),
        "{said}"
    );
    // A list that lost a line, and one that is not there.
    fs::write(&another, text.replace("length 50000\n", ""))?;
    let run = checked(&another, Some(&output))?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "{said}");
    assert!(
        said.contains(&format!(
            "{} is no list of checksums: line 3 is to begin with \"length\"",
            another.display()
        )),
        "{said}"
    );
    let run = checked(&dir.join("none.txt"), Some(&output))?;
    assert!(String::from_utf8_lossy(&run.stderr).contains("can't read"));
    fs::remove_file(&list)?;
    fs::remove_file(&another)?;

    // without the last frames the file is not whole, and the program says how far it came
    for index in [9, 10, 11, 12] {
        fs::remove_file(dir.join(format!("frame_{index:03}.png")))?;
    }
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(
        said.contains("the file is not whole: 8 of the 10 packets it takes have arrived"),
        "{said}"
    );
    // and leaves nothing behind where it was to be written
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    assert!(!run.status.success());
    assert!(!output.exists() && !part.exists() && !dir.join("read.bin.part.sums").exists());
    fs::remove_dir_all(&dir)?;
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
    // it ends with the piece that completes the file, before the end screen, and with the SHA-256 and
    // the MD5 of what it wrote
    let end = format!(
        "frame 4: the file is whole\n4 frames read, 700 bytes written to the standard output\nSHA-256 {}\nMD5 {}\n",
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert!(said.ends_with(&end), "{said}");

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

#[test]
fn page_that_goes_on_with_another_matrix_begins_the_file_anew() -> Result<()> {
    let file = bytes(40_000, 23);
    let origin = Origin::of(&file);
    let mut pipeline = Pipeline::new();
    // four of the eight packets that the file is in the matrix of these tests
    for frame in 0..4 {
        let outcome = pipeline.take(found(page_frame(&file, Some(frame))))?;
        assert_eq!(outcome, Outcome::Packet { new: true });
    }
    assert_eq!(pipeline.packets(), Some((4, 8)));
    // The page is written anew with a larger matrix: packets of 10077 bytes, of which the file is
    // four. What was held is of no use for them, and they are the file.
    let encoder = Encoder::new(&file, origin.checksum, 10_077);
    let packet = |sequence: u32| Frame::Packet {
        origin,
        block: 0,
        sequence,
        bytes: encoder.packet(0, sequence),
    };
    assert_eq!(pipeline.take(found(packet(0)))?, Outcome::Packet { new: true });
    assert_eq!(pipeline.packets(), Some((1, 4)));
    for sequence in 1..3 {
        pipeline.take(found(packet(sequence)))?;
    }
    assert_eq!(pipeline.take(found(packet(3)))?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(file));
    Ok(())
}

#[test]
fn blocks_of_a_page_that_are_solved_out_of_order_keep_their_places() -> Result<()> {
    // two blocks of 300 packets of 100 bytes, kept in memory; the second is whole before the first
    let file = bytes(60_000, 24);
    let origin = Origin::of(&file);
    let encoder = Encoder::new(&file, origin.checksum, 100);
    let mut pipeline = Pipeline::new();
    let mut outcome = Outcome::NoCode;
    for block in [1, 0] {
        for sequence in 0..300 {
            let bytes = encoder.packet(block, sequence);
            outcome = pipeline.take(found(Frame::Packet {
                origin,
                block,
                sequence,
                bytes,
            }))?;
        }
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(file));
    Ok(())
}

#[test]
fn start_screen_without_a_name_gives_the_file_none() -> Result<()> {
    let file = bytes(700, 27);
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&frame(&show(&file, "")[0])?)?, Outcome::Ready);
    assert_eq!(pipeline.name(), None);
    // and one with a name does
    assert_eq!(
        pipeline.process(&frame(&show(&file, "notes.bin")[0])?)?,
        Outcome::Ready
    );
    assert_eq!(pipeline.name(), Some("notes.bin"));
    Ok(())
}

#[test]
fn file_of_one_byte_waits_for_its_packet() -> Result<()> {
    // the shortest file that is not whole at its start screen
    let file = [0x5a];
    let mut pipeline = Pipeline::new();
    assert_eq!(show_page(&mut pipeline, &file, None)?, Outcome::Ready);
    assert_eq!(pipeline.take_file(), None);
    assert_eq!(show_page(&mut pipeline, &file, Some(0))?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(file.to_vec()));
    Ok(())
}

#[test]
fn of_two_shows_in_a_frame_the_first_is_the_one_that_counts() -> Result<()> {
    // a frame in which the scanner found two matrices: the start screen of the page, and a packet
    let file = bytes(20_000, 22);
    let mut both = found(page_frame(&file, None));
    both.extend(found(page_frame(&file, Some(0))));
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.take(both)?, Outcome::Ready);
    // nothing of the packet was taken
    assert_eq!((pipeline.packets(), pipeline.received()), (None, Some(0)));
    assert_eq!(pipeline.last_survey().codes.len(), 2);
    Ok(())
}

#[test]
fn frame_that_is_not_of_bytes_is_an_error() -> Result<()> {
    // a camera that gives 16 bits a pixel: nothing here reads that, and it must not pass for a frame
    // without a code
    let mut pipeline = Pipeline::new();
    for kind in [core::CV_16UC1, core::CV_16UC3] {
        let deep = Mat::new_size_with_default(Size::new(800, 600), kind, Scalar::all(40_000.0))?;
        let error = pipeline.process(&deep).unwrap_err();
        assert_eq!(error.to_string(), "a frame is expected in bytes");
    }
    let two = Mat::new_size_with_default(Size::new(800, 600), core::CV_8UC2, Scalar::all(0.0))?;
    let error = pipeline.process(&two).unwrap_err();
    assert_eq!(
        error.to_string(),
        "a frame is expected in one, three or four channels, not 2"
    );
    Ok(())
}

#[test]
fn matrix_that_is_not_read_counts_before_a_qr_code_that_is_not_read() -> Result<()> {
    let file = bytes(20_000, 40);
    let (first, second) = (
        page_seen(&page_frame(&file, Some(0)))?,
        page_seen(&page_frame(&file, Some(1)))?,
    );
    let mut between = Mat::default();
    core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;
    // Next to the page, in a corner of the frame, is a small QR code of which half is covered: three
    // finder patterns and nothing that can be decoded. It is no reason to say less about the page.
    let code = code::encode(b"something else", 2, Level::L)?;
    let module = 3;
    let fill = |frame: &mut Mat, column: usize, row: usize, columns: usize, rows: usize| {
        let area = Rect::new(
            (60 + module * column) as i32,
            (60 + module * row) as i32,
            (module * columns) as i32,
            (module * rows) as i32,
        );
        imgproc::rectangle(frame, area, Scalar::all(0.0), imgproc::FILLED, imgproc::LINE_8, 0)
    };
    for row in 0..code.side() {
        for column in 0..code.side() {
            if code.is_dark(row, column) {
                fill(&mut between, column, row, 1, 1)?;
            }
        }
    }
    fill(&mut between, 9, 9, code.side() - 9, code.side() - 9)?;

    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.process(&first)?, Outcome::Packet { new: true });
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    let seen = pipeline.last_survey().codes.clone();
    assert!(seen.len() == 1 && seen[0].is_matrix(), "{seen:?}");
    assert!((119..=125).contains(&seen[0].modules), "{seen:?}");
    // a frame that has that code and no page says so
    let mut alone = Mat::new_size_with_default(Size::new(1920, 1080), CV_8UC3, Scalar::all(240.0))?;
    for row in 0..code.side() {
        for column in 0..code.side() {
            if code.is_dark(row, column) {
                fill(&mut alone, column, row, 1, 1)?;
            }
        }
    }
    fill(&mut alone, 9, 9, code.side() - 9, code.side() - 9)?;
    assert_eq!(pipeline.process(&alone)?, Outcome::Unread);
    let seen = pipeline.last_survey().codes.clone();
    assert!(seen.len() == 1 && !seen[0].is_matrix(), "{seen:?}");
    Ok(())
}

/// A camera frame that shows the picture of a matrix of any size, which has red, green and blue for every
/// pixel, as `page_seen` shows the matrix of these tests: somewhat smaller than it is, a little turned
/// and out of focus.
fn seen_from(grid: Grid, picture: &[u8]) -> Result<Mat> {
    let size = Size::new(grid.width() as i32, grid.height() as i32);
    let mut matrix = Mat::new_size_with_default(size, CV_8UC3, Scalar::all(0.0))?;
    for (to, from) in matrix
        .data_bytes_mut()?
        .chunks_exact_mut(3)
        .zip(picture.chunks_exact(3))
    {
        to.copy_from_slice(&[from[2], from[1], from[0]]);
    }
    let middle = Point2f::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
    let mut transform = imgproc::get_rotation_matrix_2d(middle, 1.5, 0.9)?;
    *transform.at_2d_mut::<f64>(0, 2)? += 960.0 - middle.x as f64;
    *transform.at_2d_mut::<f64>(1, 2)? += 540.0 - middle.y as f64;
    let mut seen = Mat::default();
    imgproc::warp_affine(
        &matrix,
        &mut seen,
        &transform,
        Size::new(1920, 1080),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(240.0),
    )?;
    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&seen, &mut blurred, Size::new(0, 0), 1.0)?;
    Ok(blurred)
}

/// The same for a frame of the page.
fn seen_in(grid: Grid, frame: &Frame) -> Result<Mat> {
    seen_from(
        grid,
        &tiles::render(grid, &tiles::pack(grid, &frame.encode(grid.capacity()))),
    )
}

#[test]
fn matrix_of_another_size_is_read_after_the_one_before() -> Result<()> {
    let file = bytes(20_000, 30);
    let origin = Origin::of(&file);
    let mut pipeline = Pipeline::new();
    assert_eq!(
        show_page(&mut pipeline, &file, Some(0))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(pipeline.packets(), Some((1, 4)));
    // The page is written anew for a camera of another shape: 96 by 72 tiles. What the frames before
    // showed is the first guess and not the only one.
    let classic = Grid::new(tiles::Shape::Classic, 8);
    let encoder = Encoder::new(&file, origin.checksum, classic.capacity() - OVERHEAD);
    let packet = Frame::Packet {
        origin,
        block: 0,
        sequence: 0,
        bytes: encoder.packet(0, 0),
    };
    assert_eq!(
        pipeline.process(&seen_in(classic, &packet)?)?,
        Outcome::Packet { new: true }
    );
    assert_eq!(pipeline.packets(), Some((1, encoder.plan().symbols())));
    assert_eq!(pipeline.last_survey().codes[0].modules, 96);
    // and back
    assert_eq!(
        show_page(&mut pipeline, &file, Some(1))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(pipeline.last_survey().codes[0].modules, 128);
    Ok(())
}

#[test]
fn file_of_a_page_with_a_misread_packet_is_not_handed_out() -> Result<()> {
    // kept in memory, without checksums to hold it against: the CRC-32 that the frames name is all
    // that stands between a misread packet and the file
    let file = bytes(40_000, 25);
    let mut pipeline = Pipeline::new();
    let mut outcome = Outcome::NoCode;
    for frame in 0..8 {
        let mut frame = page_frame(&file, Some(frame));
        if let Frame::Packet {
            sequence: 5, bytes, ..
        } = &mut frame
        {
            bytes[0] ^= 1;
        }
        outcome = pipeline.take(found(frame))?;
    }
    assert_eq!(outcome, Outcome::Corrupt);
    assert_eq!(pipeline.take_file(), None);
    // what was collected is dropped, and the file counts from the beginning
    assert_eq!((pipeline.received(), pipeline.packets()), (Some(0), None));
    Ok(())
}

#[test]
fn outline_of_a_matrix_that_is_read_is_where_the_matrix_is() -> Result<()> {
    // the page is 1160 by 656 pixels, and the frame shows it at nine tenths of that, turned a little
    let file = bytes(20_000, 31);
    let mut pipeline = Pipeline::new();
    assert_eq!(
        show_page(&mut pipeline, &file, Some(0))?,
        Outcome::Packet { new: true }
    );
    let seen = &pipeline.last_survey().codes[0];
    let side = |from: usize, to: usize| {
        (seen.outline[to].x - seen.outline[from].x).hypot(seen.outline[to].y - seen.outline[from].y)
    };
    // top, right, bottom, left
    let sides = [side(0, 1), side(1, 2), side(2, 3), side(3, 0)];
    for (side, expected) in sides.iter().zip([1043.1, 589.5, 1043.1, 589.5]) {
        assert!((side - expected).abs() < 3.0, "{sides:?}");
    }
    // around the middle of the frame, where the page is
    let middle = seen.outline.iter().fold((0.0, 0.0), |sum, corner| {
        (sum.0 + corner.x / 4.0, sum.1 + corner.y / 4.0)
    });
    assert!(
        (middle.0 - 960.0).abs() < 3.0 && (middle.1 - 540.0).abs() < 3.0,
        "{middle:?}"
    );
    // 8.1 pixels of the frame to a tile
    assert!((seen.module_size - 8.1).abs() < 0.1, "{}", seen.module_size);
    Ok(())
}

#[test]
fn web_page_is_read_whichever_way_up_the_camera_is() -> Result<()> {
    let file = bytes(20_000, 29);
    let upright = page_seen(&page_frame(&file, Some(0)))?;
    for turn in [
        core::ROTATE_90_CLOCKWISE,
        core::ROTATE_180,
        core::ROTATE_90_COUNTERCLOCKWISE,
    ] {
        let mut turned = Mat::default();
        core::rotate(&upright, &mut turned, turn)?;
        let mut pipeline = Pipeline::new();
        assert_eq!(
            pipeline.process(&turned)?,
            Outcome::Packet { new: true },
            "turn {turn:?}"
        );
        assert_eq!(pipeline.packets(), Some((1, 4)));
    }
    Ok(())
}

#[test]
fn program_reads_with_a_single_thread() -> Result<()> {
    let file = bytes(700, 34);
    let dir = scratch("one-thread");
    let input = pictures(&dir, &show(&file, "notes.bin"))?;
    // with no thread left to search the frames the program would wait for ever: it gets a minute
    let mut child = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .args(["--threads", "1"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    let begun = std::time::Instant::now();
    while child.try_wait()?.is_none() {
        if begun.elapsed() > std::time::Duration::from_secs(60) {
            child.kill()?;
            child.wait()?;
            panic!("qqcode --threads 1 did not end");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let run = child.wait_with_output()?;
    assert!(run.status.success());
    assert_eq!(run.stdout, file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_says_when_the_one_code_it_saw_could_not_be_read() -> Result<()> {
    // one picture, of a screen that is changing from one code to the next
    let file = bytes(600, 37);
    let messages = show(&file, "");
    let mut between = Mat::default();
    core::add_weighted_def(
        &frame(&messages[1])?,
        0.5,
        &frame(&messages[2])?,
        0.5,
        0.0,
        &mut between,
    )?;
    let dir = scratch("unread");
    let path = dir.join("frame_000.png");
    assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &between)?);
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(dir.join("frame_%03d.png"))
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(
        said.contains("no code of qqcode-show was read; 1 frames showed one that could not be"),
        "{said}"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_says_what_it_sees_in_the_order_it_sees_it() -> Result<()> {
    // a show that goes back to its start screen: start, the first piece, start again, the three pieces
    let file = bytes(700, 33);
    let messages = show(&file, "notes.bin");
    let dir = scratch("remarks");
    let shown = [0, 1, 0, 1, 2, 3].map(|index| messages[index].clone());
    let input = pictures(&dir, &shown)?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{said}");
    assert_eq!(run.stdout, file);
    let expected = format!(
        "frame 1: start screen in view: notes.bin, 700 bytes, waiting for the first code\n\
         frame 2: 272 of 700 bytes\n\
         frame 3: start screen in view: notes.bin, 700 bytes, waiting for the first code\n\
         frame 5: 544 of 700 bytes\n\
         frame 6: the file is whole\n\
         6 frames read, 700 bytes written to the standard output\n\
         SHA-256 {}\n\
         MD5 {}\n",
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert_eq!(said, expected);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_holds_the_file_against_the_sha_256_that_its_checksums_name() -> Result<()> {
    let file = bytes(50_000, 35);
    let dir = scratch("sha");
    for (index, frame) in [None].into_iter().chain((0..12).map(Some)).enumerate() {
        let path = dir.join(format!("frame_{index:03}.png"));
        let seen = page_seen(&page_frame(&file, frame))?;
        assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &seen)?);
    }
    // A list that is right megabyte by megabyte and names another SHA-256 for the whole file. Nobody
    // wrote it like that, so a file that goes by it does not get its name.
    let right = Checks::of("notes.bin", &file);
    let text = right.to_string().replace(
        &format!("sha256 {}", right.sha256),
        &format!("sha256 {}", digest::sha256(b"another file")),
    );
    assert!(Checks::parse(&text).is_ok() && text != right.to_string());
    let list = dir.join("notes.bin.checksums.txt");
    fs::write(&list, &text)?;
    let (output, part) = (dir.join("read.bin"), dir.join("read.bin.part"));
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(dir.join("frame_%03d.png"))
        .arg("-o")
        .arg(&output)
        .arg("--check")
        .arg(&list)
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "{said}");
    assert!(
        said.contains(&format!(
            "the file that was put together has not the SHA-256 that its checksums name: it is left as {}",
            part.display()
        )),
        "{said}"
    );
    // what arrived is there, under the name it had while it arrived
    assert!(!output.exists());
    assert_eq!(fs::read(&part)?, file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_writes_the_page_as_its_options_say_and_the_checksums_next_to_it() -> Result<()> {
    let file = bytes(50_000, 32);
    let dir = scratch("show");
    let (source, page) = (dir.join("notes.bin"), dir.join("show.html"));
    fs::write(&source, &file)?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode-show"))
        .arg(&source)
        .arg("--html")
        .arg(&page)
        .args([
            "--shape", "classic", "--scale", "8", "--rate", "12", "--spare", "25",
        ])
        .output()?;
    let said = String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    // every option is where it belongs in what the page is told
    let written = fs::read_to_string(&page)?;
    for told in ["\"pace\":12,", "\"spare\":25,", "\"columns\":96,\"rows\":72,"] {
        assert!(written.contains(told), "{told}");
    }
    // and in what is said of it
    for words in [
        "notes.bin, 50000 bytes in ",
        ", 12 a second: ",
        "with 25 sums to spare in a hundred",
        "matrix of 96x72 tiles",
    ] {
        assert!(said.contains(words), "{words}: {said}");
    }
    // the checksums of the file are next to the page, and the last line is the page as a browser is
    // told to open it
    let list = fs::read_to_string(dir.join("notes.bin.checksums.txt"))?;
    assert!(Checks::parse(&list) == Ok(Checks::of("notes.bin", &file)));
    let last = said.lines().last().unwrap_or_default();
    assert!(
        last.starts_with("file:///") && last.ends_with("/show.html"),
        "{last}"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_does_not_give_a_file_with_a_misread_packet_its_name() -> Result<()> {
    // The check of a frame lets a misread one through once in a long while. Here one is made up: the
    // fourth packet comes with a byte that is wrong and a check that is right, and there are no
    // checksums of the file to find it by.
    let file = bytes(50_000, 36);
    let dir = scratch("misread");
    for (index, frame) in [None].into_iter().chain((0..10).map(Some)).enumerate() {
        let mut frame = page_frame(&file, frame);
        if let Frame::Packet {
            sequence: 3, bytes, ..
        } = &mut frame
        {
            bytes[100] ^= 0x80;
        }
        let path = dir.join(format!("frame_{index:03}.png"));
        assert!(imgcodecs::imwrite_def(
            path.to_str().unwrap(),
            &page_seen(&frame)?
        )?);
    }
    let (output, part) = (dir.join("read.bin"), dir.join("read.bin.part"));
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(dir.join("frame_%03d.png"))
        .arg("-o")
        .arg(&output)
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "{said}");
    assert!(
        said.contains(&format!(
            "the file that was put together has not the length and the checksum that its frames name: it is \
             left as {}",
            part.display()
        )),
        "{said}"
    );
    // all of it is there, as long as the file and not the file
    assert!(!output.exists());
    let left = fs::read(&part)?;
    assert!(left.len() == file.len() && left != file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn share_that_is_put_right_is_of_all_the_bytes_that_a_frame_has() -> Result<()> {
    // Sixteen tiles in a row are painted over, as by something that lies on the screen. They are 8 of
    // the 4536 bytes of symbols that the frame has and 4 of its 2268 bytes of colours, and the check
    // bytes put them right.
    let file = bytes(20_000, 38);
    let frame = page_frame(&file, Some(0));
    let mut picture = tiles::render(GRID, &tiles::pack(GRID, &frame.encode(GRID.capacity())));
    for column in 40..56 {
        let (left, top) = GRID.origin(column, 30);
        for y in top..top + tiles::TILE {
            let from = 3 * (y * GRID.width() + left);
            picture[from..from + 3 * tiles::TILE].fill(0);
        }
    }
    let mut pipeline = Pipeline::new();
    let outcome = pipeline.process(&seen_from(GRID, &picture)?)?;
    assert_eq!(outcome, Outcome::Packet { new: true });
    let wrong = pipeline.last_survey().codes[0].wrong.expect("a matrix");
    assert!((wrong - 12.0 / 6804.0).abs() < 1e-6, "{} bytes", wrong * 6804.0);
    // and nothing in a frame that is as the page drew it
    show_page(&mut pipeline, &file, Some(1))?;
    assert_eq!(pipeline.last_survey().codes[0].wrong, Some(0.0));
    Ok(())
}

#[test]
fn bytes_that_arrived_before_are_not_counted_as_gone_by_unread() -> Result<()> {
    let file = bytes(1500, 26);
    let messages = show(&file, "notes.bin");
    let mut pipeline = Pipeline::new();
    let mut take = |index: usize| pipeline.process(&frame(&messages[index])?);
    assert_eq!(take(0)?, Outcome::Ready);
    assert_eq!(take(1)?, Outcome::Piece { new: 272, skipped: 0 });
    // the second piece is missed, and comes in after the third
    assert_eq!(
        take(3)?,
        Outcome::Piece {
            new: 272,
            skipped: 272
        }
    );
    assert_eq!(take(2)?, Outcome::Piece { new: 272, skipped: 0 });
    // The show goes back to its first code and from there to the fourth. All that lies between the
    // two has arrived, so nothing went by unread.
    assert_eq!(take(1)?, Outcome::Piece { new: 0, skipped: 0 });
    assert_eq!(take(4)?, Outcome::Piece { new: 272, skipped: 0 });
    // from the fourth to the sixth, the fifth did
    assert_eq!(
        take(6)?,
        Outcome::Piece {
            new: 140,
            skipped: 272
        }
    );
    Ok(())
}

#[test]
fn matrix_that_is_not_read_is_one_only_after_a_matrix_was_read() -> Result<()> {
    let file = bytes(20_000, 28);
    let (first, second) = (
        page_seen(&page_frame(&file, Some(0)))?,
        page_seen(&page_frame(&file, Some(1)))?,
    );
    let mut between = Mat::default();
    core::add_weighted_def(&first, 0.5, &second, 0.5, 0.0, &mut between)?;
    let nothing = Mat::new_size_with_default(Size::new(1920, 1080), CV_8UC3, Scalar::all(240.0))?;
    // Four finder patterns and nothing between them that can be read: that is no matrix as long as no
    // frame showed one, since four of them can be anything.
    let mut pipeline = Pipeline::new();
    pipeline.process(&between)?;
    let codes = &pipeline.last_survey().codes;
    assert!(codes.iter().all(|code| !code.is_matrix()), "{codes:?}");
    // After a matrix was read it is one that can't be read, of as many tiles as lie between the
    // middles of its finder patterns: 122 of the 128.
    assert_eq!(pipeline.process(&first)?, Outcome::Packet { new: true });
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    let seen = pipeline.last_survey().codes[0].clone();
    assert!(
        seen.is_matrix() && (119..=125).contains(&seen.modules),
        "{seen:?}"
    );
    // and it stays one when a frame without any code came in between
    assert_eq!(pipeline.process(&nothing)?, Outcome::NoCode);
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    assert!(pipeline.last_survey().codes[0].is_matrix());
    Ok(())
}
