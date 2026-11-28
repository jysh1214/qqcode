//! End to end on synthetic frames: the page as `qqcode-show` has it, seen a little turned and out of focus,
//! and the program `qqcode` on pictures of it.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;
use opencv::core::{self, CV_8UC3, Mat, Point2f, Scalar, Size};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};

use qqcode::checks::{Checks, Note};
use qqcode::fountain::{Encoder, Schedule};
use qqcode::packet::{Frame, OVERHEAD, Origin};
use qqcode::scan::Found;
use qqcode::tiles::{self, Grid};
use qqcode::{Kind, Outcome, Pipeline, digest};

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
    seen_in(GRID, frame)
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
    let seen = pipeline.last_survey().codes[0].clone();
    assert_eq!((seen.kind, seen.modules), (Kind::Piece, 128));
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
        frame: Some(frame),
        wrong: 0.0,
    }]
}

#[test]
fn frame_without_a_matrix_shows_no_code() -> Result<()> {
    let mut pipeline = Pipeline::new();
    let nothing = Mat::new_size_with_default(Size::new(1920, 1080), CV_8UC3, Scalar::all(128.0))?;
    assert_eq!(pipeline.process(&nothing)?, Outcome::NoCode);
    assert!(pipeline.last_survey().codes.is_empty());
    assert_eq!(pipeline.received(), None);
    assert_eq!((pipeline.files(), pipeline.file()), (0, None));

    // The matrix is told by its colours: a frame in grey has none that is read, whatever it shows.
    let file = bytes(20_000, 2);
    let mut grey = Mat::default();
    imgproc::cvt_color_def(
        &page_seen(&page_frame(&file, Some(0)))?,
        &mut grey,
        imgproc::COLOR_BGR2GRAY,
    )?;
    assert_eq!(pipeline.process(&grey)?, Outcome::NoCode);
    assert_eq!(pipeline.file(), None);
    Ok(())
}

#[test]
fn frames_of_another_file_start_over() -> Result<()> {
    // four packets of 5265 bytes, and three
    let (first, second) = (bytes(20_000, 5), bytes(15_000, 6));
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.take(found(page_frame(&first, None)))?, Outcome::Ready);
    assert_eq!(
        pipeline.take(found(page_frame(&first, Some(0))))?,
        Outcome::Packet { new: true }
    );
    assert_eq!((pipeline.files(), pipeline.name()), (1, Some("notes.bin")));
    assert_eq!(pipeline.packets(), Some((1, 4)));

    // without a start screen: the second packet of the other file
    assert_eq!(
        pipeline.take(found(page_frame(&second, Some(1))))?,
        Outcome::Packet { new: true }
    );
    assert_eq!((pipeline.files(), pipeline.name()), (2, None));
    assert_eq!(pipeline.file(), Some(Origin::of(&second)));
    // nothing of the first file counts for it
    assert_eq!(
        (pipeline.packets(), pipeline.received()),
        (Some((1, 3)), Some(5265))
    );
    assert_eq!(
        pipeline.take(found(page_frame(&second, Some(0))))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(
        pipeline.take(found(page_frame(&second, Some(2))))?,
        Outcome::Complete
    );
    assert_eq!(pipeline.take_file(), Some(second));
    Ok(())
}

#[test]
fn empty_file_is_whole_at_its_start_screen() -> Result<()> {
    let mut pipeline = Pipeline::new();
    assert_eq!(show_page(&mut pipeline, &[], None)?, Outcome::Complete);
    assert_eq!(pipeline.take_file(), Some(Vec::new()));
    Ok(())
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
fn file_on_disk_is_gone_on_with_by_the_next_run() -> Result<()> {
    // eight packets of 5265 bytes, in eight sections of 5000
    let file = bytes(40_000, 41);
    let dir = scratch("resumed");
    let (part, sums, index) = (
        dir.join("read.part"),
        dir.join("read.part.sums"),
        dir.join("read.part.index"),
    );
    let checks = Checks::in_sections("notes.bin", &file, 5000);
    let run = || {
        Pipeline::storing(&part, &sums)
            .resuming(&index)
            .checked(checks.clone())
    };
    let packet = |frame: u64| found(page_frame(&file, Some(frame)));
    let mut pipeline = run();
    assert_eq!(pipeline.take(found(page_frame(&file, None)))?, Outcome::Ready);
    // three packets of the file and a sum
    for frame in [0, 1, 3, 8] {
        assert_eq!(pipeline.take(packet(frame))?, Outcome::Packet { new: true });
    }
    assert_eq!(pipeline.take_resumed(), None);
    assert_eq!(
        (pipeline.packets(), pipeline.received()),
        (Some((4, 8)), Some(21_060))
    );
    // the run is stopped, and what it has stays where it is
    assert!(!index.exists());
    pipeline.save()?;
    assert!(index.exists());
    drop(pipeline);
    let left = fs::read(&part)?;

    // The next run comes to a page that has gone on: it has no start screen to see, and nothing but
    // the packet in front of it to tell it what the file is.
    let mut pipeline = run();
    assert_eq!((pipeline.packets(), pipeline.take_resumed()), (None, None));
    assert_eq!(pipeline.take(packet(4))?, Outcome::Packet { new: true });
    assert_eq!(pipeline.take_resumed(), Some(21_060));
    assert_eq!(pipeline.take_resumed(), None);
    assert_eq!(
        (pipeline.packets(), pipeline.sound()),
        (Some((5, 8)), Some((0, 8)))
    );
    // nothing was made anew, and what the first run read is not read again
    assert_eq!(fs::read(&part)?[..4 * 5265], left[..]);
    for frame in [0, 3, 8] {
        assert_eq!(pipeline.take(packet(frame))?, Outcome::Packet { new: false });
    }
    // stopped once more, with nothing new, and gone on with by a third run to the end
    pipeline.save()?;
    drop(pipeline);
    let mut pipeline = run();
    let mut outcome = Outcome::NoCode;
    for frame in 5..24 {
        outcome = pipeline.take(packet(frame))?;
        if outcome == Outcome::Complete {
            break;
        }
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(pipeline.take_resumed(), Some(5 * 5265));
    assert_eq!(pipeline.take_notes(), []);
    assert_eq!(fs::read(&part)?, file);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn what_a_run_left_of_something_else_is_not_thrown_away() -> Result<()> {
    let file = bytes(40_000, 42);
    let origin = Origin::of(&file);
    let dir = scratch("left");
    let (part, sums, index) = (
        dir.join("read.part"),
        dir.join("read.part.sums"),
        dir.join("read.part.index"),
    );
    let run = || Pipeline::storing(&part, &sums).resuming(&index);
    let packet = |frame: u64| found(page_frame(&file, Some(frame)));
    // the page with a larger matrix: packets of 10077 bytes, of which the file is four
    let encoder = Encoder::new(&file, origin.checksum, 10_077);
    let larger = |sequence: u32| {
        found(Frame::Packet {
            origin,
            block: 0,
            sequence,
            bytes: encoder.packet(0, sequence),
        })
    };
    let mut pipeline = run();
    for frame in 0..5 {
        pipeline.take(packet(frame))?;
    }
    pipeline.save()?;
    drop(pipeline);
    let left = (fs::read(&part)?, fs::read(&index)?);

    // The same file in packets of another size: what there is of it is of no use for them, and took
    // too long to read for the page to do away with it.
    let mut pipeline = run();
    let error = pipeline.take(larger(0)).unwrap_err().to_string();
    let told = format!(
        "{} has what arrived of this file in packets of 5265 bytes, and the page shows packets of 10077 bytes \
         now. The file is gone on with by the matrix that it was begun with; for another one, {0} is to be \
         removed first, and the file begins anew",
        part.display()
    );
    assert_eq!(error, told);
    // another file
    let other = bytes(30_000, 43);
    let mut pipeline = run();
    let error = pipeline
        .take(found(page_frame(&other, Some(0))))
        .unwrap_err()
        .to_string();
    let told = format!(
        "{} has what arrived of another file, one of 40000 bytes with the CRC-32 {:08x}, and is not thrown away \
         for the file on show: it is to be moved or removed first",
        part.display(),
        origin.checksum
    );
    assert_eq!(error, told);
    // and one without a byte, which would be whole at its start screen
    let mut pipeline = run();
    let error = pipeline.take(found(page_frame(&[], None))).unwrap_err();
    assert!(
        error.to_string().contains("has what arrived of another file"),
        "{error}"
    );
    assert_eq!((fs::read(&part)?, fs::read(&index)?), left);
    // the page as it was is gone on with after all that
    let mut pipeline = run();
    assert_eq!(pipeline.take(packet(5))?, Outcome::Packet { new: true });
    assert_eq!(pipeline.packets(), Some((6, 8)));

    // What this run has itself is another matter: a page that goes on with another matrix while it is
    // read begins the file anew, as ever, and the index that was there is not that of the new file.
    assert_eq!(pipeline.take(larger(0))?, Outcome::Packet { new: true });
    assert_eq!(pipeline.packets(), Some((1, 4)));
    assert!(!index.exists());
    pipeline.save()?;
    drop(pipeline);
    let mut pipeline = run();
    assert_eq!(pipeline.take(larger(1))?, Outcome::Packet { new: true });
    assert_eq!(
        (pipeline.take_resumed(), pipeline.packets()),
        (Some(10_077), Some((2, 4)))
    );
    // so does another file that comes on show
    assert_eq!(
        pipeline.take(found(page_frame(&other, Some(0))))?,
        Outcome::Packet { new: true }
    );
    assert_eq!(pipeline.file(), Some(Origin::of(&other)));
    assert!(!index.exists());
    drop(pipeline);

    // An index whose file is gone is of nothing, and one that is damaged is said to be.
    fs::write(&index, &left.1)?;
    fs::remove_file(&part)?;
    let mut pipeline = run();
    assert_eq!(pipeline.take(larger(0))?, Outcome::Packet { new: true });
    assert_eq!(
        (pipeline.take_resumed(), pipeline.packets()),
        (None, Some((1, 4)))
    );
    pipeline.save()?;
    drop(pipeline);
    let mut damaged = fs::read(&index)?;
    damaged[20] ^= 1;
    fs::write(&index, &damaged)?;
    let mut pipeline = run();
    let error = format!("{:#}", pipeline.take(larger(1)).unwrap_err());
    let told = format!(
        "can't read {}: this is no index: its checksum is wrong",
        index.display()
    );
    assert_eq!(error, told);
    assert_eq!(fs::read(&index)?, damaged);
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn no_other_file_is_taken_than_the_one_that_the_checksums_are_of() -> Result<()> {
    let (file, other) = (bytes(20_000, 17), bytes(20_000, 18));
    let mut pipeline = Pipeline::new().checked(Checks::of("notes.bin", &file));
    // the web page of another file: its start screen, and a packet
    let error = show_page(&mut pipeline, &other, None).unwrap_err().to_string();
    let expected = format!(
        "the file on show has 20000 bytes and the CRC-32 {:08x}, and the checksums are those of \"notes.bin\", \
         a file of 20000 bytes with the CRC-32 {:08x}",
        Origin::of(&other).checksum,
        Origin::of(&file).checksum
    );
    assert_eq!(error, expected);
    let error = show_page(&mut pipeline, &other, Some(0)).unwrap_err();
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
    let mut pipeline = Pipeline::new().checked(checks);
    let mut outcome = Outcome::NoCode;
    for frame in 0..4 {
        outcome = show_page(&mut pipeline, &file, Some(frame))?;
    }
    assert_eq!(outcome, Outcome::Corrupt);
    assert_eq!(pipeline.take_file(), None);
    assert_eq!(pipeline.take_notes(), [Note::Wrong { sections: 2, from: 0 }]);
    // The file itself, held against checksums that are its own, is whole when it has arrived.
    let mut pipeline = Pipeline::new().checked(Checks::in_sections("notes.bin", &file, 8000));
    for frame in 0..4 {
        outcome = show_page(&mut pipeline, &file, Some(frame))?;
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!(pipeline.take_file().as_ref(), Some(&file));
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
    assert_eq!(pipeline.last_survey().codes[0].kind, Kind::Unread);
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
    // Where it was to be written there is nothing then. What arrived is kept next to it, with the list
    // of it, and the program says so.
    let to = |input: &Path| {
        Command::new(env!("CARGO_BIN_EXE_qqcode"))
            .arg(input)
            .arg("-o")
            .arg(&output)
            .output()
    };
    let run = to(&input)?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && !output.exists(), "{said}");
    let end = format!(
        "9 frames read, nothing written\nError: the file is not whole: 8 of the 10 packets it takes have arrived; \
         they are kept in {}, and the same command goes on with them\n",
        part.display()
    );
    assert!(said.ends_with(&end), "{said}");
    let (sums, index) = (dir.join("read.bin.part.sums"), dir.join("read.bin.part.index"));
    assert!(part.exists() && sums.exists() && index.exists());
    assert_eq!(fs::read(&part)?[..8 * 5265], file[..8 * 5265]);
    // A run that sees nothing of the show leaves all of that as it is.
    let nothing = dir.join("nothing");
    fs::create_dir_all(&nothing)?;
    let blank = Mat::new_size_with_default(Size::new(640, 480), CV_8UC3, Scalar::all(240.0))?;
    assert!(imgcodecs::imwrite_def(
        nothing.join("frame_000.png").to_str().unwrap(),
        &blank
    )?);
    let kept = fs::read(&index)?;
    let run = to(&nothing.join("frame_%03d.png"))?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success(), "{said}");
    assert!(
        said.ends_with("no frame showed a code of qqcode-show\n"),
        "{said}"
    );
    assert!(part.exists() && sums.exists() && fs::read(&index)? == kept);
    // The same command goes on with it, on a page that has gone on meanwhile: of the four frames that
    // it sees, the first two are the packets that were missing.
    let more = dir.join("more");
    fs::create_dir_all(&more)?;
    for (index, frame) in (8..12).enumerate() {
        let path = more.join(format!("frame_{index:03}.png"));
        let seen = page_seen(&page_frame(&file, Some(frame)))?;
        assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &seen)?);
    }
    let run = to(&more.join("frame_%03d.png"))?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success() && run.stdout.is_empty(), "{said}");
    let begin = format!(
        "frame 1: going on with {}, which has 42120 of the 50000 bytes\n",
        part.display()
    );
    assert!(said.starts_with(&begin), "{said}");
    let end = format!(
        "frame 2: the file is whole\n2 frames read, 50000 bytes written to {}\nSHA-256 {}\nMD5 {}\n",
        output.display(),
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert!(said.ends_with(&end), "{said}");
    assert_eq!(fs::read(&output)?, file);
    assert!(!part.exists() && !sums.exists() && !index.exists());
    fs::remove_dir_all(&dir)?;
    Ok(())
}

/// A directory of its own for the pictures of one test.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("qqcode-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Saves what a camera sees of `frames` of the page of `file` as pictures and returns the name that
/// `qqcode` reads them by.
fn pictures(dir: &Path, file: &[u8], frames: &[Option<u64>]) -> Result<PathBuf> {
    for (index, frame) in frames.iter().enumerate() {
        let path = dir.join(format!("frame_{index:03}.png"));
        let seen = page_seen(&page_frame(file, *frame))?;
        assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &seen)?);
    }
    Ok(dir.join("frame_%03d.png"))
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
    let origin = Origin::of(&bytes(700, 27));
    let start = |name: &str| {
        found(Frame::Start {
            origin,
            name: name.to_owned(),
        })
    };
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.take(start(""))?, Outcome::Ready);
    assert_eq!(pipeline.name(), None);
    // and one with a name does
    assert_eq!(pipeline.take(start("notes.bin"))?, Outcome::Ready);
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
    // it is the same file still, and whole once its packets have arrived as they are
    for frame in 0..8 {
        outcome = pipeline.take(found(page_frame(&file, Some(frame))))?;
    }
    assert_eq!(outcome, Outcome::Complete);
    assert_eq!((pipeline.files(), pipeline.take_file()), (1, Some(file)));
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
    // the start screen and the four packets that the file is
    let file = bytes(20_000, 34);
    let dir = scratch("one-thread");
    let input = pictures(&dir, &file, &[None, Some(0), Some(1), Some(2), Some(3)])?;
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
fn program_says_when_the_page_could_not_be_read_after_its_start_screen() -> Result<()> {
    // the start screen, and a picture of the page while it changes from one frame to the next
    let file = bytes(20_000, 37);
    let dir = scratch("unread");
    let input = pictures(&dir, &file, &[None])?;
    let mut between = Mat::default();
    core::add_weighted_def(
        &page_seen(&page_frame(&file, Some(0)))?,
        0.5,
        &page_seen(&page_frame(&file, Some(1)))?,
        0.5,
        0.0,
        &mut between,
    )?;
    let path = dir.join("frame_001.png");
    assert!(imgcodecs::imwrite_def(path.to_str().unwrap(), &between)?);
    // one thread, so that the start screen is read before the picture after it is looked at: a matrix
    // that can't be read is one only after a matrix was read
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode"))
        .arg(&input)
        .args(["--threads", "1"])
        .output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(
        said.ends_with(
            "2 frames read, nothing written\nError: the file is not whole: none of its packets is at hand; 1 \
             frame showed the page and could not be read: nearer, sharper, or a smaller --scale\n"
        ),
        "{said}"
    );

    // the start screen alone
    fs::remove_file(&path)?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(!run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(
        said.ends_with(
            "1 frames read, nothing written\nError: the file is not whole: none of its packets is at hand\n"
        ),
        "{said}"
    );
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_says_what_it_sees_in_the_order_it_sees_it() -> Result<()> {
    // a page that goes back to its start screen: start, the first packet, start again, the other three
    let file = bytes(20_000, 33);
    let dir = scratch("remarks");
    let shown = [None, Some(0), None, Some(1), Some(2), Some(3)];
    let input = pictures(&dir, &file, &shown)?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success(), "{said}");
    assert_eq!(run.stdout, file);
    // how far the file has come is said once a second, which these frames may take or not
    let remarks = said
        .lines()
        .filter(|line| !line.ends_with(" packets"))
        .collect::<Vec<_>>();
    let expected = format!(
        "frame 1: start screen in view: notes.bin, 20000 bytes, waiting for the first code\n\
         frame 3: start screen in view: notes.bin, 20000 bytes, waiting for the first code\n\
         frame 6: the file is whole\n\
         6 frames read, 20000 bytes written to the standard output\n\
         SHA-256 {}\n\
         MD5 {}",
        digest::sha256(&file),
        digest::md5(&file)
    );
    assert_eq!(remarks.join("\n"), expected);
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

    // The same of a file that took two runs. Nothing says which of its packets to read again, so the
    // run after those two is not to go on with it: the list of what arrived is gone, and the file
    // begins anew.
    fs::remove_file(&part)?;
    let index = dir.join("read.bin.part.index");
    let first = dir.join("first");
    fs::create_dir_all(&first)?;
    for frame in 0..6 {
        let name = format!("frame_{frame:03}.png");
        fs::copy(dir.join(&name), first.join(&name))?;
    }
    let run = |input: &Path| {
        Command::new(env!("CARGO_BIN_EXE_qqcode"))
            .arg(input.join("frame_%03d.png"))
            .arg("-o")
            .arg(&output)
            .output()
    };
    let stopped = run(&first)?;
    let said = String::from_utf8_lossy(&stopped.stderr);
    assert!(
        said.contains("5 of the 10 packets it takes have arrived; they are kept in"),
        "{said}"
    );
    assert!(index.exists());
    let whole = run(&dir)?;
    let said = String::from_utf8_lossy(&whole.stderr);
    assert!(
        said.contains(", which has 26325 of the 50000 bytes\n") && said.contains("it is left as"),
        "{said}"
    );
    assert!(!whole.status.success() && !output.exists() && part.exists() && !index.exists());
    let again = run(&dir)?;
    let said = String::from_utf8_lossy(&again.stderr);
    assert!(
        !said.contains("going on with") && said.contains("it is left as"),
        "{said}"
    );
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
    let wrong = pipeline.last_survey().codes[0].wrong;
    assert!((wrong - 12.0 / 6804.0).abs() < 1e-6, "{} bytes", wrong * 6804.0);
    // and nothing in a frame that is as the page drew it
    show_page(&mut pipeline, &file, Some(1))?;
    assert_eq!(pipeline.last_survey().codes[0].wrong, 0.0);
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
    assert_eq!(pipeline.process(&between)?, Outcome::NoCode);
    assert!(pipeline.last_survey().codes.is_empty());
    // After a matrix was read it is one that can't be read, of as many tiles as lie between the
    // middles of its finder patterns: 122 of the 128.
    assert_eq!(pipeline.process(&first)?, Outcome::Packet { new: true });
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    let seen = pipeline.last_survey().codes[0].clone();
    assert!(
        seen.kind == Kind::Unread && (119..=125).contains(&seen.modules),
        "{seen:?}"
    );
    // and it stays one when a frame without any code came in between
    assert_eq!(pipeline.process(&nothing)?, Outcome::NoCode);
    assert_eq!(pipeline.process(&between)?, Outcome::Unread);
    assert_eq!(pipeline.last_survey().codes.len(), 1);
    Ok(())
}

#[test]
fn file_longer_than_any_that_is_taken_is_another_matter() -> Result<()> {
    // a start screen is all it takes to name a file: one of 2^40 bytes is the longest that is taken
    let start = |length: u64| Frame::Start {
        origin: Origin { checksum: 7, length },
        name: "disk.img".to_owned(),
    };
    let mut pipeline = Pipeline::new();
    assert_eq!(pipeline.take(found(start(1 << 40)))?, Outcome::Ready);
    assert_eq!(pipeline.file().map(|file| file.length), Some(1 << 40));
    // a byte more and nothing is made of it: the file on show is still the one before
    assert_eq!(pipeline.take(found(start((1 << 40) + 1)))?, Outcome::Other);
    assert_eq!(
        (pipeline.files(), pipeline.file().map(|file| file.length)),
        (1, Some(1 << 40))
    );
    Ok(())
}

#[test]
fn file_that_agrees_with_the_checksums_in_its_crc_alone_is_not_taken() -> Result<()> {
    // a file is named by its length and its CRC-32 together: one that has either of them is another file
    let file = bytes(20_000, 39);
    let origin = Origin::of(&file);
    let mut pipeline = Pipeline::new().checked(Checks::of("notes.bin", &file));
    let longer = Origin {
        length: 20_001,
        ..origin
    };
    let changed = Origin {
        checksum: origin.checksum ^ 1,
        ..origin
    };
    for other in [longer, changed] {
        let start = Frame::Start {
            origin: other,
            name: "notes.bin".to_owned(),
        };
        let error = pipeline.take(found(start)).unwrap_err().to_string();
        let begin = format!(
            "the file on show has {} bytes and the CRC-32 {:08x}, and the checksums are those of \"notes.bin\"",
            other.length, other.checksum
        );
        assert!(error.starts_with(&begin), "{error}");
    }
    assert_eq!(pipeline.take(found(page_frame(&file, None)))?, Outcome::Ready);
    Ok(())
}

#[test]
fn program_writes_a_file_without_a_byte() -> Result<()> {
    // whole at its start screen, which is all that the page shows of it
    let dir = scratch("empty");
    let input = pictures(&dir, &[], &[None])?;
    let run = Command::new(env!("CARGO_BIN_EXE_qqcode")).arg(&input).output()?;
    let said = String::from_utf8_lossy(&run.stderr);
    assert!(run.status.success() && run.stdout.is_empty(), "{said}");
    assert!(said.contains("frame 1: the file is whole\n"), "{said}");
    let end = format!(
        " read, 0 bytes written to the standard output\nSHA-256 {}\nMD5 {}\n",
        digest::sha256(b""),
        digest::md5(b"")
    );
    assert!(said.ends_with(&end), "{said}");
    fs::remove_dir_all(&dir)?;
    Ok(())
}

#[test]
fn program_does_not_go_on_with_a_file_that_has_not_the_sha_256_of_its_checksums() -> Result<()> {
    // As in `program_holds_the_file_against_the_sha_256_that_its_checksums_name`, of a file that takes
    // two runs: the first sees two of its four packets and leaves the list of them.
    let file = bytes(20_000, 38);
    let dir = scratch("sha-runs");
    let input = pictures(&dir, &file, &[None, Some(0), Some(1), Some(2), Some(3)])?;
    let first = dir.join("first");
    fs::create_dir_all(&first)?;
    for frame in 0..3 {
        let name = format!("frame_{frame:03}.png");
        fs::copy(dir.join(&name), first.join(&name))?;
    }
    let right = Checks::of("notes.bin", &file);
    let text = right.to_string().replace(
        &format!("sha256 {}", right.sha256),
        &format!("sha256 {}", digest::sha256(b"another file")),
    );
    let list = dir.join("notes.bin.checksums.txt");
    fs::write(&list, &text)?;
    let (output, part) = (dir.join("read.bin"), dir.join("read.bin.part"));
    let index = dir.join("read.bin.part.index");
    let run = |input: &Path| {
        Command::new(env!("CARGO_BIN_EXE_qqcode"))
            .arg(input)
            .arg("-o")
            .arg(&output)
            .arg("--check")
            .arg(&list)
            .output()
    };
    let stopped = run(&first.join("frame_%03d.png"))?;
    let said = String::from_utf8_lossy(&stopped.stderr);
    assert!(
        said.contains("2 of the 4 packets it takes have arrived; they are kept in"),
        "{said}"
    );
    assert!(index.exists());
    // The second run goes on with them, and the file that is whole then is not the one that the list
    // names. Nothing says which of its packets to read again, so the list of what arrived is gone, and
    // the run after this one begins the file anew.
    let whole = run(&input)?;
    let said = String::from_utf8_lossy(&whole.stderr);
    assert!(
        said.contains("going on with") && said.contains("has not the SHA-256 that its checksums name"),
        "{said}"
    );
    assert!(!whole.status.success() && !output.exists() && part.exists());
    assert!(!index.exists(), "the index of a file that is whole and wrong");
    fs::remove_dir_all(&dir)?;
    Ok(())
}
