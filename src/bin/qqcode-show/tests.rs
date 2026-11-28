use anyhow::Result;
use opencv::core::{self, CV_8UC3, Mat, Point2f, Scalar, Size as Pixels, Vector};
use opencv::imgproc;
use opencv::prelude::*;
use qqcode::tiles;
use qqcode::{Kind, Outcome, Pipeline};

use super::*;
use crate::web::{Format, Show};

/// Bytes of every value, in no pattern.
fn bytes(count: usize) -> Vec<u8> {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 32) as u8
    };
    (0..count).map(|_| next()).collect()
}

/// A camera frame that shows the screen in perspective in a dim room, a little out of focus.
fn photograph(screen: &Mat, blur: f64) -> Result<Mat> {
    let size = screen.size()?;
    let (w, h) = ((size.width - 1) as f32, (size.height - 1) as f32);
    let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
    let seen =
        [(210.0, 130.0), (1650.0, 90.0), (1740.0, 960.0), (150.0, 900.0)].map(|(x, y)| Point2f::new(x, y));
    let transform = imgproc::get_perspective_transform_def(
        &Vector::<Point2f>::from_slice(&flat),
        &Vector::<Point2f>::from_slice(&seen),
    )?;
    let mut frame = Mat::default();
    imgproc::warp_perspective(
        screen,
        &mut frame,
        &transform,
        Pixels::new(1920, 1080),
        imgproc::INTER_AREA,
        core::BORDER_CONSTANT,
        Scalar::all(40.0),
    )?;
    if blur > 0.0 {
        let mut blurred = Mat::default();
        imgproc::gaussian_blur_def(&frame, &mut blurred, Pixels::new(0, 0), blur)?;
        frame = blurred;
    }
    Ok(frame)
}

#[test]
fn web_page_is_read_from_the_start_screen_to_the_whole_file() -> Result<()> {
    let file = bytes(40_000);
    let grid = Grid::new(Shape::Wide, 8);
    let format = Format {
        grid,
        rate: 15,
        spare: 25,
    };
    let show = Show::new("notes.bin", &file, format);
    // 5103 bytes a frame, 5076 of them of the file: 8 frames
    assert_eq!((grid.capacity(), show.frames()), (5103, 8));
    let mut pipeline = Pipeline::new();
    // what the page puts on the screen, seen in perspective and a little out of focus
    let mut take = |frame: Option<u64>| -> Result<Outcome> {
        let picture = tiles::render(grid, &show.tiles(frame));
        let size = Pixels::new(grid.width() as i32, grid.height() as i32);
        let mut screen = Mat::new_size_with_default(size, CV_8UC3, Scalar::all(0.0))?;
        for (to, from) in screen
            .data_bytes_mut()?
            .chunks_exact_mut(3)
            .zip(picture.chunks_exact(3))
        {
            to.copy_from_slice(&[from[2], from[1], from[0]]);
        }
        // a white page around the matrix
        let mut page = Mat::default();
        core::copy_make_border(
            &screen,
            &mut page,
            30,
            30,
            30,
            30,
            core::BORDER_CONSTANT,
            Scalar::all(255.0),
        )?;
        pipeline.process(&photograph(&page, 1.0)?)
    };
    assert_eq!(take(None)?, Outcome::Ready);
    // the camera misses the second and the fifth frame of the file, and the page goes on with sums
    // of its packets: any eight frames make the file, give or take one
    let mut outcomes = Vec::new();
    for frame in [0, 2, 3, 5, 6, 7, 8, 9, 10, 11] {
        outcomes.push(take(Some(frame))?);
        if outcomes.last() == Some(&Outcome::Complete) {
            break;
        }
    }
    assert!((8..=9).contains(&outcomes.len()), "{outcomes:?}");
    assert!(
        outcomes[..7]
            .iter()
            .all(|outcome| *outcome == Outcome::Packet { new: true })
    );
    assert_eq!(pipeline.take_file(), Some(file));
    let survey = pipeline.last_survey().codes[0].clone();
    assert_eq!((survey.kind, survey.modules), (Kind::Piece, 128));
    // some 1500 pixels of the frame for 1220 of the page
    assert!(
        (10.0..13.5).contains(&survey.module_size),
        "{}",
        survey.module_size
    );
    Ok(())
}

#[test]
fn options_are_checked() {
    let parse = |arguments: &[&str]| {
        let mut command = vec!["qqcode-show", "notes.bin"];
        command.extend(arguments);
        Cli::try_parse_from(command)
    };
    let cli = parse(&["--html", "show.html"]).unwrap();
    assert_eq!(
        (cli.file, cli.html),
        (PathBuf::from("notes.bin"), PathBuf::from("show.html"))
    );
    assert_eq!(
        (cli.shape, cli.scale, cli.rate, cli.spare, cli.ask),
        (Shape::Wide, 11, 15, 10, false)
    );
    // a page is all there is to show a file on
    assert!(parse(&[]).is_err());
    for wrong in [
        &["--scale", "3"][..],
        &["--scale", "25"],
        &["--shape", "round"],
        &["--rate", "0"],
        &["--rate", "61"],
        &["--spare", "401"],
        &["--spare", "-1"],
    ] {
        let mut arguments = vec!["--html", "show.html"];
        arguments.extend(wrong);
        assert!(parse(&arguments).is_err(), "{wrong:?}");
    }
    assert_eq!(
        (count(1, "frame"), count(0, "section")),
        ("1 frame".to_owned(), "0 sections".to_owned())
    );
}

#[test]
fn options_take_the_ends_of_their_ranges() {
    let parse = |arguments: &[&str]| {
        let mut command = vec!["qqcode-show", "notes.bin", "--html", "show.html"];
        command.extend(arguments);
        Cli::try_parse_from(command).map(|cli| (cli.scale, cli.rate, cli.spare))
    };
    assert_eq!(
        parse(&["--scale", "24", "--rate", "60", "--spare", "400"]).unwrap(),
        (24, 60, 400)
    );
    assert_eq!(
        parse(&["--scale", "4", "--rate", "1", "--spare", "0"]).unwrap(),
        (4, 1, 0)
    );
}
