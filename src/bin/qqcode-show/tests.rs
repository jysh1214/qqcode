use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use qqcode::assembly::Assembly;

use super::*;

/// The app that the command line `arguments` starts for a file of `bytes`.
fn app(bytes: &[u8], arguments: &[&str]) -> App {
    // tests run side by side, each needs a file of its own
    static FILES: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "qqcode-show-test-{}-{}.bin",
        std::process::id(),
        FILES.fetch_add(1, Ordering::Relaxed)
    ));
    fs::File::create(&path)
        .and_then(|mut file| file.write_all(bytes))
        .unwrap();
    let mut command = vec!["qqcode-show", path.to_str().unwrap()];
    command.extend(arguments);
    let app = App::new(Cli::try_parse_from(command).unwrap());
    fs::remove_file(&path).unwrap();
    app.unwrap()
}

/// Bytes of every value, none of them in a pattern that a piece could be told by.
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

fn press(app: &mut App, key: char) -> Reaction {
    let reaction = app.on_key(KeyEvent::new(KeyCode::Char(key), KeyModifiers::NONE));
    if reaction == Reaction::Redraw {
        app.lay_out(80, 24);
    }
    reaction
}

fn layout(app: &App) -> &Layout {
    app.layout.as_ref().expect("a terminal with room for a code")
}

/// Where the code on show begins in the file.
fn offset(app: &App) -> usize {
    app.piece(layout(app)).start
}

const SIZE: Size = Size {
    columns: 80,
    rows: 24,
    cell: None,
};

fn texts(app: &App) -> String {
    let runs = app.screen(&SIZE).runs;
    runs.into_iter().map(|run| run.text).collect::<Vec<_>>().join(" ")
}

#[test]
fn file_is_cut_to_what_a_code_holds() {
    // an 80x24 terminal has room for version 5, which holds 84 bytes at level M: 69 of the file
    let mut app = app(&bytes(200), &[]);
    app.lay_out(80, 24);
    let layout = layout(&app);
    assert_eq!((layout.geometry.version, layout.piece, layout.codes), (5, 69, 3));
    assert_eq!(app.stage, Stage::Ready);

    let mut low = self::app(&bytes(200), &["--level", "l", "--max-version", "3"]);
    low.lay_out(80, 24);
    let layout = self::layout(&low);
    // 53 bytes less the 15 that say where a piece goes
    assert_eq!((layout.geometry.version, layout.piece, layout.codes), (3, 38, 6));
}

#[test]
fn keys_turn_the_codes_from_the_start_screen_to_the_end_screen() {
    let mut app = app(&bytes(200), &[]);
    app.lay_out(80, 24);
    // nothing comes before the start screen
    press(&mut app, 'p');
    assert_eq!(app.stage, Stage::Ready);
    assert_eq!(
        app.on_key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        Reaction::Redraw
    );
    assert_eq!((app.stage, offset(&app)), (Stage::Codes, 0));
    press(&mut app, ' ');
    assert_eq!(offset(&app), 69);
    press(&mut app, 'n');
    assert_eq!(app.piece(layout(&app)), 138..200);
    press(&mut app, 'p');
    assert_eq!(offset(&app), 69);
    press(&mut app, 'G');
    assert_eq!(offset(&app), 138);
    // after the last code comes the end screen, and after that nothing
    press(&mut app, 'n');
    assert_eq!(app.stage, Stage::Done);
    press(&mut app, 'n');
    assert_eq!(app.stage, Stage::Done);
    press(&mut app, 'p');
    assert_eq!((app.stage, offset(&app)), (Stage::Codes, 138));
    press(&mut app, 'g');
    assert_eq!(offset(&app), 0);
    assert_eq!(press(&mut app, 'x'), Reaction::Nothing);
    assert_eq!(press(&mut app, 'q'), Reaction::Quit);
}

#[test]
fn every_screen_has_the_code_that_says_what_it_is() {
    let file = bytes(200);
    let mut app = app(&file, &[]);
    app.lay_out(80, 24);
    let id = FileId::of(&file).unwrap();
    // the code as the screen has it: version 5 from the sixth dot on, a dot a module
    let shown = |app: &App| {
        let screen = app.screen(&SIZE);
        let message = app.message(layout(app));
        let code = code::encode(&message.encode(), 5, Level::M).unwrap();
        for row in 0..37 {
            for column in 0..37 {
                let dark = screen.dot(5 + column, 5 + row) == Rgb::BLACK;
                assert_eq!(dark, code.is_dark(row, column), "module {row},{column}");
            }
        }
        message
    };

    let Message::Start { file: named, name } = shown(&app) else {
        panic!("a start message");
    };
    assert_eq!(named, id);
    assert!(
        name.starts_with("qqcode-show-test-") && name.ends_with(".bin"),
        "{name}"
    );
    let start = texts(&app);
    assert!(
        start.contains("200 bytes in 3 codes, a new one every 0.5 s.") && start.contains("Enter starts."),
        "{start}"
    );
    assert!(start.contains("start, 3 codes to come"), "{start}");

    let mut assembly = Assembly::new(id);
    for code in 0..3 {
        press(&mut app, ' ');
        let Message::Piece {
            file: named,
            offset,
            bytes,
        } = shown(&app)
        else {
            panic!("a piece");
        };
        assert_eq!((named, offset as usize), (id, 69 * code));
        assembly.add(offset as usize, &bytes);
    }
    assert!(texts(&app).contains("code 3/3, bytes 138 to 199 of 200"));
    assert_eq!(assembly.into_bytes(), Some(file));

    press(&mut app, ' ');
    assert_eq!(shown(&app), Message::End { file: id });
    let end = texts(&app);
    assert!(
        end.contains("End of") && end.contains("end, 3 codes shown"),
        "{end}"
    );
}

#[test]
fn start_screen_has_as_much_of_the_name_as_its_code_holds() {
    let mut app = app(&bytes(10), &["--level", "h", "--max-version", "3"]);
    app.lay_out(80, 24);
    // version 3 holds 24 bytes at level H, 11 of which say what the message is
    let Message::Start { name, .. } = app.message(layout(&app)) else {
        panic!("a start message");
    };
    assert_eq!(name, "qqcode-show-t");
}

#[test]
fn codes_turn_by_key_alone_if_asked() {
    let mut app = app(&bytes(200), &["--interval", "0"]);
    app.lay_out(80, 24);
    assert_eq!(app.interval, None);
    assert!(texts(&app).contains("200 bytes in 3 codes."));
    // and no key gives them a pace
    press(&mut app, 'f');
    press(&mut app, 's');
    assert_eq!(app.interval, None);
    assert!(app.command_line().ends_with("--interval 0"));
}

#[test]
fn file_without_a_byte_goes_from_the_start_to_the_end() {
    let mut app = app(&[], &[]);
    app.lay_out(80, 24);
    assert_eq!(layout(&app).codes, 0);
    assert!(texts(&app).contains("0 bytes in 0 codes"));
    let empty = FileId {
        checksum: 0,
        length: 0,
    };
    assert!(matches!(app.message(layout(&app)), Message::Start { file, .. } if file == empty));
    press(&mut app, ' ');
    assert_eq!(app.stage, Stage::Done);
    assert_eq!(app.message(layout(&app)), Message::End { file: empty });
    press(&mut app, 'p');
    press(&mut app, 'g');
    assert_eq!(app.stage, Stage::Done);
}

#[test]
fn code_on_show_survives_a_new_layout() {
    let mut app = app(&bytes(5000), &[]);
    app.lay_out(80, 24);
    // from the start screen to the eleventh code
    for _ in 0..11 {
        press(&mut app, ' ');
    }
    assert_eq!(offset(&app), 690);
    // a larger terminal, a smaller one, larger modules: the byte 690 stays on show
    for (columns, rows) in [(200, 60), (70, 24), (300, 100)] {
        app.lay_out(columns, rows);
        let piece = app.piece(layout(&app));
        assert!(piece.contains(&690), "{columns}x{rows}: {piece:?}");
    }
    app.lay_out(80, 24);
    assert_eq!(offset(&app), 690);
    press(&mut app, 'e');
    assert_eq!(app.level, Level::Q);
    assert!(app.piece(layout(&app)).contains(&690));
    // a turn makes the code on show the one to come back to
    app.lay_out(200, 60);
    let wide = offset(&app);
    press(&mut app, 'n');
    assert!(offset(&app) > 690 && app.piece(layout(&app)).contains(&app.anchor));
    assert!(wide <= 690);
}

#[test]
fn keys_change_the_layout_within_limits() {
    let mut app = app(&bytes(200), &["--module", "15", "--margin", "1"]);
    app.lay_out(80, 24);
    for _ in 0..3 {
        press(&mut app, '+');
        press(&mut app, '{');
    }
    assert_eq!((app.settings.module, app.settings.margin), (16, 0));
    // modules of 16 columns leave room for three of them: there is a notice instead of the chart
    assert_eq!(app.layout, None);
    assert!(app.screen(&SIZE).runs[0].text.contains("80x24 cells are too few"));
    // and no way off the start screen
    press(&mut app, ' ');
    assert_eq!(app.stage, Stage::Ready);
    for _ in 0..15 {
        press(&mut app, '-');
    }
    assert_eq!(app.settings.module, 1);
    // without a margin there are 48 modules to a side, room for version 7 with its 45
    assert_eq!(layout(&app).geometry.version, 7);
    for _ in 0..20 {
        press(&mut app, '}');
    }
    assert_eq!(app.settings.margin, 16);
    assert_eq!(app.layout, None);
}

#[test]
fn keys_make_the_codes_smaller_and_larger() {
    let mut app = app(&bytes(200), &[]);
    app.lay_out(80, 24);
    assert_eq!((layout(&app).geometry.version, app.settings.max_version), (5, 40));
    // one smaller than they are, not one smaller than they may be
    press(&mut app, ',');
    assert_eq!((layout(&app).geometry.version, app.settings.max_version), (4, 4));
    press(&mut app, ',');
    assert_eq!(layout(&app).geometry.version, 3);
    press(&mut app, '.');
    press(&mut app, '.');
    press(&mut app, '.');
    // as large as there is room for
    assert_eq!((layout(&app).geometry.version, app.settings.max_version), (5, 6));
    for _ in 0..10 {
        press(&mut app, ',');
    }
    // version 1 holds 14 bytes at level M, which is less than a piece takes without any of the file
    assert_eq!((app.settings.max_version, code::capacity(1, Level::M)), (1, 14));
    assert_eq!(app.layout, None);
    // and 17 at level L, the one after H
    press(&mut app, 'e');
    press(&mut app, 'e');
    assert_eq!(app.layout, None);
    press(&mut app, 'e');
    assert_eq!((app.level, layout(&app).piece), (Level::L, 2));
}

#[test]
fn keys_step_through_the_paces() {
    let mut app = app(&bytes(200), &["--interval", "0.4"]);
    app.lay_out(80, 24);
    press(&mut app, 's');
    assert_eq!(app.interval, Some(0.5));
    press(&mut app, 'f');
    press(&mut app, 'f');
    assert_eq!(app.interval, Some(0.2));
    for _ in 0..10 {
        press(&mut app, 'f');
    }
    assert_eq!(app.interval, Some(0.05));
    for _ in 0..20 {
        press(&mut app, 's');
    }
    assert_eq!(app.interval, Some(5.0));
    assert!(texts(&app).contains("a new one every 5 s"));
}

#[test]
fn command_line_repeats_the_layout() {
    let mut app = app(&bytes(200), &["--paper", "#C0c0C0", "--interval", "0.3"]);
    app.lay_out(80, 24);
    for key in ['{', '{', 'e', ',', 'h', 's'] {
        press(&mut app, key);
    }
    let line = app.command_line();
    let options = line.split_once(".bin ").expect("the file comes first").1;
    assert_eq!(
        options,
        "--module 1 --margin 2 --level q --max-version 5 --interval 0.5 --paper c0c0c0 --no-notes"
    );
    // and it parses back into the same settings
    let again = Cli::try_parse_from(line.split(' ')).unwrap();
    assert_eq!(
        (again.margin, again.level, again.max_version, again.paper),
        (2, Level::Q, 5, Rgb(192, 192, 192))
    );
    assert!(again.no_notes && again.interval == 0.5);
}

#[test]
fn notes_tell_the_code_and_the_sizes() {
    let mut app = app(&bytes(200), &[]);
    app.lay_out(80, 24);
    press(&mut app, ' ');
    press(&mut app, ' ');
    let size = Size {
        columns: 80,
        rows: 24,
        cell: Some((9.0, 19.5)),
    };
    assert_eq!(
        app.notes(layout(&app), &size)[..3],
        [
            "code 2/3, bytes 69 to 137 of 200",
            "version 5, level m, 69 bytes a code, a new one every 0.5 s",
            "module 1, margin 4, code 37x18.5 cells = 333x361 px"
        ]
    );
    // hidden notes leave only the code on the screen
    press(&mut app, 'h');
    assert!(app.screen(&size).runs.is_empty());
    press(&mut app, '?');
    assert!(!app.screen(&size).runs.is_empty());
    assert_eq!(
        app.measurements(&size),
        "terminal 80x24 cells, cell 9x19.5 px, version 5, code 37x18.5 cells = 333x361 px, 69 bytes a code, 3 codes"
    );
    app.lay_out(20, 5);
    assert_eq!(
        app.measurements(&SIZE),
        "terminal 80x24 cells, too small for a code"
    );
}

#[test]
fn options_are_checked() {
    let start = |arguments: &[&str]| {
        let mut command = vec!["qqcode-show", "Cargo.toml"];
        command.extend(arguments);
        Cli::try_parse_from(command)
            .map_err(anyhow::Error::from)
            .and_then(App::new)
    };
    assert!(start(&[]).is_ok());
    assert!(start(&["--level", "h", "--max-version", "40", "--interval", "0"]).is_ok());
    for wrong in [
        &["--module", "0"][..],
        &["--module", "17"],
        &["--margin", "17"],
        &["--level", "x"],
        &["--max-version", "0"],
        &["--max-version", "41"],
        &["--ink", "ffffff", "--paper", "000000"],
        &["--paper", "white"],
        &["--interval", "-1"],
        &["--interval", "nan"],
    ] {
        assert!(start(wrong).is_err(), "{wrong:?}");
    }
    let missing = Cli::try_parse_from(["qqcode-show", "no such file"]).unwrap();
    let error = App::new(missing).err().unwrap().to_string();
    assert!(error.contains("can't read no such file"), "{error}");
    assert_eq!(shell_word("notes/page 1.txt"), "'notes/page 1.txt'");
    assert_eq!(shell_word("it's"), r"'it'\''s'");
    assert_eq!(shell_word("-o"), "'-o'");
    assert_eq!(
        (count(1, "code"), count(0, "byte")),
        ("1 code".to_owned(), "0 bytes".to_owned())
    );
}

/// What `qqcode` makes of the screens, as pictures of them.
#[cfg(feature = "reader")]
// a file may well be missing in one part only
#[allow(clippy::single_range_in_vec_init)]
mod camera {
    use anyhow::Result;
    use opencv::core::{self, CV_8UC3, Mat, Point, Point2f, Rect, Scalar, Size as Pixels, Vector};
    use opencv::imgproc;
    use opencv::prelude::*;
    use qqcode::{Kind, Outcome, Pipeline};
    use unicode_width::UnicodeWidthStr;

    use super::*;

    /// The screen as an image, with character cells of `cell` pixels (width, height). Text is drawn in a
    /// Hershey font of about the size a terminal's font has in such a cell.
    fn rasterize(screen: &Screen, cell: (i32, i32)) -> Result<Mat> {
        let (width, height) = cell;
        assert_eq!(height % 2, 0, "a dot is half a cell high");
        let size = Pixels::new(screen.columns as i32 * width, screen.rows as i32 * height);
        let mut image = Mat::new_size_with_default(size, CV_8UC3, Scalar::all(0.0))?;
        let bgr = |Rgb(r, g, b): Rgb| Scalar::new(b as f64, g as f64, r as f64, 0.0);
        for y in 0..2 * screen.rows {
            for x in 0..screen.columns {
                let dot = Rect::new(x as i32 * width, y as i32 * height / 2, width, height / 2);
                let colour = bgr(screen.dot(x, y));
                imgproc::rectangle(&mut image, dot, colour, imgproc::FILLED, imgproc::LINE_8, 0)?;
            }
        }
        for run in &screen.runs {
            let font = imgproc::FONT_HERSHEY_DUPLEX;
            // as large as the row allows, but no wider than the cells of the run
            let mut scale = height as f64 / 40.0;
            let drawn = imgproc::get_text_size(&run.text, font, scale, 1, &mut 0)?.width;
            let room = run.text.width() as i32 * width;
            if drawn > room {
                scale *= room as f64 / drawn as f64;
            }
            let baseline = (run.row as i32 + 1) * height - height / 4;
            imgproc::put_text(
                &mut image,
                &run.text,
                Point::new(run.column as i32 * width, baseline),
                font,
                scale,
                bgr(run.colour),
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
        Ok(image)
    }

    /// A camera frame that shows the screen in perspective in a dim room, a little out of focus.
    fn photograph(screen: &Mat, blur: f64) -> Result<Mat> {
        let size = screen.size()?;
        let (w, h) = ((size.width - 1) as f32, (size.height - 1) as f32);
        let flat = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)].map(|(x, y)| Point2f::new(x, y));
        let seen = [(210.0, 130.0), (1650.0, 90.0), (1740.0, 960.0), (150.0, 900.0)]
            .map(|(x, y)| Point2f::new(x, y));
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

    /// A camera frame that shows the screen `scale` times as large as it is, a little turned.
    fn enlarge(screen: &Mat, scale: f64, blur: f64) -> Result<Mat> {
        let size = screen.size()?;
        let middle = Point2f::new(size.width as f32 / 2.0, size.height as f32 / 2.0);
        let mut transform = imgproc::get_rotation_matrix_2d(middle, 2.0, scale)?;
        // from the middle of the screen to the middle of the frame
        *transform.at_2d_mut::<f64>(0, 2)? += 960.0 - middle.x as f64;
        *transform.at_2d_mut::<f64>(1, 2)? += 540.0 - middle.y as f64;
        let mut frame = Mat::default();
        imgproc::warp_affine(
            screen,
            &mut frame,
            &transform,
            Pixels::new(1920, 1080),
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(40.0),
        )?;
        let mut blurred = Mat::default();
        imgproc::gaussian_blur_def(&frame, &mut blurred, Pixels::new(0, 0), blur)?;
        Ok(blurred)
    }

    /// The frame that a camera takes of the terminal of `columns` by `rows` as the app has it now.
    fn frame(app: &mut App, columns: usize, rows: usize, cell: (i32, i32), blur: f64) -> Result<Mat> {
        app.lay_out(columns, rows);
        let size = Size {
            columns,
            rows,
            cell: None,
        };
        photograph(&rasterize(&app.screen(&size), cell)?, blur)
    }

    #[test]
    fn show_is_read_from_the_start_screen_to_the_whole_file() -> Result<()> {
        let file = bytes(3000);
        let mut app = app(&file, &[]);
        let mut pipeline = Pipeline::new();
        // 160x50 cells of 6x12 pixels: a code of version 18 that holds 545 bytes of the file
        let mut take = |app: &mut App| pipeline.process(&frame(app, 160, 50, (6, 12), 1.0)?);

        assert_eq!(take(&mut app)?, Outcome::Ready);
        assert_eq!((layout(&app).geometry.version, layout(&app).codes), (18, 6));
        for code in 0..5 {
            app.forward();
            let new = Outcome::Piece { new: 545, skipped: 0 };
            assert_eq!(take(&mut app)?, new, "code {code}");
            // the same code in the next frame brings nothing
            let again = Outcome::Piece { new: 0, skipped: 0 };
            assert_eq!(take(&mut app)?, again);
        }
        app.forward();
        assert_eq!(take(&mut app)?, Outcome::Complete);
        assert_eq!(pipeline.take_file(), Some(file));
        Ok(())
    }

    #[test]
    fn code_in_view_is_surveyed() -> Result<()> {
        let mut app = app(&bytes(3000), &[]);
        let mut pipeline = Pipeline::new();
        pipeline.process(&frame(&mut app, 160, 50, (6, 12), 0.0)?)?;
        let survey = pipeline.last_survey().clone();
        let [code] = &survey.codes[..] else {
            panic!("one code, not {}", survey.codes.len());
        };
        assert_eq!((code.kind, code.modules), (Kind::Start, 89));
        // the screen is 960 pixels wide and comes out some 1500 wide: a dot of 6 pixels as 9 or so
        assert!((8.5..10.0).contains(&code.module_size), "{}", code.module_size);
        // the code is at the left of the screen, its corners in the order top-left, top-right,
        // bottom-right, bottom-left
        let [top_left, top_right, bottom_right, bottom_left] = code.outline;
        assert!(top_left.x < 400.0 && top_left.y < 300.0, "{top_left:?}");
        assert!(top_right.x > 1000.0 && top_right.y < 300.0, "{top_right:?}");
        assert!(
            bottom_right.x > 1000.0 && bottom_right.y > 800.0,
            "{bottom_right:?}"
        );
        assert!(bottom_left.x < 400.0 && bottom_left.y > 800.0, "{bottom_left:?}");
        assert_eq!(pipeline.name().map(|name| name.ends_with(".bin")), Some(true));
        Ok(())
    }

    #[test]
    fn show_that_is_laid_out_anew_half_way_fills_the_same_file() -> Result<()> {
        let file = bytes(3000);
        let mut app = app(&file, &[]);
        let mut pipeline = Pipeline::new();
        assert_eq!(
            pipeline.process(&frame(&mut app, 160, 50, (6, 12), 0.0)?)?,
            Outcome::Ready
        );
        app.forward();
        app.forward();
        // the second code of six, read from a terminal of this size
        let second = pipeline.process(&frame(&mut app, 160, 50, (6, 12), 0.0)?)?;
        assert_eq!(
            second,
            Outcome::Piece {
                new: 545,
                skipped: 545
            }
        );
        // The window is made smaller: the codes are of version 10 now, with 198 bytes of the file each,
        // and the one on show is the one with the byte that the second code began with, the third.
        let mut read = 545;
        let mut outcome = Outcome::NoCode;
        for _ in 0..40 {
            outcome = pipeline.process(&frame(&mut app, 120, 34, (8, 16), 0.0)?)?;
            assert_eq!(layout(&app).geometry.version, 10);
            if let Outcome::Piece { new, .. } = outcome {
                read += new;
            } else {
                break;
            }
            app.forward();
        }
        // the end screen, with what the first two of the small codes hold still missing
        assert_eq!(outcome, Outcome::Done);
        let missing = pipeline.assembly().unwrap().missing();
        assert_eq!((read, missing), (3000 - 396, vec![0..396]));
        // from the beginning again: the second code completes the file
        app.turn_to(0);
        let first = pipeline.process(&frame(&mut app, 120, 34, (8, 16), 0.0)?)?;
        assert_eq!(first, Outcome::Piece { new: 198, skipped: 0 });
        app.forward();
        assert_eq!(
            pipeline.process(&frame(&mut app, 120, 34, (8, 16), 0.0)?)?,
            Outcome::Complete
        );
        assert_eq!(pipeline.take_file(), Some(file));
        Ok(())
    }

    #[test]
    fn largest_code_is_read_at_three_pixels_a_module() -> Result<()> {
        let file = bytes(5000);
        let mut app = app(&file, &["--level", "l"]);
        let mut pipeline = Pipeline::new();
        // 420x100 cells of 2x4 pixels: version 40 with 177 modules of 2 pixels, seen one and a half
        // times as large and out of focus
        let mut take = |app: &mut App| {
            app.lay_out(420, 100);
            let size = Size {
                columns: 420,
                rows: 100,
                cell: None,
            };
            pipeline.process(&enlarge(&rasterize(&app.screen(&size), (2, 4))?, 1.5, 1.0)?)
        };
        assert_eq!(take(&mut app)?, Outcome::Ready);
        assert_eq!((layout(&app).geometry.version, layout(&app).piece), (40, 2938));
        app.forward();
        let first = Outcome::Piece {
            new: 2938,
            skipped: 0,
        };
        assert_eq!(take(&mut app)?, first);
        app.forward();
        assert_eq!(take(&mut app)?, Outcome::Complete);
        let size = pipeline.last_survey().codes[0].module_size;
        assert!((2.9..3.1).contains(&size), "{size}");
        assert_eq!(pipeline.take_file(), Some(file));
        Ok(())
    }
}
