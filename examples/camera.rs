//! A camera that is not there: shows the reader frames of the page as a model of a camera sees them, matrix
//! after matrix from few tiles to many, and says how many bytes of a file a second arrive.
//!
//! ```sh
//! cargo run --release --example camera                      # a sharp camera, the wide matrix, every scale
//! cargo run --release --example camera -- --camera webcam   # a webcam as one was measured
//! cargo run --release --example camera -- --scales 14,16,18 --frames 40
//! cargo run --release --example camera -- --known           # the reading of the tiles alone
//! ```
//!
//! The matrix fills a frame of 1920x1080, a little turned. What the model takes into account: the lens and
//! the area of a pixel, which blur; the channels of the sensor, which take in some of each other's light;
//! noise; and the packing of the frame, which keeps colour at half the sharpness of light and smooths it.
//! A lens that bends the picture is taken into account too, as `--bend` says how much. What the model leaves
//! out: the grid of the screen against that of the sensor, frames caught while the page changes, and light
//! on the screen. So the numbers are for holding one way
//! of showing a file against another, and no promise of what a camera on a desk does.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use clap::{Parser, ValueEnum};
use opencv::core::{self, CV_8UC3, CV_32F, CV_32FC1, Mat, Point2f, Scalar, Size, Vec3f, Vector};
use opencv::imgproc;
use opencv::prelude::*;

use qqcode::packet::{Frame, OVERHEAD, Origin};
use qqcode::scan::Scanner;
use qqcode::sight;
use qqcode::tiles::{self, Grid, PITCH, SCALES, Shape};

/// The frames of the camera.
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
/// How much of the frame the matrix takes up, along the side that it fills.
const FILL: f64 = 0.94;
/// Degrees that the camera is turned by.
const TURN: f64 = 0.8;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Model {
    /// A camera in focus with little noise
    Sharp,
    /// A webcam looking at a laptop's screen, as one was measured at ten pixels to a tile: one symbol and
    /// one colour in a hundred misread, and two thirds of the red in a cyan tile that a white one has
    Webcam,
}

/// What a camera does to the picture, lengths in pixels of its frames.
#[derive(Clone, Copy, Debug)]
struct Camera {
    /// The blur of the lens
    blur: f64,
    /// How much of each other channel a channel takes in
    crosstalk: f32,
    /// The noise of a channel, in steps of the 255 that a pixel has
    noise: f32,
    /// The blur of the two planes of colour, in the pixels that those have: half as many as the frame
    chroma_blur: f32,
}

impl Model {
    fn camera(self) -> Camera {
        match self {
            Model::Sharp => Camera {
                blur: 1.2,
                crosstalk: 0.02,
                noise: 3.0,
                chroma_blur: 1.0,
            },
            Model::Webcam => Camera {
                blur: 1.75,
                crosstalk: 0.05,
                noise: 4.0,
                chroma_blur: 1.7,
            },
        }
    }
}

#[derive(Parser, Debug)]
#[command(about = "Shows the reader frames of the page through a model of a camera")]
struct Cli {
    #[arg(long, value_enum, default_value = "sharp")]
    camera: Model,
    #[arg(long, value_enum, default_value = "wide")]
    shape: Shape,
    /// The scales to try, as `--scale` of qqcode-show has them; all of them if none is given
    #[arg(long, value_delimiter = ',')]
    scales: Vec<usize>,
    /// Frames of each scale
    #[arg(long, default_value_t = 16)]
    frames: usize,
    /// Frames a second that the page shows, each of which the camera catches once
    #[arg(long, default_value_t = 15)]
    rate: usize,
    #[arg(long, default_value_t = 12)]
    threads: usize,
    /// How much nearer to the middle of the frame its corners are than a lens that bends nothing would have
    /// them, in parts of a hundred: the barrel of a wide lens
    #[arg(long, default_value_t = 1.0)]
    bend: f64,
    /// Lays the frames flat by where the corners of the matrix are, which the model knows, and not by
    /// where the reader finds them: what is left is the reading of the tiles, and how many of them are
    /// misread is said for every frame, read or not
    #[arg(long)]
    known: bool,
}

/// Numbers in no pattern, the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Between 0 and 1.
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
    }

    /// Around 0, as noise is.
    fn noise(&mut self) -> f32 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
    }
}

/// A plane of `width` values a row, smoothed by a bell of `sigma` values.
fn smooth(plane: &[f32], width: usize, sigma: f32) -> Vec<f32> {
    if sigma < 0.05 {
        return plane.to_vec();
    }
    let height = plane.len() / width;
    let reach = (3.0 * sigma).ceil() as isize;
    let mut bell: Vec<f32> = (-reach..=reach)
        .map(|at| (-(at * at) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = bell.iter().sum();
    bell.iter_mut().for_each(|weight| *weight /= sum);
    // the edge repeats
    let along = |from: &[f32], to: &mut [f32], count: usize, step: usize, lines: usize, stride: usize| {
        for line in 0..lines {
            for at in 0..count {
                let mut value = 0.0;
                for (tap, weight) in bell.iter().enumerate() {
                    let near = (at as isize + tap as isize - reach).clamp(0, count as isize - 1) as usize;
                    value += weight * from[line * stride + near * step];
                }
                to[line * stride + at * step] = value;
            }
        }
    };
    let mut across = vec![0.0; plane.len()];
    along(plane, &mut across, width, 1, height, width);
    let mut down = vec![0.0; plane.len()];
    along(&across, &mut down, height, width, width, 1);
    down
}

/// The frame of the camera that shows the page with these tiles on it: blue, green, red for every pixel.
/// Returns it with how many of its pixels a tile is wide, and with where the middles of the four finder
/// patterns are in it.
fn shoot(
    grid: Grid,
    values: &[u8],
    camera: &Camera,
    bend: f64,
    rng: &mut Rng,
) -> Result<(Mat, f64, [Point2f; 4])> {
    let picture = tiles::render(grid, values);
    let (width, height) = (grid.width() as i32, grid.height() as i32);
    let mut page = Mat::new_rows_cols_with_default(height, width, CV_8UC3, Scalar::all(0.0))?;
    for (to, from) in page
        .data_bytes_mut()?
        .chunks_exact_mut(3)
        .zip(picture.chunks_exact(3))
    {
        to.copy_from_slice(&[from[2], from[1], from[0]]);
    }

    // The light of the page, at twice the sharpness of the camera's frames: turned a little, and off the
    // grid of the pixels by a part of one that differs from frame to frame. Around the matrix is the
    // light ground of the page.
    let fit = FILL * (WIDTH as f64 / width as f64).min(HEIGHT as f64 / height as f64);
    let middle = Point2f::new(width as f32 / 2.0, height as f32 / 2.0);
    let turn = TURN + 0.2 * (rng.unit() as f64 - 0.5);
    let mut transform = imgproc::get_rotation_matrix_2d(middle, turn, 2.0 * fit)?;
    *transform.at_2d_mut::<f64>(0, 2)? += WIDTH as f64 - middle.x as f64 + 2.0 * rng.unit() as f64;
    *transform.at_2d_mut::<f64>(1, 2)? += HEIGHT as f64 - middle.y as f64 + 2.0 * rng.unit() as f64;
    let mut fine = Mat::default();
    imgproc::warp_affine(
        &page,
        &mut fine,
        &transform,
        Size::new(2 * WIDTH as i32, 2 * HEIGHT as i32),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(223.0),
    )?;
    // The lens: what it shows at a place is what a lens that bends nothing shows further out, the more
    // so the further from the middle the place is.
    let (middle_x, middle_y) = (WIDTH as f64 - 0.5, HEIGHT as f64 - 0.5);
    let reach = (WIDTH as f64).hypot(HEIGHT as f64);
    let bend = bend / 100.0;
    if bend != 0.0 {
        let size = Size::new(2 * WIDTH as i32, 2 * HEIGHT as i32);
        let mut from_x = Mat::new_size_with_default(size, CV_32FC1, Scalar::all(0.0))?;
        let mut from_y = from_x.clone();
        let (across, down) = (from_x.data_typed_mut::<f32>()?, from_y.data_typed_mut::<f32>()?);
        for y in 0..2 * HEIGHT {
            for x in 0..2 * WIDTH {
                let (off_x, off_y) = (x as f64 - middle_x, y as f64 - middle_y);
                let out = 1.0 + bend * (off_x * off_x + off_y * off_y) / (reach * reach);
                across[y * 2 * WIDTH + x] = (middle_x + off_x * out) as f32;
                down[y * 2 * WIDTH + x] = (middle_y + off_y * out) as f32;
            }
        }
        let mut bent = Mat::default();
        imgproc::remap(
            &fine,
            &mut bent,
            &from_x,
            &from_y,
            imgproc::INTER_LINEAR,
            core::BORDER_CONSTANT,
            Scalar::all(223.0),
        )?;
        fine = bent;
    }
    let mut light = Mat::default();
    fine.convert_to(&mut light, CV_32F, 1.0 / 255.0, 0.0)?;
    // the lens, and then the pixels, each of which takes in the light of its area
    let mut blurred = Mat::default();
    imgproc::gaussian_blur_def(&light, &mut blurred, Size::new(0, 0), 2.0 * camera.blur)?;
    let mut sensor = Mat::default();
    imgproc::resize(
        &blurred,
        &mut sensor,
        Size::new(WIDTH as i32, HEIGHT as i32),
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;
    ensure!(sensor.is_continuous(), "a frame in one piece");

    // What the camera makes of the light: channels that take in some of each other, the values of a
    // picture, which are not in proportion to the light, and noise.
    let mut frame = vec![0f32; 3 * WIDTH * HEIGHT];
    for (pixel, light) in frame.chunks_exact_mut(3).zip(sensor.data_typed::<Vec3f>()?) {
        let sum = light[0] + light[1] + light[2];
        for channel in 0..3 {
            let mixed = (light[channel] + camera.crosstalk * (sum - light[channel]))
                / (1.0 + 2.0 * camera.crosstalk)
                * 0.9;
            let value = 255.0 * mixed.max(0.0).powf(1.0 / 2.2) + camera.noise * rng.noise();
            pixel[channel] = value.clamp(0.0, 255.0).round();
        }
    }

    // The frame as a camera hands it out: its light at full sharpness, its colour in two planes of half
    // as many pixels either way, smoothed, as a frame of Motion JPEG or of H.264 has them.
    let (half_width, half_height) = (WIDTH / 2, HEIGHT / 2);
    let mut luma = vec![0f32; WIDTH * HEIGHT];
    let mut blue = vec![0f32; half_width * half_height];
    let mut red = vec![0f32; half_width * half_height];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let pixel = &frame[3 * (y * WIDTH + x)..][..3];
            let light = 0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2];
            luma[y * WIDTH + x] = light.round();
            let at = (y / 2) * half_width + x / 2;
            blue[at] += (128.0 + 0.564 * (pixel[0] - light)) / 4.0;
            red[at] += (128.0 + 0.713 * (pixel[2] - light)) / 4.0;
        }
    }
    let round = |plane: Vec<f32>| plane.into_iter().map(f32::round).collect::<Vec<_>>();
    let blue = smooth(&round(blue), half_width, camera.chroma_blur);
    let red = smooth(&round(red), half_width, camera.chroma_blur);
    let between = |plane: &[f32], x: usize, y: usize| {
        // a value of colour is in the middle of the four pixels it is of
        let place = |at: usize, count: usize| {
            let at = ((at as f32 - 0.5) / 2.0).clamp(0.0, count as f32 - 1.0);
            let first = (at.floor() as usize).min(count - 2);
            (first, at - first as f32)
        };
        let ((left, across), (top, down)) = (place(x, half_width), place(y, half_height));
        let value = |x: usize, y: usize| plane[y * half_width + x];
        (value(left, top) * (1.0 - across) + value(left + 1, top) * across) * (1.0 - down)
            + (value(left, top + 1) * (1.0 - across) + value(left + 1, top + 1) * across) * down
    };
    let mut seen = Mat::new_rows_cols_with_default(HEIGHT as i32, WIDTH as i32, CV_8UC3, Scalar::all(0.0))?;
    for (at, pixel) in seen.data_bytes_mut()?.chunks_exact_mut(3).enumerate() {
        let (x, y) = (at % WIDTH, at / WIDTH);
        let (light, blue, red) = (
            luma[at],
            between(&blue, x, y) - 128.0,
            between(&red, x, y) - 128.0,
        );
        let values = [
            light + 1.772 * blue,
            light - 0.344 * blue - 0.714 * red,
            light + 1.402 * red,
        ];
        for channel in 0..3 {
            pixel[channel] = values[channel].clamp(0.0, 255.0).round() as u8;
        }
    }
    // a pixel of the frame is two of the finer picture
    let at = |row: i32, column: i32| transform.at_2d::<f64>(row, column).copied();
    let mut corners = [Point2f::default(); 4];
    for (corner, (x, y)) in corners.iter_mut().zip(grid.anchors()) {
        let across = at(0, 0)? * x as f64 + at(0, 1)? * y as f64 + at(0, 2)? - middle_x;
        let down = at(1, 0)? * x as f64 + at(1, 1)? * y as f64 + at(1, 2)? - middle_y;
        // where the lens puts what would be this far from the middle: that place is the one whose
        // light comes from here
        let (mut bent_x, mut bent_y) = (across, down);
        for _ in 0..8 {
            let out = 1.0 + bend * (bent_x * bent_x + bent_y * bent_y) / (reach * reach);
            (bent_x, bent_y) = (across / out, down / out);
        }
        *corner = Point2f::new(
            ((middle_x + bent_x - 0.5) / 2.0) as f32,
            ((middle_y + bent_y - 0.5) / 2.0) as f32,
        );
    }
    Ok((seen, PITCH as f64 * fit, corners))
}

/// The tiles of a frame whose finder patterns are at `corners`, as the reader takes them to be.
fn read_known(seen: &Mat, grid: Grid, corners: &[Point2f; 4]) -> Result<Vec<u8>> {
    let own = grid.anchors().map(|(x, y)| Point2f::new(x, y));
    let transform = imgproc::get_perspective_transform_def(
        &Vector::<Point2f>::from_slice(corners),
        &Vector::<Point2f>::from_slice(&own),
    )?;
    let mut flat = Mat::default();
    imgproc::warp_perspective(
        seen,
        &mut flat,
        &transform,
        Size::new(grid.width() as i32, grid.height() as i32),
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::all(0.0),
    )?;
    Ok(sight::read_tiles(flat.data_bytes()?, grid))
}

/// What became of the frames of one matrix.
#[derive(Default)]
struct Tally {
    located: usize,
    read: usize,
    wrong: f32,
    /// Of all the tiles of all the frames: symbols and colours that are misread
    symbols: usize,
    colours: usize,
    tiles: usize,
    /// How long the reading of the tiles took, of all the frames
    reading: Duration,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let camera = cli.camera.camera();
    let scales = if cli.scales.is_empty() {
        SCALES.collect()
    } else {
        cli.scales.clone()
    };
    println!(
        "camera: {:?}, {WIDTH}x{HEIGHT}, a lens that bends by {} in a hundred; page: {:?}, {} frames a second; {} \
         frames of each scale",
        cli.camera, cli.bend, cli.shape, cli.rate, cli.frames
    );
    let origin = Origin {
        checksum: 0x5eed_f00d,
        length: 1 << 30,
    };
    let mut best = (0, 0);
    for scale in scales {
        ensure!(
            SCALES.contains(&scale),
            "a matrix has the scales 4 to 24, not {scale}"
        );
        let grid = Grid::new(cli.shape, scale);
        let packet = grid.capacity() - OVERHEAD;
        let scanner = Scanner::default();
        let (next, tally, pixels) = (AtomicUsize::new(0), Mutex::new(Tally::default()), Mutex::new(0.0));
        std::thread::scope(|scope| -> Result<()> {
            let workers: Vec<_> = (0..cli.threads.max(1))
                .map(|_| {
                    scope.spawn(|| -> Result<()> {
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= cli.frames {
                                return Ok(());
                            }
                            let mut rng =
                                Rng(0x9e37_79b9_7f4a_7c15 ^ (((scale as u64) << 32) | (index as u64 + 1)));
                            let frame = Frame::Packet {
                                origin,
                                block: 0,
                                sequence: index as u32,
                                bytes: (0..packet).map(|_| (rng.next() >> 32) as u8).collect(),
                            };
                            let values = tiles::pack(grid, &frame.encode(grid.capacity()));
                            let (seen, tile, corners) = shoot(grid, &values, &camera, cli.bend, &mut rng)?;
                            *pixels.lock().unwrap_or_else(PoisonError::into_inner) = tile;
                            if cli.known {
                                let begun = Instant::now();
                                let taken = read_known(&seen, grid, &corners)?;
                                let reading = begun.elapsed();
                                let unpacked = tiles::unpack(grid, &taken);
                                let mut tally = tally.lock().unwrap_or_else(PoisonError::into_inner);
                                for (taken, shown) in taken.iter().zip(&values) {
                                    tally.symbols += usize::from(taken >> 2 != shown >> 2);
                                    tally.colours += usize::from(taken & 3 != shown & 3);
                                }
                                tally.tiles += values.len();
                                tally.reading += reading;
                                tally.located += 1;
                                let read = unpacked.filter(|unpacked| {
                                    Frame::parse(&unpacked.payload).as_ref() == Some(&frame)
                                });
                                if let Some(unpacked) = read {
                                    tally.read += 1;
                                    tally.wrong += unpacked.wrong as f32 / unpacked.bytes as f32;
                                }
                                continue;
                            }
                            let found = scanner.scan(&seen)?;
                            let mut tally = tally.lock().unwrap_or_else(PoisonError::into_inner);
                            tally.located += usize::from(!found.is_empty());
                            if let Some(read) =
                                found.iter().find(|found| found.frame.as_ref() == Some(&frame))
                            {
                                tally.read += 1;
                                tally.wrong += read.wrong;
                            }
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().expect("a worker that ends")?;
            }
            Ok(())
        })?;
        let tally = tally.into_inner().unwrap_or_else(PoisonError::into_inner);
        let rate = packet * cli.rate * tally.read / cli.frames;
        if rate > best.0 {
            best = (rate, scale);
        }
        let misread = if cli.known {
            format!(
                " | misread: {:4.1} % of the symbols, {:4.1} % of the colours | {:3} ms to lay a frame flat and read it",
                100.0 * tally.symbols as f32 / tally.tiles.max(1) as f32,
                100.0 * tally.colours as f32 / tally.tiles.max(1) as f32,
                tally.reading.as_millis() / cli.frames.max(1) as u128
            )
        } else {
            String::new()
        };
        println!(
            "scale {scale:2} | {:3}x{:<3} tiles | {:4.1} px a tile | {packet:5} bytes a frame | found {:3}, read {:3} \
             of {} | {:4.1} % put right | {:4} kB a second{misread}",
            grid.columns,
            grid.rows,
            pixels.into_inner().unwrap_or_else(PoisonError::into_inner),
            tally.located,
            tally.read,
            cli.frames,
            100.0 * tally.wrong / tally.read.max(1) as f32,
            rate / 1000,
        );
    }
    println!("best: {} kB a second at scale {}", best.0 / 1000, best.1);
    Ok(())
}
