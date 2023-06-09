use std::{f64::consts::PI, fs::File, io::BufWriter, path::Path, vec};

type Pixel = i16;

const LUM_FACTOR_RED: f64 = 0.2126;
const LUM_FACTOR_GREEN: f64 = 0.7152;
const LUM_FACTOR_BLUE: f64 = 0.0722;
const CONVOLUTE_GN: isize = 3;
const MAX_BRIGHTNESS_F64: f64 = 255.00;
const MAX_BRIGHTNESS_I16: Pixel = 255;
const HALF_BRIGHTNESS: u8 = 127;
const MAX_GRID_SIZE: usize = 100;
const MIN_GRID_SIZE: usize = 20;
const NMS_ARCTANS: [f64; 4] = [1.0, 3.0, 5.0, 7.0];
const KERNEL_GX: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
const KERNEL_GY: [f64; 9] = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0];

macro_rules! err_out_res {
    ($result: expr, $message: literal) => {
        match $result {
            Ok(fruit) => fruit,
            Err(err) => {
                eprintln!("\x1b[1;31mError occured\x1b[0m");
                eprintln!($message);
                eprintln!("{}", err);
                std::process::exit(1);
            }
        }
    };
}

macro_rules! err_out_if_false {
    ($cond: expr, $message: literal, $($varargs: literal),+) => {
        if !$cond {
            eprintln!("\x1b[1;31mError occured\x1b[0m");
            eprintln!($message, $($varargs),+);
            std::process::exit(1);
        }
    };
    ($cond: expr, $message: literal, $($varargs: expr),+) => {
        if !$cond {
            eprintln!("\x1b[1;31mError occured\x1b[0m");
            eprintln!($message, $($varargs),+);
            std::process::exit(1);
        }
    };
    ($cond: expr, $message: literal, $($varargs: ident),+) => {
        if !$cond {
            eprintln!("\x1b[1;31mError occured\x1b[0m");
            eprintln!($message, $($varargs),+);
            std::process::exit(1);
        }
    };
    ($cond: expr, $message: literal) => {
        if !$cond {
            eprintln!("\x1b[1;31mError occured\x1b[0m");
            eprintln!($message);
            std::process::exit(1);
        }
    }
}

macro_rules! to_luma {
    ($r: expr, $g: expr, $b: expr) => {
        (($r as f64 * LUM_FACTOR_RED)
            + ($b as f64 * LUM_FACTOR_BLUE)
            + ($b as f64 * LUM_FACTOR_GREEN)) as u8
    };
}

macro_rules! thresh_factor {
    ($pixel: expr) => {
        ($pixel & (!((i8::MAX as u8) & u8::MAX) & u8::MAX))
    };
    ($pixel: ident) => {
        ($pixel & (!((i8::MAX as u8) & u8::MAX) & u8::MAX))
    };
}

macro_rules! rotl_one {
    ($factor: expr) => {
        (($factor << 1) | ($factor >> (u8::BITS - 1)))
    };
    ($factor: ident) => {
        (($factor << 1) | ($factor >> (u8::BITS - 1)))
    };
}

macro_rules! threshold {
    ($pixel: expr) => {{
        (thresh_factor!($pixel) | (rotl_one!(thresh_factor!($pixel)) * u8::MAX))
    }};
}

macro_rules! get_2d_index {
    ($x: expr, $y: expr, $width: ident) => {
        (($width * ($y)) + ($x))
    };
    ($x: ident, $y: expr, $width: ident) => {
        (($width * ($y)) + ($x))
    };
    ($x: expr, $y: ident, $width: ident) => {
        (($width * ($y)) + ($x))
    };
    ($x: expr, $y: ident, $width: ident) => {
        (($width * ($y)) + ($x))
    };
    ($x: ident, $y: ident, $width: ident) => {
        (($width * ($y)) + ($x))
    };
    ($x: literal, $y: literal, $width: literal) => {
        (($width * ($y)) + ($x))
    };
}

macro_rules! get_pixel {
    ($arr: ident, $x: expr, $y: expr, $width: ident) => {
        $arr[get_2d_index!($x, $y, $width)]
    };
    ($arr: ident, $x: ident, $y: ident, $width: ident) => {
        $arr[get_2d_index!($x, $y, $width)]
    };
    ($arr: ident, $x: ident, $y: expr, $width: ident) => {
        $arr[get_2d_index!($x, $y, $width)]
    };
    ($arr: ident, $x: ident, $y: ident, $width: ident) => {
        $arr[get_2d_index!($x, $y, $width)]
    };
}

macro_rules! set_pixel {
    ($arr: ident, $x: expr, $y: expr, $width: ident, $val: ident) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: ident, $y: ident, $width: ident, $val: ident) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: expr, $y: ident, $width: ident, $val: ident) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: ident, $y: expr, $width: ident, $val: ident) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: expr, $y: expr, $width: ident, $val: expr) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: ident, $y: ident, $width: ident, $val: expr) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: expr, $y: ident, $width: ident, $val: expr) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
    ($arr: ident, $x: ident, $y: expr, $width: ident, $val: expr) => {{
        $arr[get_2d_index!($x, $y, $width)] = $val;
    }};
}

macro_rules! lerp {
    ($s: expr, $e: expr, $t: expr) => {
        ($s + ($e - $s) * $t)
    };
}

macro_rules! blerp {
    ($c00: expr, $c10: expr, $c01: expr, $c11: expr, $tx: ident, $ty: ident) => {
        threshold!((lerp!(lerp!($c00, $c10, $tx), lerp!($c01, $c11, $tx), $ty)).round() as u8)
    };
}

macro_rules! float_proportion {
    ($i: ident, $old_propo: ident, $new_propo: ident) => {
        ($i as f64) * (($old_propo as f64) - 1.0) / $new_propo as f64
    };
}

macro_rules! get_pixel_range {
    ($img: ident, $x1: ident, $x2: ident, $y1: ident, $y2: ident, $width: ident) => {{
        ($x1..$x2)
            .collect::<Vec<usize>>()
            .iter()
            .cloned()
            .map(|i| {
                ($y1..$y2)
                    .collect::<Vec<usize>>()
                    .iter()
                    .cloned()
                    .map(|j| $img[get_2d_index!(j, i, $width)])
                    .collect::<Vec<u8>>()
            })
            .flatten()
            .collect()
    }};
    ($img: ident, $outerrange: expr, $innerrange: expr, $width: ident) => {{
        (0..$outerrange)
            .collect::<Vec<usize>>()
            .iter()
            .cloned()
            .map(|i| {
                (0..$innerrange)
                    .collect::<Vec<usize>>()
                    .iter()
                    .cloned()
                    .map(|j| $img[get_2d_index!(j, i, $width)])
                    .collect::<Vec<u8>>()
            })
            .flatten()
            .collect()
    }};
}

macro_rules! line_to {
    ($img: ident, $fromx: ident, $fromy: ident, $tox: ident, $toy: ident, $width: ident, $color: ident, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
    ($img: ident, $fromx: ident, $fromy: ident, $tox: ident, $toy: ident, $width: ident, $color: literal, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
    ($img: ident, $fromx: expr, $fromy: expr, $tox: expr, $toy: expr, $width: ident, $color: ident, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
    ($img: ident, $fromx: ident, $fromy: ident, $tox: expr, $toy: expr, $width: ident, $color: literal, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
    ($img: ident, $fromx: expr, $fromy: expr, $tox: ident, $toy: ident, $width: ident, $color: ident, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
    ($img: ident, $fromx: literal, $fromy: ident, $tox: literal, $toy: expr, $width: ident, $color: ident, $eq: literal, $step: literal) => {{
        let start = get_2d_index!($fromx, $fromy, $width);
        let end = get_2d_index!($tox, $toy, $width);
        (start..(end - $eq))
            .step_by($step * $width + ($step ^ 1))
            .into_iter()
            .for_each(|i| $img[i] = $color);
    }};
}

macro_rules! chunk_to {
    ($img: ident, $fromx: ident, $fromy: ident, $tox: ident, $toy: ident, $width: ident, $eq: literal, $step: literal) => {};
}

fn read_image(path: &str) -> (Vec<u8>, (usize, usize)) {
    let data = err_out_res!(File::open(path), "Error opening image file");
    let decoder = png::Decoder::new(data);
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()];

    let size = reader.info().size();
    (bytes.to_vec(), (size.0 as usize, size.1 as usize))
}

fn write_image(path: &str, width: usize, height: usize, data: &[u8]) {
    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}

fn rgb_buffer_to_grayscale(buffer: &Vec<u8>) -> Vec<u8> {
    (0..=buffer.len() - 1)
        .step_by(3)
        .into_iter()
        .map(|i| to_luma!(buffer[i], buffer[i + 1], buffer[i + 2]))
        .collect::<Vec<u8>>()
}

fn rgba_buffer_to_grayscale(buffer: &Vec<u8>) -> Vec<u8> {
    (0..=buffer.len() - 1)
        .step_by(4)
        .into_iter()
        .map(|i| to_luma!(buffer[i], buffer[i + 1], buffer[i + 2]))
        .collect::<Vec<u8>>()
}

fn binarize_grayscale_buffer(buffer: &Vec<u8>) -> Vec<u8> {
    buffer.into_iter().map(|p| threshold!(*p)).collect()
}

fn convolution(
    img: &Vec<Pixel>,
    kernel: &[f64],
    nx: isize,
    ny: isize,
    kn: isize,
    normalize: bool,
) -> Vec<Pixel> {
    err_out_if_false!(
        kernel.len() as isize == kn * kn,
        "Length of kernel must be {}",
        kn * kn
    );
    err_out_if_false!(kn % 2 == 1, "KN % 2 must equal 1");
    err_out_if_false!(
        nx > kn && ny > kn,
        "NX and NY must be larger than KN which is {}",
        kn
    );
    let imglen_isize = img.len() as isize;
    err_out_if_false!(
        nx * ny == imglen_isize,
        "NX({}) * NY({}) must equal image buffer size({})",
        nx,
        ny,
        imglen_isize
    );

    let mut output = vec![0 as Pixel; img.len()];

    let khalf = kn / 2;

    let mut pmin = f64::MAX;
    let mut pmax = -f64::MAX;

    if normalize {
        for m in khalf..(nx - khalf) {
            for n in khalf..(ny - khalf) {
                let mut pixel = 0.0;
                let mut c = 0usize;
                for j in -khalf..=khalf {
                    for i in -khalf..=khalf {
                        pixel += (img[((n - j) * nx + m - i) as usize]) as f64 * kernel[c];
                        c += 1;
                    }
                }
                if pixel < pmin {
                    pmin = pixel;
                } else if pixel > pmax {
                    pmax = pixel;
                }
            }
        }
    }

    for m in khalf..(nx - khalf) {
        for n in khalf..(ny - khalf) {
            let mut pixel = 0.0;
            let mut c = 0usize;
            for j in -khalf..=khalf {
                for i in -khalf..=khalf {
                    pixel += (img[((n - j) * nx + m - i) as usize]) as f64 * kernel[c];
                    c += 1;
                }
            }
            if normalize {
                pixel = MAX_BRIGHTNESS_F64 * (pixel - pmin) / (pmax - pmin);
            }
            output[(n * nx + m) as usize] = pixel as Pixel;
        }
    }

    output
}

fn gaussian_filter(
    img: &Vec<Pixel>,
    nx: isize,
    ny: isize,
    sigma: f64,
    normalization: bool,
) -> Vec<Pixel> {
    let n = 2 * ((2.0f64 * sigma) as isize) + 3;
    let mean = ((n / 2) as f64).floor();
    let mut kernel = vec![0f64; (n * n) as usize];

    let mut c = 0usize;
    for i in 0..n {
        for j in 0..n {
            let (ifloat, jfloat) = (i as f64, j as f64);
            let (powifloat, powjfloat) = (
                ((ifloat - mean) / sigma).powf(2.0),
                ((jfloat - mean) / sigma).powf(2.0),
            );
            let exp = (-0.5 * (powifloat + powjfloat)).exp();
            kernel[c] = exp / (2.0 * PI * sigma * sigma);
            c += 1;
        }
    }

    convolution(img, &kernel, nx, ny, n, normalization)
}

fn canny_edge_detector(
    img: &Vec<u8>,
    nx: usize,
    ny: usize,
    tmin: i16,
    tmax: i16,
    sigma: f64,
    normalization: bool,
) -> Vec<u8> {
    let img_pixels = img.into_iter().map(|p| *p as Pixel).collect::<Vec<Pixel>>();
    let mut output = gaussian_filter(&img_pixels, nx as isize, ny as isize, sigma, normalization);

    let convolved_gx = convolution(
        &output,
        &KERNEL_GX,
        nx as isize,
        ny as isize,
        CONVOLUTE_GN,
        false,
    );
    let convolved_gy = convolution(
        &output,
        &KERNEL_GY,
        nx as isize,
        ny as isize,
        CONVOLUTE_GN,
        false,
    );

    let mut g = vec![0 as Pixel; img.len()];
    for i in 1..(nx - 2) {
        for j in 1..(ny - 2) {
            let c = i + nx * j;
            g[c] = (convolved_gx[c] as f64).hypot(convolved_gy[c] as f64) as Pixel;
        }
    }

    let mut nms = vec![0 as Pixel; img.len()];
    for i in 1..=(nx - 2) {
        for j in 1..=(ny - 2) {
            let c = i + nx * j;
            let nn = c - nx;
            let ss = c + nx;
            let ww = c + 1;
            let ee = c - 1;
            let nw = nn + 1;
            let ne = nn - 1;
            let sw = ss + 1;
            let se = ss - 1;

            let aux = (convolved_gy[c] as f64).atan2(convolved_gx[c] as f64) + PI;
            let dir = aux % PI / PI * 8.0;

            let cond_a =
                (dir <= NMS_ARCTANS[0] || dir > NMS_ARCTANS[3]) && g[c] > g[ee] && g[c] > g[ww];
            let cond_b =
                (dir > NMS_ARCTANS[0] || dir <= NMS_ARCTANS[1]) && g[c] > g[nw] && g[c] > g[se];
            let cond_c =
                (dir > NMS_ARCTANS[1] || dir <= NMS_ARCTANS[2]) && g[c] > g[nn] && g[c] > g[ss];
            let cond_d =
                (dir > NMS_ARCTANS[2] || dir <= NMS_ARCTANS[3]) && g[c] > g[ne] && g[c] > g[sw];
            if cond_a || cond_b || cond_c || cond_d {
                nms[c] = g[c]
            } else {
                nms[c] = 0;
            }
        }
    }

    let mut edges = vec![0i32; img.len() / 2];
    output = vec![0 as Pixel; output.len()];
    let mut c = 0;
    for _ in 1..(ny - 1) {
        for _ in 1..(nx - 1) {
            c += 1;
            if nms[c] >= tmax && output[c] == 0 {
                output[c] = MAX_BRIGHTNESS_I16;
                let mut nedges = 1;
                edges[0] = c as i32;

                while nedges > 0 {
                    nedges -= 1;
                    let t = edges[nedges] as usize;
                    let neighbors: &[usize] = &[
                        t - nx,
                        t + nx,
                        t + 1,
                        t - 1,
                        t - nx + 1,
                        t - nx - 1,
                        t + nx + 1,
                        t + nx - 1,
                    ];

                    for n in neighbors {
                        if nms[*n] >= tmin && output[*n] == 0 {
                            output[*n] = MAX_BRIGHTNESS_I16;
                            edges[nedges] = *n as i32;
                            nedges += 1;
                        }
                    }
                }
            }
        }
    }

    output.into_iter().map(|p| p as u8).collect()
}

fn scale_image(
    img: &Vec<u8>,
    width: usize,
    height: usize,
    scale: f64,
) -> (Vec<u8>, (usize, usize)) {
    let (new_width, new_height) = (
        (width as f64 * scale).round() as usize,
        (height as f64 * scale).round() as usize,
    );
    let mut scaled = vec![0u8; new_width * new_height];

    for x in 0..new_width {
        for y in 0..new_height {
            let (gx, gy) = (
                float_proportion!(x, width, new_width),
                float_proportion!(y, height, new_height),
            );
            let (gxu, gyu) = (gx as usize, gy as usize);
            let (gxf, gyf) = (gx - (gxu as f64), gy - (gyu as f64));
            set_pixel! {
                scaled,
                x,
                y,
                new_width,
                blerp!{
                    get_pixel!(img, gxu, gyu, width) as f64,
                    get_pixel!(img, gxu + 1, gyu, width) as f64,
                    get_pixel!(img, gxu, gyu + 1, width) as f64,
                    get_pixel!(img, gxu + 1, gyu + 1, width) as f64,
                    gxf,
                    gyf
                }
            };
        }
    }

    (scaled, (new_width, new_height))
}

fn crop_image(
    img: &Vec<u8>,
    width: usize,
    x1y1: (usize, usize),
    x2y2: (usize, usize),
) -> (Vec<u8>, (usize, usize)) {
    let ((x1, y1), (x2, y2)) = (x1y1, x2y2);
    err_out_if_false!(
        x1 < x2,
        "Cannot crop backwards. X1({}) must be smaller than X2({})",
        x1,
        x2
    );

    let (new_width, new_hight) = (x2 - x1, y2 - y1);
    let cropped = get_pixel_range! {
        img,
        x1, x2,
        y1, y2,
        width
    };

    return (cropped, (new_width, new_hight));
}

fn draw_grid(
    img: &Vec<u8>,
    width: usize,
    height: usize,
    grid_size: (usize, usize),
    color: u8,
) -> Vec<u8> {
    let mut gridded = img.clone();
    for i in (grid_size.0..height).step_by(grid_size.0) {
        line_to!(gridded, 0, i, width, i, width, color, 0, 0);
    }

    for i in (grid_size.0..width).step_by(grid_size.1) {
        line_to!(gridded, i, 0, i, height, width, color, 0, 1);
    }

    gridded
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let imagepath = &args[1];
    //let thresh = args[2].parse::<u8>().expect("Error parsing threshold");
    //let nx = args[3].parse::<usize>().expect("Error parsing threshold");
    //let ny = args[4].parse::<usize>().expect("Error parsing threshold");
    //let tmin = args[5].parse::<i16>().expect("Error parsing threshold");
    //let tmax = args[6].parse::<i16>().expect("Error parsing threshold");
    //let sigma = args[6].parse::<f64>().expect("Error parsing threshold");

    // let savepath  = &args[2];

    /*
        let (image, (width, height)) = read_image(imagepath);
        let grayscaled = rgba_buffer_to_grayscale(&image);
        let binrarized = binarize_grayscale_buffer(&grayscaled);
        let (nx, ny, tmin, tmax, sigma) = (width as usize, height as usize, 45, 50, 1.0);
        let canny_edges = canny_edge_detector(&binrarized, nx, ny, tmin, tmax, sigma, true);
        let (scaled, (nw, nh)) = scale_image(&canny_edges, width, height, 1.9);
        write_image("resized.png", nw, nh, &scaled);
        let (cropped, (nww, nhh)) = crop_image(&scaled, nh, (200, 300), (500, 600));
        write_image("cropped.png", nww, nhh, &cropped);
        let gridded = draw_square_grid(&binrarized, width, height, 40, HALF_BRIGHTNESS);
        write_image("gridded.png", width, height, &gridded);
        println!("{} {} {}", nw, nh, binrarized.len());
    */
    let width = 14;
    let height = 21;
    let white = vec![255u8; width * height];
    let gridded = draw_grid(&white, width, height, (7, 7), 127);
    write_image("gridded.png", width, height, &gridded);
}
