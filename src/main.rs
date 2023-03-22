use std::{f64::consts::PI, fs::File, io::BufWriter, path::Path};

type Pixel = i16;

const LUM_FACTOR_RED: f64 = 0.2126;
const LUM_FACTOR_GREEN: f64 = 0.7152;
const LUM_FACTOR_BLUE: f64 = 0.0722;
const CONVOLUTE_GN: isize = 3;
const MAX_BRIGHTNESS_F64: f64 = 255.00;
const MAX_BRIGHTNESS_I16: Pixel = 255;
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

macro_rules! threshold {
    ($pixel: expr, $thresh: ident, $gte: literal, $lt: literal) => {
        if $pixel >= $thresh {
            $gte
        } else {
            $lt
        }
    };
}

fn read_image(path: &str) -> (Vec<u8>, (u32, u32)) {
    let data = err_out_res!(File::open(path), "Error opening image file");
    let decoder = png::Decoder::new(data);
    let mut reader = decoder.read_info().unwrap();
    let size = reader.info().size();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let bytes = &buf[..info.buffer_size()];
    (bytes.to_vec(), size)
}

fn write_image(path: &str, width: u32, height: u32, data: &[u8]) {
    let path = Path::new(path);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);

    let mut encoder = png::Encoder::new(w, width, height);
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

fn binarize_grayscale_buffer(buffer: &Vec<u8>, thresh: u8) -> Vec<u8> {
    buffer
        .into_iter()
        .map(|p| threshold!(*p, thresh, 255, 0))
        .collect()
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let imagepath = &args[1];
    //let thresh = args[2].parse::<u8>().expect("Error parsing threshold");
    //let nx = args[3].parse::<usize>().expect("Error parsing threshold");
    //let ny = args[4].parse::<usize>().expect("Error parsing threshold");
    //let tmin = args[5].parse::<i16>().expect("Error parsing threshold");
    //let tmax = args[6].parse::<i16>().expect("Error parsing threshold");
    //let sigma = args[6].parse::<f64>().expect("Error parsing threshold");

    let thresh = 125;
    // let savepath  = &args[2];

    let (image, (width, height)) = read_image(imagepath);

    let grayscaled = rgba_buffer_to_grayscale(&image);
    write_image("gs.png", width, height, &grayscaled);
    let binrarized = binarize_grayscale_buffer(&grayscaled, thresh);
    write_image("bin.png", width, height, &binrarized);
    let (nx, ny, tmin, tmax, sigma) = (width as usize, height as usize, 45, 50, 1.0);
    let canny_edges = canny_edge_detector(&binrarized, nx, ny, tmin, tmax, sigma, true);
    write_image("fin.png", width, height, &canny_edges);
}
