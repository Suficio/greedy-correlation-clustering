use image;
use ndarray::{Array2, Array4, ArrayView4};
use ndarray_image::open_gray_image;
use ndarray_npy::ReadNpyExt;

use std::fs;
use std::path::Path;

use correlation_clustering::{boem_cc_iterations, cluster_total_distance, greedy_cc_iterations};

fn main() {
    let size = (74, 91);
    let bounds = (0, 720);

    let mut correlations = correlate(bounds);
    normalize_correlations(&mut correlations, 0.9); // Sets at which point a value is considered simillar

    let view = correlations.view();
    for _ in 0..1 {
        let mut clusters = Array2::<usize>::zeros(size);
        let mut cluster_indices = Vec::<Vec<(usize, usize)>>::new(); // Maintains a list of which indices are in which cluster

        // let cloned_correlations = shared_correlations.into_owned();
        greedy_cc_iterations(&mut clusters, &mut cluster_indices, view, 20, 256);
        boem_cc_iterations(&mut clusters, &mut cluster_indices, view);

        update_file(view, &clusters, bounds);
    }
}

fn correlate(bounds: (usize, usize)) -> Array4<f64> {
    let (lower_bound, higher_bound) = bounds;

    // Will panic if it exists
    match fs::create_dir_all("examples/correlations") {
        _ => {}
    }

    let file_name = format!(
        "examples/correlations/s_correlations_{}_{}.npy",
        lower_bound, higher_bound
    );
    Array4::<u8>::read_npy(fs::File::open(&file_name).unwrap())
        .unwrap()
        .mapv(|a| (a as f64) / 255.)
}

// Utility Functions

fn normalize_correlations(correlations: &mut Array4<f64>, cutoff: f64) {
    // Normalizes such that correlation at cutoff corresponds to 0
    // and that 0 < p < 1
    correlations.mapv_inplace(|a| {
        if a >= cutoff {
            (1. + (a - cutoff) / (1. - cutoff)) / 2.
        } else {
            (1. + (a - cutoff) / (1. + cutoff)) / 2.
        }
    })
}

pub fn update_file(
    correlations: ArrayView4<f64>,
    clusters: &Array2<usize>,
    bounds: (usize, usize),
) {
    let (lower_bound, higher_bound) = bounds;
    match open_gray_image(&Path::new(&format!(
        "examples/cluster_data/{}_{}.tiff",
        lower_bound, higher_bound
    ))) {
        Ok(data) => {
            // Correlations the same for any given bounds
            let last = cluster_total_distance(&data.map(|x| *x as usize), correlations);
            let current = cluster_total_distance(clusters, correlations);

            if current < last {
                println!(
                    "File for bounds {:?} to {:?} replaced with better solution",
                    lower_bound, higher_bound
                );
                return write_file(clusters, bounds);
            }
        }
        Err(_) => {
            // File prob not found so overwrite
            println!(
                "File for bounds {:?} to {:?} written",
                lower_bound, higher_bound
            );
            return write_file(clusters, bounds);
        }
    }

    println!(
        "File for bounds {:?} to {:?} already has better solution",
        lower_bound, higher_bound
    );
}

pub fn write_file(clusters: &Array2<usize>, bounds: (usize, usize)) {
    let size = clusters.shape();
    let kelly_colors_hex = [
        [0xFF, 0xB3, 0x00], // Vivid Yellow
        [0x80, 0x3E, 0x75], // Strong Purple
        [0xFF, 0x68, 0x00], // Vivid Orange
        [0xA6, 0xBD, 0xD7], // Very Light Blue
        [0xC1, 0x00, 0x20], // Vivid Red
        [0xCE, 0xA2, 0x62], // Grayish Yellow
        [0x81, 0x70, 0x66], // Medium Gray
        // The following don't work well for people with defective color vision
        [0x00, 0x7D, 0x34], // Vivid Green
        [0xF6, 0x76, 0x8E], // Strong Purplish Pink
        [0x00, 0x53, 0x8A], // Strong Blue
        [0xFF, 0x7A, 0x5C], // Strong Yellowish Pink
        [0x53, 0x37, 0x7A], // Strong Violet
        [0xFF, 0x8E, 0x00], // Vivid Orange Yellow
        [0xB3, 0x28, 0x51], // Strong Purplish Red
        [0xF4, 0xC8, 0x00], // Vivid Greenish Yellow
        [0x7F, 0x18, 0x0D], // Strong Reddish Brown
        [0x93, 0xAA, 0x00], // Vivid Yellowish Green
        [0x59, 0x33, 0x15], // Deep Yellowish Brown
        [0xF1, 0x3A, 0x13], // Vivid Reddish Orange
        [0x23, 0x2C, 0x16], // Dark Olive Green
    ];

    let mut buffer: Vec<u8> = vec![0; size[0] * size[1]];
    let mut visual_buffer: Vec<u8> = vec![0; 3 * size[0] * size[1]];

    for (index, cluster) in clusters.indexed_iter() {
        // Buffer uses different row order
        let m = index.0 * size[1] + index.1;
        buffer[m] = *cluster as u8;

        let rgb = kelly_colors_hex[cluster % kelly_colors_hex.len()];
        visual_buffer[3 * m] = rgb[0];
        visual_buffer[3 * m + 1] = rgb[1];
        visual_buffer[3 * m + 2] = rgb[2];
    }

    match fs::create_dir_all("examples/cluster_data") {
        _ => {}
    }

    match fs::create_dir_all("examples/visual_cluster_data") {
        _ => {}
    }

    let (lower_bound, higher_bound) = bounds;
    image::save_buffer(
        &Path::new(&format!(
            "examples/cluster_data/{}_{}.tiff",
            lower_bound, higher_bound
        )),
        &buffer,
        size[1] as u32,
        size[0] as u32,
        image::ColorType::L8,
    )
    .unwrap();

    image::save_buffer(
        &Path::new(&format!(
            "examples/visual_cluster_data/{}_{}.tiff",
            lower_bound, higher_bound
        )),
        &visual_buffer,
        size[1] as u32,
        size[0] as u32,
        image::ColorType::Rgb8,
    )
    .unwrap();
}
