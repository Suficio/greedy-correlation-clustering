use ndarray::{Array2, ArrayView4};
use rand::{seq::index::sample, thread_rng};

mod boem_cc;
mod greedy_cc;

use boem_cc::cc_boem;
use greedy_cc::cc_vote_logarithmic;

pub fn boem_cc_iterations(
    clusters: &mut Array2<usize>,
    cluster_indices: &mut Vec<Vec<(usize, usize)>>,
    correlations: ArrayView4<f64>,
) -> f64 {
    cc_boem(clusters, cluster_indices, correlations);
    cluster_total_distance(clusters, correlations)
}

pub fn greedy_cc_iterations(
    clusters: &mut Array2<usize>,
    cluster_indices: &mut Vec<Vec<(usize, usize)>>,
    correlations: ArrayView4<f64>,
    iterations: usize,
    no_clusters: usize,
) -> (f64, f64) {
    let size = clusters.shape();
    let indices = indices(size[0], size[1]);

    // Initial run
    cc_vote_logarithmic(
        clusters,
        cluster_indices,
        correlations,
        index_permutations(indices.len()).map(|a| indices[a]),
        no_clusters,
    );

    let mut clusters_star = clusters.clone();
    let mut clusters_star_cost = cluster_total_distance(clusters, correlations);
    let mut cluster_star_indices = cluster_indices.clone();
    let mut cluster_cost_average = clusters_star_cost;

    // Clusters doesnt need to be reset as it overwrites all data
    cluster_indices.clear();

    for i in 1..iterations {
        println!(
            "Clustering...{:.2}%",
            (i as f32) / (iterations as f32) * 100.
        );

        cc_vote_logarithmic(
            clusters,
            cluster_indices,
            correlations,
            index_permutations(indices.len()).map(|a| indices[a]),
            no_clusters,
        );
        let cost = cluster_total_distance(clusters, correlations);

        cluster_cost_average = ((i as f64) * cluster_cost_average + cost) / ((i as f64) + 1.);

        // Minimize cluster configuration
        if clusters_star_cost > cost {
            clusters_star = clusters.clone();
            clusters_star_cost = cost;
            cluster_star_indices = cluster_indices.clone();
        }

        cluster_indices.clear();
    }

    println!(
        "Found best cluster configuration, {:.4}% better than average",
        100. * (cluster_cost_average - clusters_star_cost) / cluster_cost_average
    );
    *clusters = clusters_star;
    *cluster_indices = cluster_star_indices;

    (clusters_star_cost, cluster_cost_average)
}

// Cluster utility functions

pub fn cluster_total_distance(clusters: &Array2<usize>, correlations: ArrayView4<f64>) -> f64 {
    let mut sum = 0.;

    for (i, cluster_i) in clusters.indexed_iter() {
        for (j, cluster_j) in clusters.indexed_iter() {
            if i == j {
                continue;
            }
            // Calculate objective value using additive weights
            // x_ij*w_ij^- + (1 - x_ij)*w_ij^+
            let p_ij = correlations[(i.0, i.1, j.0, j.1)];
            if cluster_i == cluster_j {
                sum += 1. - p_ij;
            } else {
                sum += p_ij;
            }
        }
    }

    sum
}

// Utility functions

pub fn indices(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut indices = Vec::<(usize, usize)>::with_capacity(width * height);
    for x in 0..width {
        for y in 0..height {
            indices.push((x, y));
        }
    }

    indices
}

pub fn index_permutations(length: usize) -> impl Iterator<Item = usize> {
    let mut rng = thread_rng();
    sample(&mut rng, length, length).into_iter()
}
