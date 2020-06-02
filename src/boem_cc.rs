use ndarray::{Array2, ArrayView4};

use crate::indices;

pub fn cc_boem(
    clusters: &mut Array2::<usize>,
    cluster_indices: &mut Vec::<Vec::<(usize, usize)>>,
    correlations: ArrayView4::<f64>
) {
    let size = clusters.shape();
    let mut potential_clusters = Array2::<usize>::zeros((size[0], size[1]));
    let mut potential_net_distance = Array2::<f64>::zeros((size[0], size[1]));

    let pi = indices(size[0], size[1]);

    // Initialize all potential moves
    for c in 0..cluster_indices.len() {
        update_cluster(clusters, cluster_indices, &mut potential_clusters, &mut potential_net_distance, c, correlations);
    }

    // Sets some limit to iterations
    for _ in 0..10000 {
        let mut i_star = (0, 0);
        let mut c_star_net_distance = 0.;
        // Determine best element to move
        for i in &pi {
            let net_distance = potential_net_distance[*i];
            if net_distance < c_star_net_distance {
                i_star = *i;
                c_star_net_distance = net_distance;
            }
        }

        // Move element
        let c_original = clusters[i_star];
        let c_star = potential_clusters[i_star];
    
        println!("Move {:?} from {:?} to {:?} with improvement {:?}", i_star, c_original, c_star, c_star_net_distance);

        if c_original == c_star {break}

        let index = cluster_indices[c_original].iter().position(|i| *i == i_star).unwrap();
        cluster_indices[c_original].remove(index);

        clusters[i_star] = c_star;
        cluster_indices[c_star].push(i_star);

        // Update the two clusters
        update_cluster(clusters, cluster_indices, &mut potential_clusters, &mut potential_net_distance, c_star, correlations);
        update_cluster(clusters, cluster_indices, &mut potential_clusters, &mut potential_net_distance, c_original, correlations);
    }
}

fn update_cluster(
    clusters: &Array2::<usize>,
    cluster_indices: &Vec::<Vec::<(usize, usize)>>,
    potential_clusters: &mut Array2::<usize>,
    potential_net_distance: &mut Array2::<f64>,
    cluster: usize,
    correlations: ArrayView4::<f64>
) {
    for i in &cluster_indices[cluster] {
        let mut c_star = 0;
        let mut c_star_net_distance = 0.;

        for c in 0..cluster_indices.len() {
            let c_net_distance = net_cluster_distance(cluster_indices, correlations, clusters[*i], c, *i);

            if c_net_distance < c_star_net_distance {
                c_star = c;
                c_star_net_distance = c_net_distance;
            }
        }

        potential_clusters[*i] = c_star;
        potential_net_distance[*i] = c_star_net_distance;
    }
}

fn net_cluster_distance(
    cluster_indices: &Vec::<Vec::<(usize, usize)>>,
    correlations: ArrayView4::<f64>,
    c0: usize,
    c1: usize,
    i: (usize, usize)
) -> f64 {
    let mut sum = 0.;

    // Calculate objective value using additive weights
    // x_ij*w_ij^- + (1 - x_ij)*w_ij^+
    for j in &cluster_indices[c0] {
        if i == *j {continue}
        let p_ij = correlations[(i.0, i.1, j.0, j.1)];
        sum += 2.*p_ij - 1.;
    }

    for j in &cluster_indices[c1] {
        if i == *j {continue}
        let p_ij = correlations[(i.0, i.1, j.0, j.1)];
        sum += 1. - 2.*p_ij;
    }

    // Sums are computed twice
    2.*sum
}
