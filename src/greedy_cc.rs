use ndarray::{Array2, ArrayView4};

pub fn cc_vote_logarithmic(
    clusters: &mut Array2<usize>,
    cluster_indices: &mut Vec<Vec<(usize, usize)>>,
    correlations: ArrayView4<f64>,
    pi: impl Iterator<Item = (usize, usize)>,
    no_clusters: usize,
) {
    // https://pdfs.semanticscholar.org/1c37/b0b78434f415ced54c87615cadacd955430a.pdf
    for i in pi {
        let mut c_star_index = 0;
        let mut c_star_quality = std::f64::NEG_INFINITY;

        for (c_index, c) in cluster_indices.iter().enumerate() {
            let mut c_quality = 0.;
            for j in c.iter() {
                // Calculate net weight according to logarithmic weights:
                // w_ij^+ = ln(p_ij), w_ij^- = ln(1 - p_ij)
                let p_ij = correlations[(i.0, i.1, j.0, j.1)];
                c_quality += p_ij.ln() - (1. - p_ij).ln();
            }

            // Maximizes c_star_quality
            if c_star_quality < c_quality {
                c_star_index = c_index;
                c_star_quality = c_quality;
            }
        }

        if c_star_quality <= 0. && cluster_indices.len() < no_clusters {
            // Create cluster in case a suitable one is not found
            // and no_clusters has not been met
            let c_singleton = Vec::<(usize, usize)>::with_capacity(1);

            c_star_index = cluster_indices.len();
            cluster_indices.push(c_singleton);
        }

        cluster_indices[c_star_index].push(i);
        clusters[i] = c_star_index;
    }
}

#[allow(dead_code)]
pub fn cc_vote_additive(
    clusters: &mut Array2<usize>,
    cluster_indices: &mut Vec<Vec<(usize, usize)>>,
    correlations: ArrayView4<f64>,
    pi: impl Iterator<Item = (usize, usize)>,
    no_clusters: usize,
) {
    // https://pdfs.semanticscholar.org/1c37/b0b78434f415ced54c87615cadacd955430a.pdf
    for i in pi {
        let mut c_star_index = cluster_indices.len();
        let mut c_star_quality = std::f64::NEG_INFINITY;

        for (c_index, c) in cluster_indices.iter().enumerate() {
            let mut c_quality = 0.;
            for j in c.iter() {
                // Calculate net weight according to additive weights:
                // w_ij^+_ = w_ij^+ - w_ij^-, w_ij^+ = p_ij, w_ij^- = 1 - p_ij
                c_quality += 2. * correlations[(i.0, i.1, j.0, j.1)] - 1.
            }

            // Maximizes c_star_quality
            if c_star_quality < c_quality {
                c_star_index = c_index;
                c_star_quality = c_quality;
            }
        }

        if c_star_quality <= 0. && cluster_indices.len() < no_clusters {
            // Create cluster in case a suitable one is not found
            // and no_clusters has not been met
            let c_singleton = Vec::<(usize, usize)>::with_capacity(1);

            c_star_index = cluster_indices.len();
            cluster_indices.push(c_singleton);
        }

        cluster_indices[c_star_index].push(i);
        clusters[i] = c_star_index;
    }
}
