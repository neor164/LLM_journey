//! This module will contain the Ball Tree implementation.

use num_traits::{Float, NumCast}; // Ensure this is at the top
use std::iter::Sum; // Keep this import
use rand::seq::SliceRandom; // For random sampling
use rand::thread_rng;       // For random number generator
use super::heap_utils::KBestNeighbors; // Import shared heap utilities

use super::knn::DataPoint; 

pub fn mean<F>(vector: &[F]) -> F
where
    F: Float + Sum<F>, // Float implies Copy and NumCast. Sum<F> allows summing F's.
         {

    let length = vector.len();

    if length == 0 {
        return F::nan(); // Or handle as appropriate
    }
    let sum = vector.iter().cloned().sum::<F>();
    match NumCast::from(length) {
        Some(n_as_f) => {
            if n_as_f == F::zero() { // Should be caught by n == 0 check
                F::nan()
            } else {
                sum / n_as_f // This is where your division happens
            }
        }
        None => {
            // Failed to convert length to F (highly unlikely for standard floats)
            F::nan()
        }
    }
}

pub fn variance<F> (vector: &[F]) -> F
where
    F: Float + Sum<F>, 
{

    let length = vector.len();

    // Sample variance is undefined for n < 2 because of division by (n-1)
    if length < 2 {
        return F::nan();
    }
    let mean_val = mean(vector);
    let sum_of_squared_differences = vector.iter().map(|f| { // f_ref is &F
            let f_val = *f; // Dereference to get F (since F: Float => F: Copy)
            let difference = f_val - mean_val;
            difference.powi(2) // Square the difference
        }).sum::<F>();

    match NumCast::from(length - 1) {
        Some(n_minus_1_as_f) => {
            // The n < 2 check should prevent n_minus_1_as_f from being zero.
            // But as a safeguard if F has unusual zero properties:
            if n_minus_1_as_f == F::zero() {
                F::nan()
            } else {
                sum_of_squared_differences / n_minus_1_as_f
            }
        }
        None => {
            // This means n-1 couldn't be cast to F, highly unlikely for standard floats.
            F::nan()
        }
        
    }

}
#[derive(Debug, Clone)]
pub enum BallNodeContents<F, L> {
    Internal {
        // The left child ball/node.
        left_child: Box<BallTreeNode<F, L>>,
        // The right child ball/node.
        right_child: Box<BallTreeNode<F, L>>,
    },
    Leaf {
        points: Vec<DataPoint<F, L>>,
    },
}

#[derive(Debug, Clone)]
pub struct BallTreeNode<F, L> {
    pub centroid: Vec<F>, 
    pub radius: F, 
    pub contents: BallNodeContents<F, L>,
}

/// Represents the Ball Tree itself.
#[derive(Debug)]
pub struct BallTree<F, L> {
    /// The root node of the Ball Tree. It's an Option because the tree might be empty.
    pub root: Option<Box<BallTreeNode<F, L>>>,
}

impl<F, L> BallTree <F,L>
where
    F: num_traits::Float + num_traits::AsPrimitive<f64> + Clone + std::iter::Sum, 
    L: Clone,
{


    const DEFAULT_LEAF_SIZE: usize = 40;
    // For new_random_variance_split and the general `new` constructor
    const DEFAULT_SAMPLE_SIZE_FOR_VARIANCE: usize = 64; // Smaller sample for variance calc
    const DEFAULT_FALLBACK_THRESHOLD_FOR_RANDOM_VARIANCE: usize = 128; // If fewer points, use exact variance
    const DEFAULT_DIMENSION_SAMPLING_THRESHOLD: usize = 50; // If k_dims > this, sample dimensions
    const DEFAULT_SAMPLE_SIZE_FOR_CENTROID: usize = 64; // Sample size for centroid calculation
    const DEFAULT_SAMPLE_SIZE_FOR_DIMENSIONS_FACTOR: f64 = 0.5; // e.g., sample sqrt(k) or 0.5*k dimensions


    fn calculate_centroid(points_slice: &[DataPoint<F, L>], k_dimensions: usize) -> Vec<F> {
        if points_slice.is_empty() {
            return vec![F::zero(); k_dimensions]; 
        }
        let num_points = F::from(points_slice.len()).unwrap_or_else(F::one); // Convert usize to F
        let mut centroid_coords = vec![F::zero(); k_dimensions];

        for point in points_slice {
            for dim in 0..k_dimensions {
                centroid_coords[dim] = centroid_coords[dim] + point.features[dim];
            }
        }

        for dim in 0..k_dimensions {
            centroid_coords[dim] = centroid_coords[dim] / num_points;
        }
        centroid_coords
    }


    fn distance_squared(p1_features: &[F], p2_features: &[F]) -> F {
        p1_features.iter().zip(p2_features.iter()).map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        }).sum() // F needs std::iter::Sum
    }

    /// Calculates the Euclidean distance between two feature vectors.
    fn euclidean_distance(p1_features: &[F], p2_features: &[F]) -> f64 {
        let sum_sq_diff: F = p1_features.iter().zip(p2_features.iter()).map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        }).sum();
        sum_sq_diff.as_().sqrt() // Convert F to f64, then sqrt
    }

    fn calculate_bounding_ball(
        points_slice: &[DataPoint<F, L>],
        k_dimensions: usize,
        sample_size_for_centroid_opt: Option<usize>,
    ) -> (Vec<F>, F) {
        if points_slice.is_empty() {
            return (vec![F::zero(); k_dimensions], F::nan()); // Return NaN for radius if empty
        }

        let centroid = if let Some(sample_size) = sample_size_for_centroid_opt {
            if points_slice.len() > sample_size && sample_size > 0 {
                let mut rng: rand::prelude::ThreadRng = thread_rng();
                // Ensure sample_size doesn't exceed points_slice.len()
                let actual_sample_size = sample_size.min(points_slice.len());
                let centroid_sample: Vec<DataPoint<F,L>> = points_slice
                    .choose_multiple(&mut rng, actual_sample_size)
                    .cloned()
                    .collect();
                if centroid_sample.is_empty() { // Should not happen if actual_sample_size > 0
                    Self::calculate_centroid(points_slice, k_dimensions) // Fallback
                } else {
                    Self::calculate_centroid(&centroid_sample, k_dimensions)
                }
            } else { // Not enough points to sample or sample_size is 0
                Self::calculate_centroid(points_slice, k_dimensions)
            }
        } else { // No sample size provided, use exact
            Self::calculate_centroid(points_slice, k_dimensions)
        };

        let mut max_dist_sq = F::zero();

        for point in points_slice {
            let dist_sq = Self::distance_squared(&centroid, &point.features);
            if dist_sq > max_dist_sq {
                max_dist_sq = dist_sq;
            }
        }
        let radius = max_dist_sq.sqrt();
        if radius.is_nan() && !points_slice.is_empty() { // e.g. if centroid was NaN from empty sample
            return (Self::calculate_centroid(points_slice, k_dimensions), F::zero()); // Fallback for radius
        }
        (centroid, radius)
    }

    pub fn new(points: Vec<DataPoint<F, L>>, leaf_size_opt: Option<usize>) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let k = match points.first() {
            Some(p) => p.features.len(),
            None => return None, 
        };

        if k == 0 {
            return None;
        }

        for p in points.iter() {
            if p.features.len() != k {
                panic!("All data points must have the same number of features (dimensionality). Found point with {} features, expected {}.", p.features.len(), k);
            }
        }

        let leaf_size = leaf_size_opt.unwrap_or(Self::DEFAULT_LEAF_SIZE);
        if leaf_size == 0 {
            panic!("leaf_size must be greater than 0.");
        }

        let root_node = Self::build_recursive(points, leaf_size, k, None); // Pass None for centroid sampling
        Some(BallTree { root: root_node })
    }

    pub fn new_random_variance_split(
        points: Vec<DataPoint<F, L>>,
        leaf_size_opt: Option<usize>,
        sample_size_for_variance_opt: Option<usize>,
        fallback_threshold_opt: Option<usize>,
        dimension_sampling_threshold_opt: Option<usize>, // New param
        sample_size_for_dimensions_factor_opt: Option<f64>, // New param
        sample_size_for_centroid_opt: Option<usize>, // New param
    ) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let k = match points.first() {
            Some(p) => p.features.len(),
            None => return None,
        };

        if k == 0 {
            return None;
        }

        for p in points.iter() {
            if p.features.len() != k {
                panic!("All data points must have the same number of features (dimensionality). Found point with {} features, expected {}.", p.features.len(), k);
            }
        }

        let leaf_size = leaf_size_opt.unwrap_or(Self::DEFAULT_LEAF_SIZE);
        let sample_size = sample_size_for_variance_opt.unwrap_or(Self::DEFAULT_SAMPLE_SIZE_FOR_VARIANCE);
        let fallback_threshold = fallback_threshold_opt.unwrap_or(Self::DEFAULT_FALLBACK_THRESHOLD_FOR_RANDOM_VARIANCE);
        let dim_sampling_thresh = dimension_sampling_threshold_opt.unwrap_or(Self::DEFAULT_DIMENSION_SAMPLING_THRESHOLD);
        let dim_sample_factor = sample_size_for_dimensions_factor_opt.unwrap_or(Self::DEFAULT_SAMPLE_SIZE_FOR_DIMENSIONS_FACTOR);
        let centroid_sample_size = sample_size_for_centroid_opt.unwrap_or(Self::DEFAULT_SAMPLE_SIZE_FOR_CENTROID);

        if leaf_size == 0 {
            panic!("leaf_size must be greater than 0.");
        }
        if sample_size < 2 { // Need at least 2 points to calculate variance
            panic!("sample_size_for_variance must be at least 2.");
        }

        let root_node = Self::build_recursive_random_variance(
            points,
            leaf_size,
            k,
            sample_size,
            fallback_threshold,
            dim_sampling_thresh,
            dim_sample_factor,
            Some(centroid_sample_size), // Pass centroid sample size
        );
        Some(BallTree { root: root_node })
    }

    pub fn new_auto(
        points: Vec<DataPoint<F, L>>,
        leaf_size_opt: Option<usize>,
        // Threshold to switch from exact variance to random sample variance
        auto_decision_threshold_opt: Option<usize>,
        sample_size_for_variance_opt: Option<usize>,
        fallback_threshold_for_random_opt: Option<usize>,
        dimension_sampling_threshold_opt: Option<usize>,
        sample_size_for_dimensions_factor_opt: Option<f64>,
        sample_size_for_centroid_opt: Option<usize>,
    ) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        // Use a different threshold name to avoid confusion with k-d tree's auto_decision
        const BALLTREE_AUTO_VARIANCE_THRESHOLD: usize = 1000; // Example value

        let auto_thresh = auto_decision_threshold_opt.unwrap_or(BALLTREE_AUTO_VARIANCE_THRESHOLD);

        if points.len() < auto_thresh {
            Self::new(points, leaf_size_opt) 
        } else {
            Self::new_random_variance_split(
                points,
                leaf_size_opt,
                sample_size_for_variance_opt,
                fallback_threshold_for_random_opt,
                dimension_sampling_threshold_opt,
                sample_size_for_dimensions_factor_opt,
                sample_size_for_centroid_opt,
            )
        }
    }

    fn build_recursive_random_variance(
        mut points: Vec<DataPoint<F, L>>,
        leaf_size: usize,
        k_dimensions: usize,
        sample_size_for_variance: usize,
        fallback_threshold: usize,
        dimension_sampling_threshold: usize,
        dimension_sample_factor: f64,
        sample_size_for_centroid: Option<usize>, // New param
    ) -> Option<Box<BallTreeNode<F, L>>> {
        if points.is_empty() {
            return None;
        }

        // Determine if we should sample for centroid in this specific call
        let current_sample_size_for_centroid = if points.len() > fallback_threshold { sample_size_for_centroid } else { None };

        let (current_node_centroid, current_node_radius) =
            Self::calculate_bounding_ball(&points, k_dimensions, current_sample_size_for_centroid);

        if points.len() <= leaf_size || points.len() < 2 {
            return Some(Box::new(BallTreeNode {
                centroid: current_node_centroid,
                radius: current_node_radius,
                contents: BallNodeContents::Leaf { points },
            }));
        }

        let mut best_splitting_dimension = 0;
        let mut max_variance = F::neg_infinity();

        if k_dimensions == 0 { 
             return Some(Box::new(BallTreeNode {
                centroid: current_node_centroid,
                radius: current_node_radius,
                contents: BallNodeContents::Leaf { points },
            }));
        }

        let use_sampled_variance = points.len() >= fallback_threshold && points.len() >= sample_size_for_variance;

        // These vectors will hold the points for the left and right children after partitioning.
        let left_points_vec: Vec<DataPoint<F, L>>;
        let right_points_vec: Vec<DataPoint<F, L>>;

        if use_sampled_variance {
            let mut rng = thread_rng();
            // Ensure sample_size doesn't exceed points.len(), and is at least 2 for variance
            let actual_sample_size = sample_size_for_variance.min(points.len()).max(2);
            
            if points.len() < 2 { // Should be caught by leaf_size or other checks
                 return Some(Box::new(BallTreeNode { // Fallback to leaf
                    centroid: current_node_centroid,
                    radius: current_node_radius,
                    contents: BallNodeContents::Leaf { points },
                }));
            }

            let mut sample_for_variance: Vec<DataPoint<F, L>> = points // Make mutable for sorting
                .choose_multiple(&mut rng, actual_sample_size)
                .cloned()
                .collect();

            if sample_for_variance.len() < 2 { // If sampling failed to get enough points
                // Fallback to using all points for variance or make leaf if too few overall
                // This indicates an issue or very few points. For simplicity, make leaf.
                return Some(Box::new(BallTreeNode {
                    centroid: current_node_centroid,
                    radius: current_node_radius,
                    contents: BallNodeContents::Leaf { points },
                }));
            }

            // Dimension sampling logic
            if k_dimensions > dimension_sampling_threshold && k_dimensions > 1 {
                let num_dims_to_sample = ((k_dimensions as f64 * dimension_sample_factor).ceil() as usize).max(1).min(k_dimensions);
                let mut dim_indices: Vec<usize> = (0..k_dimensions).collect();
                dim_indices.shuffle(&mut rng);

                for &dim_idx in dim_indices.iter().take(num_dims_to_sample) {
                    let dim_values: Vec<F> = sample_for_variance.iter().map(|p| p.features[dim_idx]).collect();
                    let var = variance(&dim_values);
                    if !var.is_nan() && var > max_variance {
                        max_variance = var;
                        best_splitting_dimension = dim_idx;
                    }
                }
            } else { // Iterate all dimensions
                for dim_idx in 0..k_dimensions {
                    let dim_values: Vec<F> = sample_for_variance.iter().map(|p| p.features[dim_idx]).collect();
                    let var = variance(&dim_values);
                    if !var.is_nan() && var > max_variance {
                        max_variance = var;
                        best_splitting_dimension = dim_idx;
                    }
                }
            }

            // --- Partitioning based on the median of the sample ---
            // Sort the sample used for variance to find its median value for partitioning
            sample_for_variance.sort_by(|a, b| {
                a.features[best_splitting_dimension]
                    .partial_cmp(&b.features[best_splitting_dimension])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Ensure sample_for_variance is not empty before accessing (should be guaranteed by actual_sample_size.max(2))
            let pivot_value_from_sample = sample_for_variance[sample_for_variance.len() / 2].features[best_splitting_dimension];

            let mut temp_left = Vec::with_capacity(points.len() / 2);
            let mut temp_right = Vec::with_capacity(points.len() / 2);

            for point in points.into_iter() { // Consumes the original `points` Vec
                if point.features[best_splitting_dimension] < pivot_value_from_sample {
                    temp_left.push(point);
                } else {
                    // Points with feature value equal to the pivot also go to the right.
                    temp_right.push(point);
                }
            }
            left_points_vec = temp_left;
            right_points_vec = temp_right;

        } else { // Use all points for variance (like in original build_recursive) and true median for partitioning
            // Dimension sampling logic (can also be applied here if desired)
            if k_dimensions > dimension_sampling_threshold && k_dimensions > 1 {
                let num_dims_to_sample = ((k_dimensions as f64 * dimension_sample_factor).ceil() as usize).max(1).min(k_dimensions);
                let mut rng = thread_rng(); 
                let mut dim_indices: Vec<usize> = (0..k_dimensions).collect();
                dim_indices.shuffle(&mut rng);
                for &dim_idx in dim_indices.iter().take(num_dims_to_sample) {
                    let dim_values: Vec<F> = points.iter().map(|p| p.features[dim_idx]).collect();
                    let var = variance(&dim_values);
                    if !var.is_nan() && var > max_variance {
                        max_variance = var;
                        best_splitting_dimension = dim_idx;
                    }
                }
            } else { // Iterate all dimensions
                for dim_idx in 0..k_dimensions {
                    let dim_values: Vec<F> = points.iter().map(|p| p.features[dim_idx]).collect();
                    let var = variance(&dim_values);
                    if !var.is_nan() && var > max_variance {
                        max_variance = var;
                        best_splitting_dimension = dim_idx;
                    }
                }
            }

            // If max_variance is still problematic after checking all points, make it a leaf
            if max_variance.is_nan() || max_variance <= F::epsilon() {
                return Some(Box::new(BallTreeNode {
                    centroid: current_node_centroid,
                    radius: current_node_radius,
                    contents: BallNodeContents::Leaf { points }, // `points` is still the original Vec here
                }));
            }

            // Partition using true median of all points
            let median_idx = points.len() / 2;
            points.select_nth_unstable_by(median_idx, |a, b| {
                a.features[best_splitting_dimension]
                    .partial_cmp(&b.features[best_splitting_dimension])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let temp_right_points_vec = points.split_off(median_idx); 
            // points now contains the left part
            left_points_vec = points;  // Assign to the outer scoped variable
            right_points_vec = temp_right_points_vec; // Assign to the outer scoped variable
        }

        // Now left_points_vec and right_points_vec are definitely in scope.
        if left_points_vec.is_empty() || right_points_vec.is_empty() {
            let mut combined_points = left_points_vec;
            combined_points.extend(right_points_vec);
            let (leaf_centroid, leaf_radius) = Self::calculate_bounding_ball(&combined_points, k_dimensions, sample_size_for_centroid);
            return Some(Box::new(BallTreeNode {
                centroid: leaf_centroid,
                radius: leaf_radius,
                contents: BallNodeContents::Leaf { points: combined_points },
            }));
        }

        let left_child_opt = Self::build_recursive_random_variance(
            left_points_vec, leaf_size, k_dimensions, sample_size_for_variance, fallback_threshold, dimension_sampling_threshold, dimension_sample_factor, sample_size_for_centroid
        );
        let right_child_opt: Option<Box<BallTreeNode<F, L>>> = Self::build_recursive_random_variance(
            right_points_vec, leaf_size, k_dimensions, sample_size_for_variance, fallback_threshold, dimension_sampling_threshold, dimension_sample_factor, sample_size_for_centroid
        );

        match (left_child_opt, right_child_opt) {
            (Some(left_child), Some(right_child)) => {
                Some(Box::new(BallTreeNode {
                    centroid: current_node_centroid,
                    radius: current_node_radius,
                    contents: BallNodeContents::Internal {
                        left_child,
                        right_child,
                    },
                }))
            }
            _ => {
                eprintln!("Warning: BallTree random variance construction resulted in a None child. This should ideally not happen.");
                // Fallback: make current node a leaf with all points it started with.
                // This requires having the original `points` before they were split and moved.
                // The current `points` variable is `left_points_vec` at this stage.
                // This is a tricky recovery. For now, returning None.
                None
            }
        }
    }

    fn build_recursive(
        mut points: Vec<DataPoint<F, L>>,
        leaf_size: usize,
        k_dimensions: usize,
        sample_size_for_centroid: Option<usize>, 
    ) -> Option<Box<BallTreeNode<F, L>>> {
        if points.is_empty() {
            return None;
        }

        // For the "exact" build_recursive, we typically don't sample the centroid.
        // The `sample_size_for_centroid` param is mostly for the random variant.

        let (current_node_centroid, current_node_radius) =
            Self::calculate_bounding_ball(&points, k_dimensions, sample_size_for_centroid);

        // Base Case: Create a Leaf node
        if points.len() <= leaf_size {
            return Some(Box::new(BallTreeNode {
                centroid: current_node_centroid,
                radius: current_node_radius,
                contents: BallNodeContents::Leaf { points }, // `points` is moved here
            }));
        }


        // 1. Find the dimension of greatest spread (variance)
        let mut best_splitting_dimension = 0;
        let mut max_variance = F::neg_infinity(); // Start with negative infinity for proper comparison

        if k_dimensions == 0 { // Should have been caught earlier by `new`
             return Some(Box::new(BallTreeNode { // Fallback to leaf
                centroid: current_node_centroid,
                radius: current_node_radius,
                contents: BallNodeContents::Leaf { points },
            }));
        }

        for dim_idx in 0..k_dimensions {
            let dim_values: Vec<F> = points.iter().map(|p| p.features[dim_idx]).collect();
            let var = variance(&dim_values); // Your variance function

            // If variance is NaN (e.g., <2 points for sample variance), or if it's greater
            if !var.is_nan() && var > max_variance {
                max_variance = var;
                best_splitting_dimension = dim_idx;
            }
        }


        if max_variance.is_nan() || max_variance <= F::epsilon() { // Using epsilon for small variance check
            return Some(Box::new(BallTreeNode {
                centroid: current_node_centroid,
                radius: current_node_radius,
                contents: BallNodeContents::Leaf { points },
            }));
        }

        let median_idx = points.len() / 2;
        // Use select_nth_unstable_by to find the median element and partition around it in O(N)
        // This places the element that would be at median_idx in its sorted position.
        // All elements before it are <= and all elements after are >=.
        points.select_nth_unstable_by(median_idx, |a, b| {
            a.features[best_splitting_dimension]
                .partial_cmp(&b.features[best_splitting_dimension])
                .unwrap_or(std::cmp::Ordering::Equal)
        });


        // `split_off` creates `right_points`, `points` becomes `left_points`
        let right_points_vec = points.split_off(median_idx);
        let left_points_vec = points; // `points` now contains the left part

        // Handle degenerate splits: if one side is empty, make this node a leaf.
        // This requires recombining the points.
        if left_points_vec.is_empty() || right_points_vec.is_empty() {
            let mut combined_points = left_points_vec; // Move left points
            combined_points.extend(right_points_vec); // Extend with right points
            // Recalculate bounding ball just in case, though current_node_... should be for combined.
            let (leaf_centroid, leaf_radius) = Self::calculate_bounding_ball(&combined_points, k_dimensions, sample_size_for_centroid);
            return Some(Box::new(BallTreeNode {
                centroid: leaf_centroid,
                radius: leaf_radius,
                contents: BallNodeContents::Leaf { points: combined_points },
            }));
        }

        // 3. Recursively call build_recursive for left_points and right_points
        let left_child_opt = Self::build_recursive(left_points_vec, leaf_size, k_dimensions, sample_size_for_centroid);
        let right_child_opt = Self::build_recursive(right_points_vec, leaf_size, k_dimensions, sample_size_for_centroid);

        match (left_child_opt, right_child_opt) {
            (Some(left_child), Some(right_child)) => {

                Some(Box::new(BallTreeNode {
                    centroid: current_node_centroid,
                    radius: current_node_radius,
                    contents: BallNodeContents::Internal {
                        left_child,
                        right_child,
                    },
                }))
            }
            _ => {

                eprintln!("Warning: BallTree construction resulted in a None child from a non-empty partition. This should not happen.");

                None // Propagate None if children couldn't be formed.
            }
        }
    }

    /// Finds the N nearest neighbors to the `target_features`.
    pub fn find_nearest_neighbors(&self, target_features: &[F], n_neighbors: usize) -> Vec<DataPoint<F, L>> {
        if n_neighbors == 0 {
            return Vec::new();
        }
        let k_dimensions = match &self.root {
            Some(root_node) => root_node.centroid.len(),
            None => return Vec::new(), // Empty tree
        };
        if k_dimensions == 0 { return Vec::new(); }
        if target_features.len() != k_dimensions {
            eprintln!("Warning: Target point dimensionality ({}) does not match tree dimensionality ({}).", target_features.len(), k_dimensions);
            return Vec::new();
        }

        let mut best_n = KBestNeighbors::new(n_neighbors);
        Self::search_nn_recursive(&self.root, target_features, &mut best_n);
        best_n.into_sorted_points()
    }

    fn search_nn_recursive<'a>(
        current_node_opt: &'a Option<Box<BallTreeNode<F, L>>>,
        target_features: &[F],
        best_n: &mut KBestNeighbors<DataPoint<F, L>>,
    ) {
        let current_node = match current_node_opt {
            Some(node) => node,
            None => return, 
        };

        let dist_target_to_centroid = Self::euclidean_distance(target_features, &current_node.centroid);

        // Pruning: If the farthest known neighbor is closer than the closest possible point
        // in this ball (target to centroid distance MINUS radius), then prune.
        if let Some(farthest_dist) = best_n.current_farthest_distance() {
            if dist_target_to_centroid - current_node.radius.as_() > farthest_dist {
                return; // This ball cannot contain a better neighbor
            }
        }

        match &current_node.contents {
            BallNodeContents::Leaf { points } => {
                for point_data in points {
                    let dist = Self::euclidean_distance(target_features, &point_data.features);
                    best_n.add(dist, point_data.clone());
                }
            }
            BallNodeContents::Internal { left_child, right_child } => {
                let dist_target_to_left_centroid = Self::euclidean_distance(target_features, &left_child.centroid);
                let dist_target_to_right_centroid = Self::euclidean_distance(target_features, &right_child.centroid);

                // Search the child whose centroid is closer to the target first
                if dist_target_to_left_centroid < dist_target_to_right_centroid {
                    Self::search_nn_recursive(&Some(left_child.clone()), target_features, best_n); // Recurse on primary
                    // Check again before searching secondary (best_n might have improved)
                    if let Some(farthest_dist) = best_n.current_farthest_distance() {
                         // Prune secondary if its ball is too far
                        if dist_target_to_right_centroid - right_child.radius.as_() <= farthest_dist {
                            Self::search_nn_recursive(&Some(right_child.clone()), target_features, best_n);
                        }
                    } else { // Heap not full yet, must search
                        Self::search_nn_recursive(&Some(right_child.clone()), target_features, best_n);
                    }
                } else {
                    Self::search_nn_recursive(&Some(right_child.clone()), target_features, best_n); // Recurse on primary
                     if let Some(farthest_dist) = best_n.current_farthest_distance() {
                        if dist_target_to_left_centroid - left_child.radius.as_() <= farthest_dist {
                            Self::search_nn_recursive(&Some(left_child.clone()), target_features, best_n);
                        }
                    } else {
                         Self::search_nn_recursive(&Some(left_child.clone()), target_features, best_n);
                    }
                }
            }
        }
    }

    pub fn find_within_radius(&self, target_features: &[F], radius_search: f64) -> Vec<DataPoint<F, L>> {
        if radius_search < 0.0 { 
            return Vec::new();
        }
        let k_dimensions = match &self.root {
            Some(root_node) => root_node.centroid.len(),
            None => return Vec::new(), // Empty tree
        };
        if k_dimensions == 0 { return Vec::new(); }
        if target_features.len() != k_dimensions {
            eprintln!("Warning: Target point dimensionality ({}) does not match tree dimensionality ({}).", target_features.len(), k_dimensions);
            return Vec::new();
        }

        let mut found_points = Vec::new();
        Self::search_radius_recursive(&self.root, target_features, radius_search, &mut found_points);
        found_points
    }

    fn search_radius_recursive<'a>(
        current_node_opt: &'a Option<Box<BallTreeNode<F, L>>>,
        target_features: &[F],
        radius_search: f64,
        found_points: &mut Vec<DataPoint<F, L>>,
    ) {
        let current_node = match current_node_opt {
            Some(node) => node,
            None => return,
        };

        let dist_target_to_centroid = Self::euclidean_distance(target_features, &current_node.centroid);

        if dist_target_to_centroid - current_node.radius.as_() > radius_search {
            return; 
        }

        match &current_node.contents {
            BallNodeContents::Leaf { points } => {
                for point_data in points {
                    if Self::euclidean_distance(target_features, &point_data.features) <= radius_search {
                        found_points.push(point_data.clone());
                    }
                }
            }
            BallNodeContents::Internal { left_child, right_child } => {
                // No specific order needed, just recurse if not pruned
                Self::search_radius_recursive(&Some(left_child.clone()), target_features, radius_search, found_points);
                Self::search_radius_recursive(&Some(right_child.clone()), target_features, radius_search, found_points);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knn::knn::DataPoint; 
    use std::panic;

    // Helper to create DataPoint<f64, i32> for tests
    fn make_dp(features: Vec<f64>, label: i32) -> DataPoint<f64, i32> {
        DataPoint { features, label }
    }

    const EPSILON: f64 = 1e-9;

    // --- Construction Tests ---
    #[test]
    fn test_balltree_empty_input() {
        let points: Vec<DataPoint<f64, i32>> = vec![];
        assert!(BallTree::new(points, Some(10)).is_none());
    }

    #[test]
    fn test_balltree_zero_dimensionality() {
        let points = vec![make_dp(vec![], 0)];
        assert!(BallTree::new(points, Some(10)).is_none());
    }

    #[test]
    fn test_balltree_inconsistent_dimensionality() {
        let points = vec![make_dp(vec![1.0], 0), make_dp(vec![1.0, 2.0], 1)];
        let result = panic::catch_unwind(|| BallTree::new(points, Some(10)));
        assert!(result.is_err(), "BallTree::new should panic on inconsistent dimensions");
    }

    #[test]
    fn test_balltree_zero_leaf_size() {
        let points = vec![make_dp(vec![1.0], 0)];
        let result = panic::catch_unwind(|| BallTree::new(points, Some(0)));
        assert!(result.is_err(), "BallTree::new should panic on zero leaf_size");
    }

    #[test]
    fn test_balltree_single_point() {
        let points = vec![make_dp(vec![1.0, 2.0], 42)];
        let tree = BallTree::new(points.clone(), Some(10)).expect("Tree creation failed");
        let root_node = tree.root.as_ref().unwrap();

        assert_eq!(root_node.centroid, vec![1.0, 2.0]);
        assert!((root_node.radius - 0.0).abs() < EPSILON);
        if let BallNodeContents::Leaf { points: leaf_points } = &root_node.contents {
            assert_eq!(leaf_points.len(), 1);
            assert_eq!(leaf_points[0].features, vec![1.0, 2.0]);
            assert_eq!(leaf_points[0].label, 42);
        } else {
            panic!("Single point tree should be a leaf node.");
        }
    }

    #[test]
    fn test_balltree_multiple_points_become_leaf() {
        let points = vec![
            make_dp(vec![1.0, 1.0], 1),
            make_dp(vec![2.0, 2.0], 2),
            make_dp(vec![3.0, 3.0], 3),
        ];
        let tree = BallTree::new(points.clone(), Some(5)).expect("Tree creation failed"); // leaf_size > num_points
        let root_node = tree.root.as_ref().unwrap();

        // Expected centroid: ( (1+2+3)/3, (1+2+3)/3 ) = (2.0, 2.0)
        assert_eq!(root_node.centroid, vec![2.0, 2.0]);
        // Expected radius: dist((2,2), (1,1)) = sqrt(1^2+1^2) = sqrt(2)
        // or dist((2,2), (3,3)) = sqrt(1^2+1^2) = sqrt(2)
        assert!((root_node.radius - (2.0_f64).sqrt()).abs() < EPSILON);

        if let BallNodeContents::Leaf { points: leaf_points } = &root_node.contents {
            assert_eq!(leaf_points.len(), 3);
            // Check if all original points are present (order might change due to internal processing)
            assert!(leaf_points.iter().any(|p| p.label == 1));
            assert!(leaf_points.iter().any(|p| p.label == 2));
            assert!(leaf_points.iter().any(|p| p.label == 3));
        } else {
            panic!("Should be a leaf node.");
        }
    }

    #[test]
    fn test_balltree_points_force_split() {
        let points = vec![
            make_dp(vec![1.0, 0.0], 1),
            make_dp(vec![2.0, 0.0], 2),
            make_dp(vec![10.0, 0.0], 3),
            make_dp(vec![11.0, 0.0], 4),
            make_dp(vec![12.0, 0.0], 5),
        ];
        // leaf_size is 2, so 5 points should force splits.
        let tree = BallTree::new(points.clone(), Some(2)).expect("Tree creation failed");
        let root_node = tree.root.as_ref().unwrap();

        if let BallNodeContents::Internal { .. } = &root_node.contents {
            // Good, it's an internal node.
            // Further structural checks are complex and depend heavily on the splitting heuristic.
            // We'll rely on search tests to validate overall correctness.
        } else {
            panic!("Should be an internal node due to splitting.");
        }
    }

    #[test]
    fn test_balltree_identical_points() {
        let points = vec![
            make_dp(vec![5.0, 5.0], 1),
            make_dp(vec![5.0, 5.0], 2), // Same features, different label
            make_dp(vec![5.0, 5.0], 3),
        ];
        // leaf_size is 1, but identical points should ideally end up in a leaf quickly.
        // The variance will be 0 (or NaN for <2 points), forcing a leaf.
        let tree = BallTree::new(points.clone(), Some(1)).expect("Tree creation failed");
        let root_node = tree.root.as_ref().unwrap();

        assert_eq!(root_node.centroid, vec![5.0, 5.0]);
        assert!((root_node.radius - 0.0).abs() < EPSILON);
        if let BallNodeContents::Leaf { points: leaf_points } = &root_node.contents {
            assert_eq!(leaf_points.len(), 3);
        } else {
            panic!("Identical points should result in a leaf node.");
        }
    }

    // --- Nearest Neighbor Search Tests ---
    fn build_sample_tree() -> BallTree<f64, i32> {
        let points = vec![
            make_dp(vec![1.0, 1.0], 1), // P1
            make_dp(vec![2.0, 2.0], 2), // P2
            make_dp(vec![1.0, 2.0], 3), // P3 (closer to P2 than P1 is to P2)
            make_dp(vec![10.0, 10.0], 4),// P4
            make_dp(vec![11.0, 11.0], 5),// P5
        ];
        BallTree::new(points, Some(2)).expect("Sample tree creation failed")
    }

    #[test]
    fn test_nn_empty_tree() {
        let tree: BallTree<f64, i32> = BallTree { root: None };
        let neighbors = tree.find_nearest_neighbors(&[1.0], 1);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_nn_zero_neighbors_requested() {
        let tree = build_sample_tree();
        let neighbors = tree.find_nearest_neighbors(&[1.5, 1.5], 0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_nn_simple_case_k1() {
        let tree = build_sample_tree();
        let target = vec![1.8, 1.8]; // Closest to P2 (2,2)
        let neighbors = tree.find_nearest_neighbors(&target, 1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].label, 2);
    }

    #[test]
    fn test_nn_simple_case_k_gt_1() {
        let tree = build_sample_tree();
        let target = vec![1.2, 1.2]; // Closest to P1, then P3, then P2
        let neighbors = tree.find_nearest_neighbors(&target, 3);
        assert_eq!(neighbors.len(), 3);
        // Expected order: P1, P3, P2 (or P2, P3 depending on tie-breaking if distances are equal)
        // dist(T,P1) = dist((1.2,1.2), (1,1)) = sqrt(0.2^2+0.2^2) = sqrt(0.08)
        // dist(T,P3) = dist((1.2,1.2), (1,2)) = sqrt(0.2^2+0.8^2) = sqrt(0.04+0.64) = sqrt(0.68)
        // dist(T,P2) = dist((1.2,1.2), (2,2)) = sqrt(0.8^2+0.8^2) = sqrt(0.64+0.64) = sqrt(1.28)
        assert_eq!(neighbors[0].label, 1); // P1
        assert_eq!(neighbors[1].label, 3); // P3
        assert_eq!(neighbors[2].label, 2); // P2
    }

    #[test]
    fn test_nn_k_larger_than_dataset() {
        let tree = build_sample_tree(); // 5 points
        let target = vec![0.0, 0.0];
        let neighbors = tree.find_nearest_neighbors(&target, 10); // Request 10
        assert_eq!(neighbors.len(), 5); // Should return all 5 points, sorted
    }

    #[test]
    fn test_nn_dim_mismatch() {
        let tree = build_sample_tree(); // 2D tree
        let target_1d = vec![0.0];
        let neighbors = tree.find_nearest_neighbors(&target_1d, 1);
        assert!(neighbors.is_empty(), "Should return empty on dimension mismatch");
    }

    // --- Radius Search Tests ---
    #[test]
    fn test_radius_empty_tree() {
        let tree: BallTree<f64, i32> = BallTree { root: None };
        let neighbors = tree.find_within_radius(&[1.0], 1.0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_radius_negative_radius() {
        let tree = build_sample_tree();
        let neighbors = tree.find_within_radius(&[1.0, 1.0], -0.5);
        assert!(neighbors.is_empty());
    }

     #[test]
    fn test_radius_zero_radius() {
        let tree = build_sample_tree();
        // Target is P1 (1,1)
        let neighbors = tree.find_within_radius(&[1.0, 1.0], 0.0);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.iter().any(|p| p.label == 1));

        // Target not exactly on a point
        let neighbors_miss = tree.find_within_radius(&[1.0001, 1.0001], 0.0);
        assert!(neighbors_miss.is_empty());
    }

    #[test]
    fn test_radius_simple_case() {
        let tree = build_sample_tree();
        let target = vec![1.5, 1.5];
        // P1 (1,1) dist = sqrt(0.5^2+0.5^2) = sqrt(0.5) ~ 0.707
        // P2 (2,2) dist = sqrt(0.5^2+0.5^2) = sqrt(0.5) ~ 0.707
        // P3 (1,2) dist = sqrt(0.5^2+0.5^2) = sqrt(0.5) ~ 0.707

        // Radius that includes P1, P2, P3
        let neighbors1 = tree.find_within_radius(&target, 0.8);
        assert_eq!(neighbors1.len(), 3);
        assert!(neighbors1.iter().any(|p| p.label == 1));
        assert!(neighbors1.iter().any(|p| p.label == 2));
        assert!(neighbors1.iter().any(|p| p.label == 3));

        // Radius that includes only one of them (e.g., if target was exactly on P1)
        let neighbors2 = tree.find_within_radius(&[1.0, 1.0], 0.1); // Only P1
        assert_eq!(neighbors2.len(), 1);
        assert!(neighbors2.iter().any(|p| p.label == 1));

        // Radius that includes none of the close ones
        let neighbors3 = tree.find_within_radius(&target, 0.1);
        assert!(neighbors3.is_empty());
    }

    #[test]
    fn test_radius_dim_mismatch() {
        let tree = build_sample_tree(); // 2D tree
        let target_1d = vec![0.0];
        let neighbors = tree.find_within_radius(&target_1d, 1.0);
        assert!(neighbors.is_empty(), "Should return empty on dimension mismatch for radius search");
    }
}
