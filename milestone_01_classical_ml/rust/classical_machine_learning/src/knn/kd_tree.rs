//! This module will contain the k-d tree implementation.

use num_traits::{Float, AsPrimitive}; // Ensure this is at the top
use crate::common_types::DataPoint;
use rand::seq::SliceRandom; // For random sampling
use rand::thread_rng; // For random number generator
use super::heap_utils::KBestNeighbors; // Import shared heap utilities

#[derive(Debug)] 
pub struct KdTree<F, L> {
    pub root: Option<Box<TreeNode<F, L>>>, 
}
#[derive(Debug)] 
pub struct TreeNode<F,L> { 
    point: DataPoint<F,L>,
    splitting_dimension: usize,
    /// The left child of this node.
    left: Option<Box<TreeNode<F,L>>>,
    /// The right child of this node.
    right: Option<Box<TreeNode<F,L>>>,
}

impl<F, L> KdTree<F, L>
where
    F: Float + AsPrimitive<f64> + Clone + std::iter::Sum, 
    L: Clone,
{
    // Default values for the `new` constructor
    const DEFAULT_AUTO_DECISION_THRESHOLD: usize = 1000;
    const DEFAULT_SAMPLE_SIZE_FOR_RANDOM: usize = 101; 
    const DEFAULT_FALLBACK_THRESHOLD_FOR_RANDOM: usize = 25;

    pub fn new_exact_median(mut points: Vec<DataPoint<F, L>>) -> Option<Self> {
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

        let root_node = Self::build_recursive_exact_median(&mut points, k, 0);
        Some(KdTree { root: root_node })
    }

    pub fn new_random_sample_pivot(
        points: Vec<DataPoint<F, L>>,
        sample_size: usize,
        fallback_threshold: usize,
    ) -> Option<Self> {
        if sample_size == 0 {
            panic!("sample_size for new_random_sample_pivot must be greater than 0.");
        }
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

        // We pass the initial `points` Vec by value to the recursive helper,
        // as it will be consumed and partitioned into new Vecs for children.
        let root_node = Self::build_recursive_random_sample_pivot(
            points,
            k,
            0,
            sample_size,
            fallback_threshold,
        );
        Some(KdTree { root: root_node })
    }


    pub fn new(
        points: Vec<DataPoint<F, L>>,
        auto_decision_threshold_opt: Option<usize>,
        sample_size_for_random_opt: Option<usize>,
        fallback_threshold_for_random_opt: Option<usize>,
    ) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let auto_thresh = auto_decision_threshold_opt.unwrap_or(Self::DEFAULT_AUTO_DECISION_THRESHOLD);
        let sample_size = sample_size_for_random_opt.unwrap_or(Self::DEFAULT_SAMPLE_SIZE_FOR_RANDOM);
        let fallback_thresh = fallback_threshold_for_random_opt.unwrap_or(Self::DEFAULT_FALLBACK_THRESHOLD_FOR_RANDOM);

        if points.len() < auto_thresh {
            Self::new_exact_median(points)
        } else {
            Self::new_random_sample_pivot(points, sample_size, fallback_thresh)
        }
    }
    fn build_recursive_exact_median(
        points_slice: &mut [DataPoint<F, L>],
        k_dimensions: usize,
        depth: usize,
    ) -> Option<Box<TreeNode<F, L>>> {
        if points_slice.is_empty() {
            return None;
        }

        let splitting_dimension = depth % k_dimensions;
        points_slice.sort_by(|a, b| {
            a.features[splitting_dimension]
                .partial_cmp(&b.features[splitting_dimension])
                .unwrap_or(std::cmp::Ordering::Equal) // Fallback for non-total orders (e.g. NaN)
        });

        let median_idx = points_slice.len() / 2;

        let median_point_data = points_slice[median_idx].clone();

        let left_child = Self::build_recursive_exact_median(&mut points_slice[0..median_idx], k_dimensions, depth + 1);
        
        let right_child = if median_idx + 1 < points_slice.len() {
            Self::build_recursive_exact_median(&mut points_slice[median_idx + 1 ..], k_dimensions, depth + 1)
        } else {
            None
        };

        Some(Box::new(TreeNode {
            point: median_point_data,
            splitting_dimension,
            left: left_child,
            right: right_child,
        }))
    }

    /// Recursive helper to build the tree using random sample pivot selection.
    /// Takes ownership of `points` as it partitions them into new Vecs for children.
    fn build_recursive_random_sample_pivot(
        mut points: Vec<DataPoint<F, L>>, // Takes ownership
        k_dimensions: usize,
        depth: usize,
        sample_size: usize,
        fallback_threshold: usize,
    ) -> Option<Box<TreeNode<F, L>>> {
        if points.is_empty() {
            return None;
        }

        // Fallback to exact median for small lists or if sample_size is too large/ineffective
        if points.len() < fallback_threshold || points.len() <= sample_size {
            // `build_recursive_exact_median` expects a slice, so we provide one from `points`
            return Self::build_recursive_exact_median(&mut points, k_dimensions, depth);
        }

        let splitting_dimension = depth % k_dimensions;

        // 1. Randomly select `sample_size` points
        let mut rng = thread_rng();
        let mut sample_points: Vec<DataPoint<F, L>> = points
            .choose_multiple(&mut rng, sample_size)
            .cloned() // Clone the sampled points so we have ownership
            .collect();

        // 2. Sort the sample and pick its median as the pivot for the current node
        sample_points.sort_by(|a, b| {
            a.features[splitting_dimension]
                .partial_cmp(&b.features[splitting_dimension])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // The median of the sample becomes the point for the current node
        let node_point_data = sample_points[sample_points.len() / 2].clone();

        // 3. Partition the original `points` list based on `node_point_data`
        let mut left_children_points: Vec<DataPoint<F, L>> = Vec::new();
        let mut right_children_points: Vec<DataPoint<F, L>> = Vec::new();

        for point in points { // Consumes the input `points` Vec
            if point.features[splitting_dimension] < node_point_data.features[splitting_dimension] {
                left_children_points.push(point);
            } else {
                // Points with feature value equal to the pivot's on the splitting dimension
                // go to the right. This also handles the case where `point` might be
                // identical to `node_point_data` if it was in the original list.
                right_children_points.push(point);
            }
        }

        // 4. Recursively build children
        let left_child = Self::build_recursive_random_sample_pivot(
            left_children_points, k_dimensions, depth + 1, sample_size, fallback_threshold
        );
        let right_child = Self::build_recursive_random_sample_pivot(
            right_children_points, k_dimensions, depth + 1, sample_size, fallback_threshold
        );

        Some(Box::new(TreeNode {
            point: node_point_data, // The median of the sample
            splitting_dimension,
            left: left_child,
            right: right_child,
        }))
    }

    /// Calculates the Euclidean distance between two points.
    fn euclidean_distance(p1_features: &[F], p2_features: &[F]) -> f64 {
        let sum_sq_diff: F = p1_features.iter().zip(p2_features.iter()).map(|(&a, &b)| {
            let diff = a - b;
            diff * diff
        }).sum(); // sum() requires F: Sum, which Float provides via Add + Zero.
        sum_sq_diff.as_().sqrt() // F to f64, then sqrt
    }

    pub fn find_nearest_neighbors(&self, target_features: &[F], n_neighbors: usize) -> Vec<DataPoint<F, L>> {
        if n_neighbors == 0 {
            return Vec::new();
        }

        let tree_dimensionality = if let Some(root_node) = &self.root {
            if root_node.point.features.is_empty() { return Vec::new(); } // Should not happen with valid tree
            root_node.point.features.len()
        } else {
            return Vec::new(); // Empty tree
        };

        if tree_dimensionality == 0 { return Vec::new(); }
        if target_features.len() != tree_dimensionality {
            // Consider returning Result<_, Error> for production code
            eprintln!("Warning: Target point dimensionality ({}) does not match tree dimensionality ({}).", target_features.len(), tree_dimensionality);
            return Vec::new();
        }

        let mut best_n = KBestNeighbors::new(n_neighbors);
        Self::search_recursive(&self.root, target_features, &mut best_n, tree_dimensionality);
        best_n.into_sorted_points()
    }

    pub fn find_within_radius(&self, target_features: &[F], radius: f64) -> Vec<DataPoint<F, L>> {
        if radius <= 0.0 {
            return Vec::new();
        }

        let tree_dimensionality = if let Some(root_node) = &self.root {
            if root_node.point.features.is_empty() { return Vec::new(); }
            root_node.point.features.len()
        } else {
            return Vec::new(); // Empty tree
        };

        if tree_dimensionality == 0 { return Vec::new(); }
        if target_features.len() != tree_dimensionality {
            eprintln!("Warning: Target point dimensionality ({}) does not match tree dimensionality ({}).", target_features.len(), tree_dimensionality);
            return Vec::new();
        }

        let mut found_points = Vec::new();
        Self::search_radius_recursive(&self.root, target_features, radius, &mut found_points, tree_dimensionality);
        found_points
    }

    /// Recursive helper function for radius search.
    fn search_radius_recursive(
        current_node_opt: &Option<Box<TreeNode<F, L>>>,
        target_features: &[F],
        radius: f64,
        found_points: &mut Vec<DataPoint<F, L>>,
        k_dimensions: usize,
    ) {
        let current_node = match current_node_opt {
            Some(node) => node,
            None => return,
        };

        let dist_to_node_point = Self::euclidean_distance(target_features, &current_node.point.features);
        if dist_to_node_point <= radius {
            found_points.push(current_node.point.clone());
        }

        let splitting_dim = current_node.splitting_dimension;
        let node_coord_at_dim = current_node.point.features[splitting_dim];
        let target_coord_at_dim = target_features[splitting_dim];

        let (primary_child, secondary_child) = if target_coord_at_dim < node_coord_at_dim {
            (&current_node.left, &current_node.right)
        } else {
            (&current_node.right, &current_node.left)
        };

        Self::search_radius_recursive(primary_child, target_features, radius, found_points, k_dimensions);

        let dist_to_hyperplane = (target_coord_at_dim - node_coord_at_dim).abs().as_();
        if dist_to_hyperplane <= radius { // Note: comparison is with radius directly
            Self::search_radius_recursive(secondary_child, target_features, radius, found_points, k_dimensions);
        }
    }

    /// Recursive helper function for nearest neighbor search.
    fn search_recursive(
        current_node_opt: &Option<Box<TreeNode<F, L>>>,
        target_features: &[F],
        best_n: &mut KBestNeighbors<DataPoint<F, L>>,
        k_dimensions: usize, // Pass k_dimensions for splitting_dimension calculation
    ) {
        let current_node = match current_node_opt {
            Some(node) => node,
            None => return, // Base case: Reached a null child (leaf)
        };

        // splitting_dimension is stored in the node, no need to recalculate from depth
        let splitting_dim = current_node.splitting_dimension; 
        let node_coord_at_dim = current_node.point.features[splitting_dim];
        let target_coord_at_dim = target_features[splitting_dim];

        // Determine which child subtree to visit first ("primary" or "closer" side)
        let (primary_child, secondary_child) = if target_coord_at_dim < node_coord_at_dim {
            (&current_node.left, &current_node.right)
        } else {
            (&current_node.right, &current_node.left)
        };

        Self::search_recursive(primary_child, target_features, best_n, k_dimensions);

        // Check the current node's point
        let dist_to_node_point = Self::euclidean_distance(target_features, &current_node.point.features);
        best_n.add(dist_to_node_point, current_node.point.clone());

        // Check if the secondary child subtree needs to be searched (pruning)
        let dist_to_hyperplane = (target_coord_at_dim - node_coord_at_dim).abs().as_(); // Perpendicular distance

        if best_n.current_farthest_distance().map_or(true, |radius| dist_to_hyperplane < radius) {
            Self::search_recursive(secondary_child, target_features, best_n, k_dimensions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{KdTree, TreeNode, KBestNeighbors}; // Added KBestNeighbors, HeapElement for potential direct tests
    use crate::common_types::DataPoint;
    use std::panic;

    // Helper to create DataPoint<f64, i32> for tests
    fn make_dp(features: Vec<f64>, label: i32) -> DataPoint<f64, i32> {
        DataPoint { features, label }
    }

    // Helper to check node properties
    fn check_node_props<F: PartialEq + std::fmt::Debug, L: PartialEq + std::fmt::Debug>(
        node_opt: &Option<Box<TreeNode<F, L>>>,
        expected_features: &[F],
        expected_label: L,
        expected_dim: usize,
    ) {
        let node = node_opt.as_ref().expect("Node should exist but was None");
        assert_eq!(&node.point.features, expected_features, "Node point features mismatch");
        assert_eq!(node.point.label, expected_label, "Node point label mismatch");
        assert_eq!(node.splitting_dimension, expected_dim, "Node splitting dimension mismatch");
    }

    #[test]
    fn test_empty_input() {
        let points: Vec<DataPoint<f64, i32>> = vec![];
        assert!(KdTree::new_exact_median(points.clone()).is_none());
        assert!(KdTree::new_random_sample_pivot(points.clone(), 1, 10).is_none());
        assert!(KdTree::new(points.clone(), Some(100), Some(1), Some(10)).is_none());
    }

    #[test]
    fn test_zero_dimensionality_points() {
        let points = vec![make_dp(vec![], 0)];
        assert!(KdTree::new_exact_median(points.clone()).is_none());
        assert!(KdTree::new_random_sample_pivot(points.clone(), 1, 10).is_none());
        assert!(KdTree::new(points.clone(), Some(100), Some(1), Some(10)).is_none());
    }

    #[test]
    fn test_inconsistent_dimensionality() {
        let points = vec![make_dp(vec![1.0, 2.0], 0), make_dp(vec![3.0], 1)];
        
        let result_exact = panic::catch_unwind(|| KdTree::new_exact_median(points.clone()));
        assert!(result_exact.is_err(), "new_exact_median should panic on inconsistent dimensions");

        let result_random = panic::catch_unwind(|| KdTree::new_random_sample_pivot(points.clone(), 1, 10));
        assert!(result_random.is_err(), "new_random_sample_pivot should panic on inconsistent dimensions");

        let result_auto = panic::catch_unwind(|| KdTree::new(points.clone(), Some(100), Some(1), Some(10)));
        assert!(result_auto.is_err(), "new (auto) should panic on inconsistent dimensions");
    }

    #[test]
    fn test_single_point_tree() {
        let points = vec![make_dp(vec![1.0, 2.0], 42)];
        
        // Exact Median
        let tree_exact = KdTree::new_exact_median(points.clone()).expect("Tree creation failed for single point (exact)");
        check_node_props(&tree_exact.root, &[1.0, 2.0], 42, 0);
        assert!(tree_exact.root.as_ref().unwrap().left.is_none(), "Single node tree should have no left child (exact)");
        assert!(tree_exact.root.as_ref().unwrap().right.is_none(), "Single node tree should have no right child (exact)");

        // Random Pivot (should fallback to exact for a single point if fallback_threshold is reasonable)
        let tree_random = KdTree::new_random_sample_pivot(points.clone(), 1, 10).expect("Tree creation failed for single point (random)");
        check_node_props(&tree_random.root, &[1.0, 2.0], 42, 0);
        assert!(tree_random.root.as_ref().unwrap().left.is_none(), "Single node tree should have no left child (random)");
        assert!(tree_random.root.as_ref().unwrap().right.is_none(), "Single node tree should have no right child (random)");

        // Auto (should use exact for a single point)
        let tree_auto = KdTree::new(points.clone(), Some(100), Some(1), Some(10)).expect("Tree creation failed for single point (auto)");
        check_node_props(&tree_auto.root, &[1.0, 2.0], 42, 0);
        assert!(tree_auto.root.as_ref().unwrap().left.is_none(), "Single node tree should have no left child (auto)");
        assert!(tree_auto.root.as_ref().unwrap().right.is_none(), "Single node tree should have no right child (auto)");
    }

    #[test]
    fn test_exact_median_construction_2d() {
        let points = vec![
            make_dp(vec![2.0, 3.0], 1), // P1
            make_dp(vec![5.0, 4.0], 2), // P2
            make_dp(vec![9.0, 6.0], 3), // P3
            make_dp(vec![4.0, 7.0], 4), // P4
            make_dp(vec![8.0, 1.0], 5), // P5
            make_dp(vec![7.0, 2.0], 6), // P6
        ];

        let tree = KdTree::new_exact_median(points).expect("Exact median tree creation failed");
        
        // Root: P6(7,2), dim=0
        check_node_props(&tree.root, &[7.0, 2.0], 6, 0);
        let root_node = tree.root.as_ref().unwrap();

        // Left child of root: P2(5,4), dim=1
        check_node_props(&root_node.left, &[5.0, 4.0], 2, 1);
        let left_node = root_node.left.as_ref().unwrap();

        // Right child of root: P3(9,6), dim=1
        check_node_props(&root_node.right, &[9.0, 6.0], 3, 1);
        let right_node = root_node.right.as_ref().unwrap();

        // Children of P2(5,4)
        // LL: P1(2,3), dim=0 (2%2)
        check_node_props(&left_node.left, &[2.0, 3.0], 1, 0);
        assert!(left_node.left.as_ref().unwrap().left.is_none());
        assert!(left_node.left.as_ref().unwrap().right.is_none());

        // LR: P4(4,7), dim=0 (2%2)
        check_node_props(&left_node.right, &[4.0, 7.0], 4, 0);
        assert!(left_node.right.as_ref().unwrap().left.is_none());
        assert!(left_node.right.as_ref().unwrap().right.is_none());

        // Children of P3(9,6)
        // RL: P5(8,1), dim=0 (2%2)
        check_node_props(&right_node.left, &[8.0, 1.0], 5, 0);
        assert!(right_node.left.as_ref().unwrap().left.is_none());
        assert!(right_node.left.as_ref().unwrap().right.is_none());

        // RR: None
        assert!(right_node.right.is_none());
    }

    #[test]
    fn test_random_pivot_panic_on_zero_sample_size() {
        let points = vec![make_dp(vec![1.0], 0)];
        let result = panic::catch_unwind(|| KdTree::new_random_sample_pivot(points, 0, 10));
        assert!(result.is_err(), "new_random_sample_pivot should panic if sample_size is 0");
    }

    #[test]
    fn test_random_pivot_fallback_to_exact() {
        let points_small = vec![
            make_dp(vec![2.0, 3.0], 1),
            make_dp(vec![5.0, 4.0], 2),
        ];

        // Case 1: points.len() < fallback_threshold
        let tree_fallback1 = KdTree::new_random_sample_pivot(points_small.clone(), 1, 5)
            .expect("Tree creation failed (fallback1)");
        // Root should be (5.0, 4.0) if sorted by x-axis (dim 0), then (2.0,3.0) is left.
        // Or (2.0,3.0) if median_idx = len/2 = 1, then points_slice[0] is (2.0,3.0)
        // Let's trace build_recursive_exact_median for points_small:
        // points_slice = [(2,3), (5,4)], k=2, depth=0. dim=0.
        // sorted: [(2,3), (5,4)]. median_idx = 2/2 = 1. median_point_data = (5,4).
        // left_child from [(2,3)]. right_child from [].
        check_node_props(&tree_fallback1.root, &[5.0, 4.0], 2, 0);
        check_node_props(&tree_fallback1.root.as_ref().unwrap().left, &[2.0, 3.0], 1, 1);
        assert!(tree_fallback1.root.as_ref().unwrap().right.is_none());

        // Case 2: points.len() <= sample_size
        let tree_fallback2 = KdTree::new_random_sample_pivot(points_small.clone(), 3, 1)
            .expect("Tree creation failed (fallback2)");
        check_node_props(&tree_fallback2.root, &[5.0, 4.0], 2, 0); // Same structure as exact
        check_node_props(&tree_fallback2.root.as_ref().unwrap().left, &[2.0, 3.0], 1, 1);
        assert!(tree_fallback2.root.as_ref().unwrap().right.is_none());
    }

    #[test]
    fn test_random_pivot_uses_sampling_builds_tree() {
        // Hard to test exact structure due to randomness, but we can check it builds.
        let points: Vec<_> = (0..30).map(|i| make_dp(vec![i as f64, (30 - i) as f64], i)).collect();
        
        let tree = KdTree::new_random_sample_pivot(points.clone(), 5, 10);
        assert!(tree.is_some(), "Random pivot tree creation failed for larger dataset");
        let tree_unwrapped = tree.unwrap();
        assert!(tree_unwrapped.root.is_some(), "Root node should exist");
        // Check if root point is one of the input points
        let root_point_features = &tree_unwrapped.root.as_ref().unwrap().point.features;
        assert!(points.iter().any(|p| &p.features == root_point_features), "Root point not from input set");
    }

    #[test]
    fn test_new_dispatch_exact_with_custom_params() {
        let points = vec![
            make_dp(vec![2.0, 3.0], 1),
            make_dp(vec![5.0, 4.0], 2),
        ];
        // auto_decision_threshold is 5, points.len() is 2, so should use exact.
        let tree_auto = KdTree::new(points.clone(), Some(5), Some(3), Some(1))
            .expect("Auto tree creation failed (should use exact)");
        
        // Expected structure from exact_median for these 2 points:
        // Root: (5,4) dim 0. Left: (2,3) dim 1. Right: None.
        check_node_props(&tree_auto.root, &[5.0, 4.0], 2, 0);
        check_node_props(&tree_auto.root.as_ref().unwrap().left, &[2.0, 3.0], 1, 1);
        assert!(tree_auto.root.as_ref().unwrap().right.is_none());
    }

    #[test]
    fn test_new_dispatch_exact_with_defaults() {
        let points = vec![
            make_dp(vec![2.0, 3.0], 1),
            make_dp(vec![5.0, 4.0], 2), // points.len() = 2
        ];
        // Using None for params, so defaults are used.
        // DEFAULT_AUTO_DECISION_THRESHOLD is 1000. 2 < 1000, so should use exact.
        let tree_auto = KdTree::new(points.clone(), None, None, None)
            .expect("New with defaults failed (should use exact)");
        
        check_node_props(&tree_auto.root, &[5.0, 4.0], 2, 0);
        check_node_props(&tree_auto.root.as_ref().unwrap().left, &[2.0, 3.0], 1, 1);
        assert!(tree_auto.root.as_ref().unwrap().right.is_none());
    }

    #[test]
    fn test_new_dispatch_random_with_custom_params() {
        let points: Vec<_> = (0..15).map(|i| make_dp(vec![i as f64], i)).collect();
        // auto_decision_threshold is 10, points.len() is 15, so should use random.
        // sample_size=3, fallback_threshold_for_random=5
        let tree_auto = KdTree::new(points.clone(), Some(10), Some(3), Some(5));
        assert!(tree_auto.is_some(), "Auto tree creation failed (should use random)");
        let tree_unwrapped = tree_auto.unwrap();
        assert!(tree_unwrapped.root.is_some(), "Root node should exist for auto random");

        // Check if root point is one of the input points
        let root_point_features = &tree_unwrapped.root.as_ref().unwrap().point.features;
        assert!(points.iter().any(|p| &p.features == root_point_features), "Auto random root point not from input set");
    }

    #[test]
    fn test_new_dispatch_random_with_defaults() {
        // Make enough points to exceed default auto_decision_threshold (1000)
        let num_points = KdTree::<f64, i32>::DEFAULT_AUTO_DECISION_THRESHOLD + 100;
        let points: Vec<_> = (0..num_points).map(|i| make_dp(vec![i as f64], i.try_into().unwrap())).collect();
        
        let tree_auto = KdTree::new(points.clone(), None, None, None); // Use defaults
        assert!(tree_auto.is_some(), "New with defaults failed (should use random)");
        let tree_unwrapped = tree_auto.unwrap();
        assert!(tree_unwrapped.root.is_some(), "Root node should exist for new random with defaults");

        let root_point_features = &tree_unwrapped.root.as_ref().unwrap().point.features;
        assert!(points.iter().any(|p| &p.features == root_point_features), "New random with defaults root point not from input set");
    }

     #[test]
    fn test_three_points_exact_median() {
        let points = vec![
            make_dp(vec![1.0], 1),
            make_dp(vec![2.0], 2),
            make_dp(vec![3.0], 3),
        ];
        let tree = KdTree::new_exact_median(points).unwrap();
        // dim 0. Sorted: [1], [2], [3]. Median idx 3/2=1. Median point [2.0]
        check_node_props(&tree.root, &[2.0], 2, 0);
        let root_node = tree.root.as_ref().unwrap();
        // Left: [1.0]. dim 0 (depth 1 % k 1 = 0)
        check_node_props(&root_node.left, &[1.0], 1, 0);
        // Right: [3.0]. dim 0
        check_node_props(&root_node.right, &[3.0], 3, 0);
    }

    #[test]
    fn test_k_best_neighbors_logic() {
        let mut k_best = KBestNeighbors::new(3); // Keep 3 best (smallest distance)

        // Add some items (distance, data_label)
        k_best.add(10.0, "P10"); // Heap: [(10, P10)]
        k_best.add(5.0, "P5");  // Heap: [(10,P10), (5,P5)] -> after sift: [(10,P10), (5,P5)] (max heap)
        k_best.add(12.0, "P12"); // Heap: [(12,P12), (5,P5), (10,P10)]
        
        assert_eq!(k_best.len(), 3);
        assert_eq!(k_best.current_farthest_distance(), Some(12.0)); // Farthest of the 3 best

        k_best.add(4.0, "P4");  // P4 (4.0) is better than P12 (12.0)
                                // Heap before pop: [(12,P12), (5,P5), (10,P10)] -> add (4,P4)
                                // Peek is (12,P12). 4.0 < 12.0. Pop (12,P12). Push (4,P4).
                                // Heap becomes elements with distances 4,5,10. Max is 10.
        assert_eq!(k_best.len(), 3);
        assert_eq!(k_best.current_farthest_distance(), Some(10.0));

        k_best.add(15.0, "P15"); // P15 (15.0) is not better than P10 (10.0)
        assert_eq!(k_best.len(), 3);
        assert_eq!(k_best.current_farthest_distance(), Some(10.0));

        let sorted_points = k_best.into_sorted_points();
        assert_eq!(sorted_points, vec!["P4", "P5", "P10"]);
    }

    // --- Tests for find_within_radius ---
    #[test]
    fn test_radius_search_empty_tree() {
        let tree: KdTree<f64, i32> = KdTree { root: None };
        let neighbors = tree.find_within_radius(&[1.0, 2.0], 5.0);
        assert!(neighbors.is_empty());
    }

    #[test]
    fn test_radius_search_negative_or_zero_radius() {
        let points = vec![make_dp(vec![1.0], 0)];
        let tree = KdTree::new_exact_median(points).unwrap();
        assert!(tree.find_within_radius(&[1.0], 0.0).is_empty());
        assert!(tree.find_within_radius(&[1.0], -1.0).is_empty());
    }

    #[test]
    fn test_radius_search_simple_case() {
        let points = vec![
            make_dp(vec![1.0, 1.0], 1), // dist to (3,3) = sqrt(2^2+2^2) = sqrt(8) ~ 2.82
            make_dp(vec![2.0, 2.0], 2), // dist to (3,3) = sqrt(1^2+1^2) = sqrt(2) ~ 1.41
            make_dp(vec![3.0, 3.0], 3), // dist to (3,3) = 0
            make_dp(vec![4.0, 4.0], 4), // dist to (3,3) = sqrt(1^2+1^2) = sqrt(2) ~ 1.41
            make_dp(vec![5.0, 5.0], 5), // dist to (3,3) = sqrt(2^2+2^2) = sqrt(8) ~ 2.82
            make_dp(vec![10.0, 10.0], 6),// dist to (3,3) = sqrt(7^2+7^2) = sqrt(98) ~ 9.89
        ];
        let tree = KdTree::new_exact_median(points).unwrap();

        let target = vec![3.0, 3.0];
        
        // Radius 0.5: should only find point (3,3)
        let neighbors_r0_5 = tree.find_within_radius(&target, 0.5);
        assert_eq!(neighbors_r0_5.len(), 1);
        assert!(neighbors_r0_5.iter().any(|p| p.label == 3));

        // Radius 2.0: should find (2,2), (3,3), (4,4)
        let neighbors_r2_0 = tree.find_within_radius(&target, 2.0);
        assert_eq!(neighbors_r2_0.len(), 3);
        assert!(neighbors_r2_0.iter().any(|p| p.label == 2));
        assert!(neighbors_r2_0.iter().any(|p| p.label == 3));
        assert!(neighbors_r2_0.iter().any(|p| p.label == 4));

        // Radius 3.0: should find (1,1), (2,2), (3,3), (4,4), (5,5)
        let neighbors_r3_0 = tree.find_within_radius(&target, 3.0); // sqrt(8) is approx 2.82
        assert_eq!(neighbors_r3_0.len(), 5);
        assert!(neighbors_r3_0.iter().all(|p| p.label != 6)); // Point 6 should be outside
    }
}
