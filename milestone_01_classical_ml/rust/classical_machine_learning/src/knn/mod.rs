#[derive(Debug)]
pub enum KnnDistance {
    Euclidean,
    Manhattan,
    Hamming { threshold: f64 },
    Minkowski { p: u32 }, // p is the order for Minkowski distance
    Cosine,
} // Enum definition ends here

// Declare kd_tree.rs as a submodule of the `knn` module (defined by this mod.rs file)
pub mod kd_tree;
pub mod ball_tree; // Added ball_tree module
pub mod heap_utils; // Added heap_utils module

pub mod knn{
    #[derive(Debug, Clone)]
    pub struct DataPoint<F, L> { // DataPoint struct definition
        pub features: Vec<F>,
        pub label: L,}

    /// Defines the search strategy to be used by the KNN classifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum SearchStrategy {
        BruteForce,
        KdTree,
        BallTree,
    }

    use std::collections::HashMap;

    use super::KnnDistance; // Corrected import path
    use crate::knn::kd_tree::KdTree; // Use crate path for sibling module
    use std::ops::Sub;
    use num_traits::{Float, AsPrimitive};
    use super::ball_tree::BallTree; // Import BallTree

    // --- Helper functions for vector operations ---

    /// Calculates the dot product of two f64 vectors.
    fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    /// Calculates the magnitude (L2 norm) of an f64 vector.
    fn magnitude(vec: &[f64]) -> f64 {
        vec.iter().map(|x| x * x).sum::<f64>().sqrt()
    }


    /// The K-Nearest Neighbors Classifier.
    #[derive(Debug)]
    pub struct KnnClassifier<F, L> {
        k: usize,
        training_data: Vec<DataPoint<F, L>>, // Re-added to store training data
        pub distance_metric: KnnDistance, // Made public if tests need to inspect it
        pub kd_tree: Option<KdTree<F, L>>, // K-d tree for faster neighbor search
        pub ball_tree: Option<BallTree<F, L>>, // Ball tree for higher dimensions
        pub search_strategy: SearchStrategy,   // Dynamically chosen strategy
    }

    impl<F, L> KnnClassifier<F, L> {
        // Thresholds for dynamic strategy selection in fit()
        pub const N_THRESHOLD_FOR_BRUTE_FORCE: usize = 1000; // If N < this, use brute-force
        pub const K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL: usize = 20; // If k_dims < this, prefer K-d tree, else Ball tree
     }

   
    impl<F, L> KnnClassifier<F, L>
    where
        F: Copy + Sub<Output = F> + Float + AsPrimitive<f64> + std::iter::Sum, // Added std::iter::Sum
        L: Clone + Eq + std::hash::Hash, // Eq + Hash for HashMap in brute_force
    {
        pub fn new(k: usize, distance_metric: KnnDistance) -> Self {
            Self {
                k,
                training_data: Vec::new(), // Initialize training_data as empty
                distance_metric,
                kd_tree: None, // Initialize kd_tree as None
                ball_tree: None, 
                search_strategy: SearchStrategy::BruteForce, // Default to BruteForce before fit
            }    
        }

        /// Trains the classifier.
        /// This method stores the training data and dynamically chooses a search strategy:
        /// - Brute-force: If the number of training points `N` is less than `N_THRESHOLD_FOR_BRUTE_FORCE`.
        /// - K-d tree: If `N` is larger and dimensionality `k_dimensions` is less than `K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL`.
        /// - Ball tree: If `N` is larger and `k_dimensions` is high.
        ///
        /// If `strategy_override` is `Some`, that strategy will be used regardless of heuristics.
        /// Any existing training data and search structures will be replaced.
        pub fn fit(&mut self, training_data: Vec<DataPoint<F, L>>, strategy_override: Option<SearchStrategy>) {
            // Clear previous state
            self.kd_tree = None;
            self.ball_tree = None;

            self.training_data = training_data;

            if self.training_data.is_empty() {
                self.search_strategy = SearchStrategy::BruteForce; // Or a specific "NotFitted" state
                return;
            }

            let n_points = self.training_data.len();
            let k_dimensions = self.training_data.first().map_or(0, |dp| dp.features.len());

            if k_dimensions == 0 {
                self.search_strategy = SearchStrategy::BruteForce; // Cannot build trees with 0 dimensions
                return;
            }

            let mut chosen_strategy = SearchStrategy::BruteForce; // Default

            if let Some(override_val) = strategy_override {
                chosen_strategy = override_val;
            } else {
                // Heuristic selection if no override
                if n_points >= Self::N_THRESHOLD_FOR_BRUTE_FORCE {
                    if k_dimensions < Self::K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL {
                        chosen_strategy = SearchStrategy::KdTree;
                    } else {
                        chosen_strategy = SearchStrategy::BallTree;
                    }
                }
                // else it remains BruteForce
            }

            // Build the tree structure if the chosen strategy requires it
            // And update self.search_strategy based on successful tree build or fallback
            match chosen_strategy {
                SearchStrategy::BruteForce => {
                    self.search_strategy = SearchStrategy::BruteForce;
                    // No tree needed, data is already stored
                }
                SearchStrategy::KdTree => {
                    // Check if dimensionality is suitable for KD-tree
                    // The KdTree::new itself might have checks, but we ensure it's not trivially small for a tree.
                    // The N_THRESHOLD_FOR_BRUTE_FORCE check is primarily for heuristic choice,
                    // but if overridden, we should attempt to build unless k_dimensions is 0.
                    if k_dimensions > 0 {
                         self.kd_tree = KdTree::new(self.training_data.clone(), None, None, None);
                         if self.kd_tree.is_some() { // If tree construction SUCCEEDED
                             self.search_strategy = SearchStrategy::KdTree;
                         } else { // If tree construction FAILED
                             eprintln!("Warning: K-d tree construction failed or data unsuitable, falling back to brute-force.");
                             self.search_strategy = SearchStrategy::BruteForce;
                         }
                    } else {
                         eprintln!("Warning: Data has 0 dimensions. Cannot build K-d tree, falling back to brute-force.");
                         self.search_strategy = SearchStrategy::BruteForce;
                    }
                }
                SearchStrategy::BallTree => {
                    // Check if dimensionality is suitable for Ball tree (typically higher dims)
                     if k_dimensions > 0 {
                        self.ball_tree = BallTree::new_auto(self.training_data.clone(), None, None, None, None, None, None, None);
                        if self.ball_tree.is_some() { // If tree construction SUCCEEDED
                            self.search_strategy = SearchStrategy::BallTree;
                        } else { // If tree construction FAILED
                            eprintln!("Warning: Ball tree construction failed or data unsuitable, falling back to brute-force.");
                            self.search_strategy = SearchStrategy::BruteForce;
                        }
                     } else {
                         eprintln!("Warning: Data has 0 dimensions. Cannot build Ball tree, falling back to brute-force.");
                         self.search_strategy = SearchStrategy::BruteForce;
                     }
                }
            }
        }

        /// Returns the currently active search strategy.
        pub fn get_search_strategy(&self) -> SearchStrategy {
            self.search_strategy
        }
        
        /// Predicts the label for a single test sample by dispatching to the chosen search strategy.
        ///
        /// # Panics
        /// Panics if the model has not been trained (i.e., `fit` has not been called or was called with empty data).
        /// Panics if `k` is 0.
        pub fn predict_single(&self, test_sample_features: &[F]) -> L {
            if self.training_data.is_empty() {
                panic!("Cannot predict with no training data. Call fit() first.");
            }
            if self.k == 0 {
                panic!("k must be greater than 0 for prediction.");
            }
            let predictions = self.predict(&[test_sample_features.to_vec()]);
            predictions.into_iter().next().expect("Batch predict should return exactly one prediction for a single input sample") // This expect is safe if test_data is guaranteed to have 1 element
        }
        pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
            // It's crucial that your internal KNN logic ensures vectors a and b
            // have the same length before calling this.
            // If your DataPoint features can be empty, decide how to handle that.
            if a.is_empty() { // Assuming b is also empty due to prior length checks
                return 0.0; // Distance between two empty/zero vectors
            } 

            let dot = dot_product(a, b); // Assuming dot_product is available in this scope
            let mag_a = magnitude(a);   // Assuming magnitude is available in this scope
            let mag_b = magnitude(b);

            if mag_a == 0.0 || mag_b == 0.0 {
                // If both are zero vectors (dot will be 0, mag_a and mag_b will be 0)
                if dot == 0.0 { // More robust check: mag_a == 0.0 && mag_b == 0.0
                    return 0.0; // Distance between two zero vectors is 0
                }
                return 1.0; // One vector is zero, the other is not; maximal distance
            }
            
            let similarity = dot / (mag_a * mag_b);
            // Clamp similarity to [-1.0, 1.0] to avoid issues due to floating point inaccuracies
            let clamped_similarity = similarity.max(-1.0).min(1.0);
            
            1.0 - clamped_similarity 
        }
        pub fn minkowski_distance(&self, a: &[F], b: &[F], p_val: u32) -> f64 {
            let sum_of_powers: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| {
                    // Dereference x and y before subtraction
                    let diff_val = *x - *y; 
                    (diff_val.abs().powi(p_val as i32)).as_() // .as_() converts to f64
                }).sum();           
            // Calculate exponent directly using f64 for precision.
            sum_of_powers.powf(1.0 / (p_val as f64))
        }
        
        pub fn euclidean_distance(&self, a: &[F], b: &[F]) -> f64 {

            self.minkowski_distance(a, b, 2)
        }

        pub fn manhattan_distance(&self, a: &[F], b: &[F]) -> f64 {

            self.minkowski_distance(a, b, 1)
        }
        
        pub fn hamming_distance(&self, a: &[F], b: &[F], threshold: F) -> usize {
            assert_eq!(a.len(), b.len(), "Input vectors for Hamming distance must have the same length.");
            a.iter()
                .zip(b.iter())
                .filter(|pair| (*pair.0 - *pair.1).abs() > threshold)
                .count()
         }
         
        pub fn predict_brute_force(&self, test_data: &[Vec<F>]) -> Vec<L> {
            if self.training_data.is_empty() {
                if test_data.is_empty() {
                    return Vec::new();
                }
                panic!("Cannot predict with no training data.");
            }
            if self.k == 0 {
                panic!("k must be greater than 0 for prediction.");
            }

            let mut all_predictions = Vec::with_capacity(test_data.len());

            for test_sample_features in test_data.iter() {
                let mut distances_to_train: Vec<(f64, &L)> = Vec::with_capacity(self.training_data.len());

                for train_sample in self.training_data.iter() {
                    let distance = match self.distance_metric {
                        KnnDistance::Euclidean => {
                            self.euclidean_distance(test_sample_features, &train_sample.features)
                        }
                        KnnDistance::Manhattan => {
                            self.manhattan_distance(test_sample_features, &train_sample.features)
                        }
                        KnnDistance::Minkowski { p } => {
                            self.minkowski_distance(test_sample_features, &train_sample.features, p)
                        }
                        KnnDistance::Hamming { threshold } => {
                            // Convert f64 threshold from enum to type F for the hamming_distance function
                            // Use F::from (from NumCast trait) to convert f64 to F
                            let threshold_f: F = F::from(threshold)
                                .unwrap_or_else(|| panic!("Failed to convert Hamming threshold (f64) to feature type F."));
                            // Hamming distance returns usize, cast to f64 for consistent sorting with other metrics
                            self.hamming_distance(test_sample_features, &train_sample.features, threshold_f) as f64
                        }
                        KnnDistance::Cosine => {
                            // Convert features from type F to f64 for cosine_distance_rust
                            let test_sample_f64: Vec<f64> = test_sample_features.iter().map(|&x| x.as_()).collect();
                            let train_sample_f64: Vec<f64> = train_sample.features.iter().map(|&x| x.as_()).collect();
                            // Note: cosine_distance_rust is defined on KnnClassifier, so self.cosine_distance_rust
                            // but it's a static-like method not depending on self fields other than types.
                            Self::cosine_distance(&test_sample_f64, &train_sample_f64)
                        }
                    };
                    distances_to_train.push((distance, &train_sample.label));
                }

                // Sort training samples by distance (ascending)
                distances_to_train.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // Get the labels of the k-nearest neighbors
                let top_k_labels = distances_to_train.iter().take(self.k).map(|&(_, label)| label);

                // Perform majority vote
                let mut label_counts = HashMap::new();
                for label in top_k_labels {
                    *label_counts.entry(label).or_insert(0) += 1;
                }

                let predicted_label = label_counts.into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(label, _)| label.clone()) // Clone the label to own it
                    .expect("Failed to determine majority label: k might be 0 or no neighbors found (should be caught earlier).");
                
                all_predictions.push(predicted_label);
            }
            all_predictions
        }

        /// Predicts labels for a batch of test samples using the chosen search strategy.
        ///
        /// # Panics
        /// Panics if the model has not been trained.
        /// Panics if `k` is 0.
        pub fn predict(&self, test_data: &[Vec<F>]) -> Vec<L> {
            if self.training_data.is_empty() {
                if test_data.is_empty() { return Vec::new(); }
                panic!("Cannot predict with no training data. Call fit() first.");
            }
            if self.k == 0 {
                panic!("k must be greater than 0 for prediction.");
            }
            if test_data.is_empty() {
                return Vec::new();
            }

            match self.search_strategy {
                SearchStrategy::BruteForce => self.predict_brute_force(test_data),
                SearchStrategy::KdTree => self.predict_kdtree(test_data),
                SearchStrategy::BallTree => self.predict_balltree(test_data),
            }
        }

        /// Predicts labels for a batch of test samples using the K-d tree.
        /// (Internal helper, called by `predict` if K-d tree strategy is chosen)
        pub fn predict_kdtree(&self, test_data: &[Vec<F>]) -> Vec<L> {
            let kd_tree_ref = self.kd_tree.as_ref().expect("K-d tree not built. This should not happen if search_strategy is KdTree.");
            if self.k == 0 {
                panic!("k must be greater than 0 for prediction.");
            }
            if self.training_data.is_empty() { 
                panic!("Cannot predict with no training data.");
            }
            if test_data.is_empty() {
                return Vec::new();
            }

            let mut all_predictions = Vec::with_capacity(test_data.len());

            for test_sample_features in test_data.iter() {
                // find_nearest_neighbors returns Vec<DataPoint<F, L>>
                let nearest_neighbors_datapoints = kd_tree_ref.find_nearest_neighbors(test_sample_features, self.k);

                if nearest_neighbors_datapoints.is_empty() && self.k > 0 {
                    // This might happen if k is larger than the number of points in the tree,
                    // or if find_nearest_neighbors had an issue (e.g., dimensionality mismatch, though checked there).
                    // For robust KNN, if fewer than k neighbors are found, one might use all found,
                    // or panic if strict k is required. Here, we expect k neighbors if possible.
                    // Consider how to handle this: panic, return Option<L>, or use fewer neighbors.
                    // For now, maintaining panic for consistency with single-sample version's expectation.
                    panic!("Failed to find sufficient neighbors for a sample using K-d tree. Found {} but expected {}.", nearest_neighbors_datapoints.len(), self.k);
                }

                let mut label_counts = HashMap::new();
                for neighbor_dp in nearest_neighbors_datapoints.iter() {
                    *label_counts.entry(&neighbor_dp.label).or_insert(0) += 1;
                }

                let predicted_label = label_counts.into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(label, _)| label.clone())
                    .expect("Failed to determine majority label from K-d tree neighbors for a sample.");
                all_predictions.push(predicted_label);
            }
            all_predictions
        }

        /// Predicts labels for a batch of test samples using the Ball tree.
        /// (Internal helper, called by `predict` if Ball tree strategy is chosen)
        pub fn predict_balltree(&self, test_data: &[Vec<F>]) -> Vec<L> {
            let ball_tree_ref = self.ball_tree.as_ref().expect("Ball tree not built. This should not happen if search_strategy is BallTree.");
            if self.k == 0 {
                panic!("k must be greater than 0 for prediction.");
            }
            if self.training_data.is_empty() { 
                panic!("Cannot predict with no training data.");
            }
            if test_data.is_empty() {
                return Vec::new();
            }

            let mut all_predictions = Vec::with_capacity(test_data.len());

            for test_sample_features in test_data.iter() {
                let nearest_neighbors_datapoints = ball_tree_ref.find_nearest_neighbors(test_sample_features, self.k);

                if nearest_neighbors_datapoints.is_empty() && self.k > 0 {
                     panic!("Failed to find sufficient neighbors for a sample using Ball tree. Found {} but expected {}.", nearest_neighbors_datapoints.len(), self.k);
                }

                let mut label_counts = HashMap::new();
                for neighbor_dp in nearest_neighbors_datapoints.iter() {
                    *label_counts.entry(&neighbor_dp.label).or_insert(0) += 1;
                }

                let predicted_label = label_counts.into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(label, _)| label.clone())
                    .expect("Failed to determine majority label from Ball tree neighbors for a sample.");
                all_predictions.push(predicted_label);
            }
            all_predictions
        }
}


}
#[cfg(test)]
mod tests {
    use super::*;
    // Import items from the knn module to make KnnClassifier accessible.
    // KnnDistnce is in the global scope from super::*
    use super::knn::KnnClassifier;
    use super::knn::SearchStrategy; // Import SearchStrategy for tests
    use super::knn::DataPoint; // For creating test data // Corrected import path

    #[test]
    fn test_minkowski_distance() { // Renamed test function to match method
        // The specific distance metric used for classifier initialization doesn't affect
        // this test, as minkowski_distance takes 'p' directly.
        // We assume F=f64 for features and L=i32 for labels.
        // L needs Eq + Hash for brute_force, but i32 satisfies this.
        // allow any types here if not constrained by method signatures later.
        // For now, keeping f64, i32 for consistency with previous tests.
        let classifier: KnnClassifier<f64, i32> =
            KnnClassifier::new(3, KnnDistance::Euclidean);

        let vec_a = vec![1.0, 2.0, 3.0];
        let vec_b = vec![4.0, 5.0, 6.0];
        let epsilon = 1e-9; // For floating point comparisons

        // Test case 1: p = 1 (Manhattan distance)
        // Expected: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9.0
        let p1 = 1;
        let dist1 = classifier.minkowski_distance(&vec_a, &vec_b, p1); // Corrected method call
        assert!((dist1 - 9.0).abs() < epsilon, "Minkowski p=1 failed. Expected 9.0, got {}", dist1);

        // Test case 2: p = 2 (Euclidean distance)
        // Expected: sqrt((1-4)^2 + (2-5)^2 + (3-6)^2) = sqrt(9 + 9 + 9) = sqrt(27)
        let p2 = 2;
        let dist2 = classifier.minkowski_distance(&vec_a, &vec_b, p2); // Corrected method call
        let expected_dist2 = (27.0_f64).sqrt();
        assert!((dist2 - expected_dist2).abs() < epsilon, "Minkowski p=2 failed. Expected {}, got {}", expected_dist2, dist2);

        // Test case 3: p = 3
        // Expected: cbrt(|1-4|^3 + |2-5|^3 + |3-6|^3) = cbrt(3^3 + 3^3 + 3^3) = cbrt(27 + 27 + 27) = cbrt(81)
        let p3 = 3;
        let dist3 = classifier.minkowski_distance(&vec_a, &vec_b, p3); // Corrected method call
        let expected_dist3 = (81.0_f64).cbrt(); // or .powf(1.0/3.0)
        assert!((dist3 - expected_dist3).abs() < epsilon, "Minkowski p=3 failed. Expected {}, got {}", expected_dist3, dist3);

        // Test case 4: Empty vectors (assuming 0.0 distance)
        let empty_vec: Vec<f64> = vec![];
        let dist_empty = classifier.minkowski_distance(&empty_vec, &empty_vec, p2); // Corrected method call
        assert!((dist_empty - 0.0).abs() < epsilon, "Minkowski with empty vectors failed. Expected 0.0, got {}", dist_empty);
    }
    #[test]
    fn test_predict_kdtree_simple_case() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        
        let training_data = vec![
            DataPoint { features: vec![1.0, 1.0], label: "A" },
            DataPoint { features: vec![1.0, 2.0], label: "A" },
            DataPoint { features: vec![2.0, 1.0], label: "A" },
            DataPoint { features: vec![5.0, 5.0], label: "B" },
            DataPoint { features: vec![5.0, 6.0], label: "B" },
            DataPoint { features: vec![6.0, 5.0], label: "B" },
        ];
        classifier.fit(training_data, Some(SearchStrategy::KdTree)); // Explicitly request KdTree

        assert_eq!(
            classifier.get_search_strategy(),
            SearchStrategy::KdTree,
            "Strategy should be KdTree when overridden and data is suitable."
        );
        assert!(classifier.kd_tree.is_some(), "K-d tree should be built when overridden and data is suitable.");
        
        let test_sample_close_to_a = vec![1.5, 1.5];
        let predictions_a = classifier.predict(&[test_sample_close_to_a]);
        assert_eq!(predictions_a.len(), 1);
        assert_eq!(predictions_a[0], "A");

        let test_sample_close_to_b = vec![5.5, 5.5];
        let predictions_b = classifier.predict(&[test_sample_close_to_b]);
        assert_eq!(predictions_b.len(), 1);
        assert_eq!(predictions_b[0], "B");
    }

     #[test]
    fn test_predict_kdtree_k1() {
        let mut classifier: KnnClassifier<f64, i32> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let training_data = vec![
            DataPoint { features: vec![0.0, 0.0], label: 10 },
            DataPoint { features: vec![10.0, 10.0], label: 20 },
        ];
        classifier.fit(training_data, Some(SearchStrategy::KdTree)); // Explicitly request KdTree

        let predictions1 = classifier.predict(&[vec![1.0, 1.0]]);
        assert_eq!(predictions1.len(), 1);
        assert_eq!(predictions1[0], 10);

        let predictions2 = classifier.predict(&[vec![9.0, 9.0]]);
        assert_eq!(predictions2.len(), 1);
        assert_eq!(predictions2[0], 20);
    }

    #[test]
    fn test_predict_kdtree_multiple_samples() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let training_data = vec![
            DataPoint { features: vec![0.0], label: "X" },
            DataPoint { features: vec![10.0], label: "Y" },
        ];
        classifier.fit(training_data, Some(SearchStrategy::KdTree)); // Explicitly request KdTree

        let test_samples = vec![vec![1.0], vec![9.0], vec![-1.0]];
        let predictions = classifier.predict(&test_samples);
        assert_eq!(predictions.len(), 3);
        assert_eq!(predictions[0], "X");
        assert_eq!(predictions[1], "Y");
        assert_eq!(predictions[2], "X");
    }

    #[test]
    fn test_fit_chooses_brute_force_for_small_n() {
        let mut classifier: KnnClassifier<f64, i32> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let num_points = KnnClassifier::<f64, i32>::N_THRESHOLD_FOR_BRUTE_FORCE / 2;
        let training_data: Vec<_> = (0..num_points).map(|i| DataPoint{features: vec![i as f64], label: i as i32}).collect();
        classifier.fit(training_data, None); // Allow heuristics
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BruteForce);
    }

    #[test]
    fn test_fit_chooses_kd_tree_for_large_n_low_dim() {
        let mut classifier: KnnClassifier<f64, i32> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let num_points = KnnClassifier::<f64, i32>::N_THRESHOLD_FOR_BRUTE_FORCE * 2;
        let num_dims = KnnClassifier::<f64, i32>::K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL / 2;
        let training_data: Vec<_> = (0..num_points)
            .map(|i| DataPoint{features: vec![i as f64; num_dims], label: i as i32})
            .collect();
        classifier.fit(training_data, None); // Allow heuristics
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::KdTree);
        assert!(classifier.kd_tree.is_some());
        assert!(classifier.ball_tree.is_none());
    }

    #[test]
    fn test_fit_chooses_ball_tree_for_large_n_high_dim() {
        let mut classifier: KnnClassifier<f64, i32> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let num_points = KnnClassifier::<f64, i32>::N_THRESHOLD_FOR_BRUTE_FORCE * 2;
        let num_dims = KnnClassifier::<f64, i32>::K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL * 2;
         let training_data: Vec<_> = (0..num_points)
            .map(|i| DataPoint{features: vec![i as f64; num_dims], label: i as i32})
            .collect();
        classifier.fit(training_data, None); // Allow heuristics
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BallTree);
        assert!(classifier.ball_tree.is_some());
        assert!(classifier.kd_tree.is_none());
    }

    #[test]
    fn test_predict_single_wrapper() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(1, KnnDistance::Euclidean);
        let training_data = vec![
            DataPoint { features: vec![0.0], label: "X" },
            DataPoint { features: vec![10.0], label: "Y" },
        ];
        classifier.fit(training_data, None); // Heuristics will likely pick BruteForce for this small data
        let prediction = classifier.predict_single(&[1.0]);
        assert_eq!(prediction, "X");
    }

    #[test]
    fn test_override_strategy_to_kdtree_succeeds() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        let training_data = vec![
            DataPoint { features: vec![1.0, 1.0], label: "A" },
            DataPoint { features: vec![1.0, 2.0], label: "A" },
            DataPoint { features: vec![5.0, 5.0], label: "B" },
        ];
        classifier.fit(training_data.clone(), Some(SearchStrategy::KdTree));
        
        // Assuming KdTree::new succeeds for this data
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::KdTree, "Strategy should be KdTree when overridden and build succeeds.");
        assert!(classifier.kd_tree.is_some(), "K-d tree should be built.");
        assert!(classifier.ball_tree.is_none(), "Ball tree should not be built.");
    }

    #[test]
    fn test_override_strategy_to_balltree_succeeds() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        let training_data = vec![
            DataPoint { features: vec![1.0, 1.0, 1.0], label: "A" }, // 3D data
            DataPoint { features: vec![1.0, 2.0, 1.0], label: "A" },
            DataPoint { features: vec![5.0, 5.0, 5.0], label: "B" },
        ];
        classifier.fit(training_data.clone(), Some(SearchStrategy::BallTree));

        // Assuming BallTree::new_auto succeeds for this data
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BallTree, "Strategy should be BallTree when overridden and build succeeds.");
        assert!(classifier.ball_tree.is_some(), "Ball tree should be built.");
        assert!(classifier.kd_tree.is_none(), "K-d tree should not be built.");
    }

    #[test]
    fn test_override_strategy_to_kdtree_falls_back_on_zero_dim() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        let training_data_zero_dim = vec![
            DataPoint { features: vec![], label: "A" },
            DataPoint { features: vec![], label: "B" },
        ];
        classifier.fit(training_data_zero_dim.clone(), Some(SearchStrategy::KdTree));
        
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BruteForce, "Strategy should fall back to BruteForce for 0-dim data even if KdTree is overridden.");
        assert!(classifier.kd_tree.is_none(), "K-d tree should not be built for 0-dim data.");
    }

    #[test]
    fn test_override_strategy_to_balltree_falls_back_on_zero_dim() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        let training_data_zero_dim = vec![
            DataPoint { features: vec![], label: "A" },
            DataPoint { features: vec![], label: "B" },
        ];
        classifier.fit(training_data_zero_dim.clone(), Some(SearchStrategy::BallTree));
        
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BruteForce, "Strategy should fall back to BruteForce for 0-dim data even if BallTree is overridden.");
        assert!(classifier.ball_tree.is_none(), "Ball tree should not be built for 0-dim data.");
    }

    #[test]
    fn test_override_strategy_to_bruteforce() {
        let mut classifier: KnnClassifier<f64, &str> = KnnClassifier::new(3, KnnDistance::Euclidean);
        // Data that would heuristically choose a tree if not overridden
        let num_points = KnnClassifier::<f64, &str>::N_THRESHOLD_FOR_BRUTE_FORCE * 2;
        let num_dims = KnnClassifier::<f64, &str>::K_DIMENSIONS_THRESHOLD_FOR_KD_VS_BALL / 2;
        let training_data: Vec<_> = (0..num_points)
            .map(|i| DataPoint{features: vec![i as f64; num_dims], label: "L"})
            .collect();

        classifier.fit(training_data.clone(), Some(SearchStrategy::BruteForce));
        assert_eq!(classifier.get_search_strategy(), SearchStrategy::BruteForce, "Strategy should be BruteForce when overridden.");
        assert!(classifier.kd_tree.is_none(), "K-d tree should not be built when BruteForce is overridden.");
        assert!(classifier.ball_tree.is_none(), "Ball tree should not be built when BruteForce is overridden.");
    }
}
