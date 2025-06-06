//! Decision Tree classifier and regressor implementations.

use crate::common_types::DataPoint;
use num_traits::{Float, Zero, One, NumCast, FromPrimitive};
use std::fmt::Debug;
use std::collections::HashMap; // For counting class labels or calculating variance

/// Defines the type of split condition at an internal node.
#[derive(Debug, Clone)] // Added Clone
enum SplitRule<F: Float + Debug + Clone> { 
    NumericLessThan { threshold: F },
    CategoricalInSet { values_for_left: Vec<F> },
}

/// Helper to get unique, sorted float values from an iterator of feature values.
/// Handles potential NaNs by filtering them out.
/// Sorts using partial_cmp, suitable for float types.
fn get_unique_sorted_float_values<F>(feature_values_iter: impl Iterator<Item = F>) -> Vec<F>
where
    F: Float + PartialOrd + Clone, // F must be a float type
{
    let mut values: Vec<F> = feature_values_iter
        .filter(|v| !v.is_nan()) // Filter out NaNs, as they don't provide useful split points
        .collect();

    if values.is_empty() {
        return vec![];
    }

    // Sort using partial_cmp. For floats, NaNs (if not filtered) would make this tricky.
    // Since NaNs are filtered, standard sort_unstable_by with partial_cmp is fine.
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Dedup based on partial equality.
    // This is important for floats where direct equality (==) might be problematic
    // if not for Ord, but dedup_by works with a custom comparison.
    values.dedup_by(|a, b| (*a).partial_cmp(&*b) == Some(std::cmp::Ordering::Equal));

    values
}

/// Finds a threshold based on the median of unique feature values and partitions the data.
/// Returns the threshold, and two new Vecs for left and right subsets.
/// Points < threshold go left, points >= threshold go right.
/// NaNs in features are currently sent to the right subset.
/// This is one strategy for choosing a split point, inspired by median-based partitioning.
fn find_median_based_split<F, L>(
    data: &[DataPoint<F, L>],
    feature_idx: usize,
) -> Option<(F, Vec<DataPoint<F, L>>, Vec<DataPoint<F, L>>)>
where
    F: Float + Debug + Clone + PartialOrd + NumCast + FromPrimitive,
    L: Clone + Debug,
{
    if data.is_empty() {
        return None;
    }
    if feature_idx >= data[0].features.len() {
        panic!("Feature index out of bounds in find_median_based_split: {} vs len {}", feature_idx, data[0].features.len());
    }

    let unique_sorted_values = get_unique_sorted_float_values(
        data.iter().map(|dp| dp.features[feature_idx].clone())
    );

    if unique_sorted_values.len() < 2 {
        // Need at least two unique non-NaN values to form a meaningful split threshold.
        return None;
    }


    let mid_point_idx = unique_sorted_values.len() / 2;
    // Ensure we use indices [mid_point_idx - 1] and [mid_point_idx]
    let threshold = (unique_sorted_values[mid_point_idx - 1] + unique_sorted_values[mid_point_idx]) / F::from_f64(2.0).unwrap();

    let mut left_subset: Vec<DataPoint<F, L>> = Vec::new();
    let mut right_subset: Vec<DataPoint<F, L>> = Vec::new();

    for dp in data.iter() {
        let feature_val = dp.features[feature_idx];
        if feature_val.is_nan() {
            right_subset.push(dp.clone());
            continue;
        }
        if feature_val < threshold {
            left_subset.push(dp.clone());
        } else {
            right_subset.push(dp.clone());
        }
    }

    if left_subset.is_empty() || right_subset.is_empty() {
        return None;
    }

    Some((threshold, left_subset, right_subset))
}

#[derive(Debug)]
enum Node<F, L> 
where 
    F: Float + Debug, 
    L: Debug,         
{
    Leaf { value: L }, 
    Internal {
        feature_index: usize,
        split_rule: SplitRule<F>,
        left_child: Box<Node<F, L>>,
        right_child: Box<Node<F, L>>,
    },
}

/// Criterion used for measuring the quality of a split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Criterion {
    Gini,
    Entropy,
}

#[derive(Debug)]
pub struct DecisionTreeClassifier<F, L>
where
    F: Float + Zero + One + Debug + Clone + PartialOrd + NumCast + FromPrimitive, 
    L: Debug + Clone + PartialEq + Eq + std::hash::Hash, 
{
    root: Option<Box<Node<F, L>>>,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    criterion: Criterion, 
}

impl<F, L> DecisionTreeClassifier<F, L>
where
    F: Float + Zero + One + Debug + Clone + PartialOrd + NumCast + FromPrimitive + std::iter::Sum,
    L: Debug + Clone + PartialEq + Eq + std::hash::Hash + Copy, // Added Copy for L if appropriate
{
    // Default values for hyperparameters
    const DEFAULT_MIN_SAMPLES_SPLIT: usize = 2;
    const DEFAULT_MIN_SAMPLES_LEAF: usize = 1;
    const DEFAULT_CRITERION: Criterion = Criterion::Gini;

    pub fn new(
        max_depth: Option<usize>,
        min_samples_split: Option<usize>,
        min_samples_leaf: Option<usize>,
        criterion: Option<Criterion>,
    ) -> Self {
        let actual_min_samples_split = min_samples_split.unwrap_or(Self::DEFAULT_MIN_SAMPLES_SPLIT);
        DecisionTreeClassifier {
            root: None,
            max_depth,
            min_samples_split: if actual_min_samples_split < 2 { Self::DEFAULT_MIN_SAMPLES_SPLIT } else { actual_min_samples_split },
            min_samples_leaf: min_samples_leaf.unwrap_or(Self::DEFAULT_MIN_SAMPLES_LEAF),
            criterion: criterion.unwrap_or(Self::DEFAULT_CRITERION),
        }
    }

    pub fn fit(&mut self, training_data: &[DataPoint<F, L>]) {
        // TODO: Implement the tree building logic (e.g., CART, ID3, C4.5)
        // - Calculate impurity (Gini, Entropy)
        // - Find the best split (feature and threshold)
        // - Recursively build the tree
        if training_data.is_empty() {
            self.root = None; // Ensure root is None if training data is empty
            return; // Or panic/error
        }
        self.root = self.build_tree(training_data, 0);
        // The unimplemented! was here, it should be removed once build_tree is functional.
        // If build_tree is fully implemented, the line above is sufficient.
        // For now, to make it compile if build_tree is still a stub:
        if self.root.is_none() && !training_data.is_empty() {
            unimplemented!("build_tree did not return a root node for non-empty data.");
        }
    }
    fn calculate_majority_label(&self, data: &[DataPoint<F, L>]) -> Option<L> {
        if data.is_empty() {

            return None;
        }
        let mut label_counts: HashMap<L, usize> = HashMap::new();
        for dp in data {
            *label_counts.entry(dp.label).or_insert(0) += 1;
        }

        Some(label_counts
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(label, _)| label)
            .expect("Label counts should not be empty if data is not empty."))
    }

    fn calculate_impurity(&self, data: &[DataPoint<F, L>]) -> F {
        if data.is_empty() {
            return F::zero(); // Or handle as an error/special case
        }
        match self.criterion {
            Criterion::Gini => self.calculate_gini_impurity(data),
            Criterion::Entropy => self.calculate_entropy(data),
        }
    }

    fn calculate_gini_impurity(&self, data: &[DataPoint<F, L>]) -> F {
        // Implementation:
        // 1. Count occurrences of each label in `data`.
        // 2. For each label, calculate (count / total_samples)^2.
        // 3. Gini = 1.0 - sum_of_squared_proportions.
        // Remember F is a float type.

        let mut label_counts: HashMap<L, usize> = HashMap::new();
        let total_samples_usize = data.len();

        for dp in data {
            *label_counts.entry(dp.label).or_insert(0) += 1;
        }
        let total_samples_f = match F::from_usize(total_samples_usize) {
            Some(val) => val,
            None => return F::nan(), 
        };

        if total_samples_f <= F::zero() { 
            return F::zero(); // Or F::nan()
        }

        let mut ssp = F::zero();
        for (_label, count_usize) in label_counts.iter() {
            let count_f = F::from_usize(*count_usize).unwrap_or_else(F::zero); // Should not fail if total_samples_f worked
            let p = count_f / total_samples_f;
            ssp = ssp + (p * p); 
            
        }
        F::from_usize(1).unwrap() - ssp
    }

    fn calculate_entropy(&self, data: &[DataPoint<F, L>]) -> F {
     
        if data.is_empty() { 
            return F::zero();
        }

        let mut label_counts: HashMap<L, usize> = HashMap::new();
        let total_samples_usize = data.len();

        for dp in data {
            *label_counts.entry(dp.label).or_insert(0) += 1;
        }

        let total_samples_f = match F::from_usize(total_samples_usize) {
            Some(val) => val,
            None => return F::nan(), 
        };

        if total_samples_f <= F::zero() { 
            return F::zero(); // Or F::nan()
        }

        let mut entropy = F::zero();
        for (_label, count_usize) in label_counts.iter() {
            let count_f = F::from_usize(*count_usize).unwrap_or_else(F::zero); // Should not fail if total_samples_f worked
            let p = count_f / total_samples_f;
            if p > F::zero() { // Avoid log(0)
                entropy = entropy - (p * p.log2()); 
            }
        }
        entropy
    }

    /// Checks if all features have constant values across the given data subset.
    /// Returns true if no split can be made based on feature values.
    fn no_features_to_split_on(&self, data: &[DataPoint<F, L>]) -> bool {
        if data.is_empty() || data.len() == 1 {
            return true; // Cannot split empty or single-point data based on features
        }

        let num_features = data[0].features.len();
        if num_features == 0 {
            return true; // No features at all
        }

        for feature_idx in 0..num_features {
            // Get the value of the current feature from the first data point
            let first_feature_value = data[0].features[feature_idx];
            // Check if any other data point has a different value for this feature
            if data.iter().skip(1).any(|dp| dp.features[feature_idx] != first_feature_value) {
                // Found a feature where values differ, so a split might be possible on this feature.
                return false; // At least one feature can potentially be split on
            }
        }

        // If we looped through all features and none had differing values, then no split is possible.
        true
    }

    // This is the main recursive tree-building function
    fn build_tree(&self, current_data: &[DataPoint<F, L>], current_depth: usize) -> Option<Box<Node<F, L>>> {
       let num_features = current_data[0].features.len();
        if num_features == 0 {
            return None; 
        }
        let is_max_depth_reached = self.max_depth.map_or(false, |max_d| current_depth >= max_d);

        if is_max_depth_reached ||
           self.calculate_impurity(current_data) == F::zero() ||
           current_data.len() < self.min_samples_split ||
           self.no_features_to_split_on(current_data)
        {
            return Some(Box::new(Node::Leaf { value: self.calculate_majority_label(current_data)? }))
        }
        let mut best_gain = -F::infinity(); 
        let mut best_feature_index: Option<usize> = None; 
        let mut best_split_rule: Option<SplitRule<F>> = None;
        let mut best_left_data: Option<Vec<DataPoint<F, L>>> = None;
        let mut best_right_data: Option<Vec<DataPoint<F, L>>> = None;
        let parent_impurity = self.calculate_impurity(current_data);

        for feature_idx in 0..num_features {
            if let Some((threshold, left_subset, right_subset)) = find_median_based_split(current_data, feature_idx) {
                // Check if the split meets min_samples_leaf criteria
                if left_subset.len() >= self.min_samples_leaf && right_subset.len() >= self.min_samples_leaf {
                    let left_impurity = self.calculate_impurity(&left_subset);
                    let right_impurity = self.calculate_impurity(&right_subset);
                    
                    let num_total = F::from_usize(current_data.len()).unwrap_or_else(F::one);
                    let num_left = F::from_usize(left_subset.len()).unwrap_or_else(F::zero);
                    let num_right = F::from_usize(right_subset.len()).unwrap_or_else(F::zero);

                    let weighted_child_impurity = (num_left / num_total) * left_impurity + (num_right / num_total) * right_impurity;
                    let current_gain = parent_impurity - weighted_child_impurity;

                    if current_gain > best_gain {
                        best_gain = current_gain;
                        best_feature_index = Some(feature_idx);
                        best_split_rule = Some(SplitRule::NumericLessThan { threshold });
                        best_left_data = Some(left_subset);
                        best_right_data = Some(right_subset);
                    }
                }
            }
        }


        if let (Some(split_rule_val), Some(feat_idx), Some(left_data), Some(right_data)) = 
            (best_split_rule, best_feature_index, best_left_data, best_right_data) {

            if best_gain > F::epsilon() { 
                let left_child = self.build_tree(&left_data, current_depth + 1);
                let right_child = self.build_tree(&right_data, current_depth + 1);

                if let (Some(lc), Some(rc)) = (left_child, right_child) {
                    return Some(Box::new(Node::Internal {
                        feature_index: feat_idx,
                        split_rule: split_rule_val.clone(), 
                        left_child: lc,
                        right_child: rc,
                    }));
                }
            }
        }

        Some(Box::new(Node::Leaf { value: self.calculate_majority_label(current_data)? }))

    }
    pub fn predict(&self, features: &[F]) -> Option<L> {
        self.traverse_tree(self.root.as_ref(), features)
    }

    fn traverse_tree(&self, current_node_opt: Option<&Box<Node<F, L>>>, features: &[F]) -> Option<L> {
        let current_node = current_node_opt?;

        match &**current_node { // Dereference Box<Node<F,L>> to Node<F,L>
            Node::Leaf { value } => Some(*value), // Assuming L is Copy
            Node::Internal { feature_index, split_rule, left_child, right_child } => {
                let feature_value = features.get(*feature_index)?; // Get the feature value for the current split

                match split_rule {
                    SplitRule::NumericLessThan { threshold } => {
                        if *feature_value < *threshold {
                            self.traverse_tree(Some(left_child), features)
                        } else {
                            self.traverse_tree(Some(right_child), features)
                        }
                    }
                    SplitRule::CategoricalInSet { values_for_left } => {
                        if values_for_left.contains(feature_value) { // Vec::contains uses PartialEq
                            self.traverse_tree(Some(left_child), features)
                        } else {
                            self.traverse_tree(Some(right_child), features)
                        }
                    }
                }
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dt_new() {
        // Example with f64 features and i32 labels
        // Test with explicit values
        let tree1 = DecisionTreeClassifier::<f64, i32>::new(Some(5), Some(3), Some(2), Some(Criterion::Entropy));
        assert_eq!(tree1.max_depth, Some(5));
        assert_eq!(tree1.min_samples_split, 3);
        assert_eq!(tree1.min_samples_leaf, 2);
        assert_eq!(tree1.criterion, Criterion::Entropy);
        assert!(tree1.root.is_none());

        // Test with defaults by passing None
        let tree2 = DecisionTreeClassifier::<f64, i32>::new(None, None, None, None);
        assert_eq!(tree2.max_depth, None);
        assert_eq!(tree2.min_samples_split, DecisionTreeClassifier::<f64,i32>::DEFAULT_MIN_SAMPLES_SPLIT);
        assert_eq!(tree2.min_samples_leaf, DecisionTreeClassifier::<f64,i32>::DEFAULT_MIN_SAMPLES_LEAF);
        assert_eq!(tree2.criterion, DecisionTreeClassifier::<f64,i32>::DEFAULT_CRITERION);
        assert!(tree2.root.is_none());

        // Test min_samples_split < 2 gets defaulted to DEFAULT_MIN_SAMPLES_SPLIT (which is 2)
        let tree3 = DecisionTreeClassifier::<f64, i32>::new(None, Some(1), None, None);
        assert_eq!(tree3.min_samples_split, DecisionTreeClassifier::<f64,i32>::DEFAULT_MIN_SAMPLES_SPLIT);
    }

    #[test]
    fn test_dt_predict_empty_tree() {
        // Use None to get default values for other params
        let tree = DecisionTreeClassifier::<f64, i32>::new(Some(5), None, None, None);
        assert_eq!(tree.predict(&[1.0, 2.0]), None);
    }

    #[test]
    fn test_dt_predict_single_leaf_node() {
        // Use None to get default values for other params
        let mut tree = DecisionTreeClassifier::<f64, i32>::new(Some(1), None, None, None);
        tree.root = Some(Box::new(Node::Leaf { value: 42 }));
        assert_eq!(tree.predict(&[1.0, 2.0]), Some(42));
    }

    // More complex predict tests would require a manually constructed tree
    // or a working fit method.
    #[test]
    fn test_dt_predict_simple_internal_node_numeric() {
        let mut tree = DecisionTreeClassifier::<f64, i32>::new(Some(2), None, None, None);
        tree.root = Some(Box::new(Node::Internal {
            feature_index: 0,
            split_rule: SplitRule::NumericLessThan { threshold: 5.0 },
            left_child: Box::new(Node::Leaf { value: 10 }),
            right_child: Box::new(Node::Leaf { value: 20 }),
        }));

        assert_eq!(tree.predict(&[4.0, 0.0]), Some(10)); // Goes left
        assert_eq!(tree.predict(&[6.0, 0.0]), Some(20)); // Goes right
    }
}