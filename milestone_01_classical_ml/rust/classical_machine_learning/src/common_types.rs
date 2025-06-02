//! This module contains common data structures used across various machine learning algorithms.

/// Represents a single data point, with features and a label.
///
/// - `F`: The type of the features (e.g., `f64`, `i32`).
/// - `L`: The type of the label (e.g., `i32`, `String`, an enum).
#[derive(Debug, Clone)]
pub struct DataPoint<F, L> {
    pub features: Vec<F>,
    pub label: L,
}

// Optional: A constructor for convenience, though direct struct initialization also works.
impl<F, L> DataPoint<F, L> {
    pub fn new(features: Vec<F>, label: L) -> Self {
        DataPoint { features, label }
    }
}