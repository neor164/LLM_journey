//! Core SVM classifier structure and kernel definitions.

use crate::common_types::DataPoint;
use rand; // For random number generation in SMO
use num_traits::{Float, Zero, One, AsPrimitive}; 
use std::fmt::Debug;

// --- Helper for dot product, could be moved to a common math utility module ---
fn dot_product_generic<F: Float + std::iter::Sum>(a: &[F], b: &[F]) -> F {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[derive(Debug, Clone)]
pub enum KernelConfig<F: Float> {
    /// Linear kernel: `K(x, y) = x · y`. No transformation.
    Linear,
    /// Polynomial kernel: `K(x, y) = (gamma * (x · y) + coef0)^degree`.
    Polynomial {
        gamma: F,
        degree: i32, // Degree is typically an integer
        coef0: F,    // Also known as r
    },

    Rbf {
        gamma: F,
    },
    // TODO: Potentially add Sigmoid kernel later if needed
    // Sigmoid { gamma: F, coef0: F },
}

/// Support Vector Machine (SVM) classifier.
///
/// SVMs are supervised learning models that analyze data for classification and
/// regression analysis. Given a set of training examples, each marked as belonging
/// to one of two categories, an SVM training algorithm builds a model that assigns
/// new examples to one category or the other, making it a non-probabilistic binary
/// linear classifier.
#[derive(Debug)]
pub struct SvmClassifier<F, L>
where
    F: Float + Zero + One + Debug + Clone + std::iter::Sum, 
    L: Debug + Clone + PartialEq + Into<F> + Copy, 
{
   
    pub C: F,
    /// Kernel configuration. Specifies the type of kernel to be used and its parameters.
    pub kernel_config: KernelConfig<F>,
    /// Tolerance for stopping criterion.
    pub tolerance: F,
    /// Maximum number of iterations for the SMO algorithm.
    pub max_passes: usize,

    // --- Learned parameters (populated by fit method) ---
    /// Lagrange multipliers for each support vector.
    alphas: Vec<F>,
    /// Bias term (also known as intercept).
    b: F,
    /// The support vectors. These are the data points from the training set
    /// that lie closest to the decision boundary.
    support_vectors: Vec<DataPoint<F, L>>,
    // TODO: Consider storing features and labels of support vectors separately
    // if DataPoint becomes too heavy or if direct access to raw features/labels is frequent.
    // support_vector_features: Vec<Vec<F>>,
    // support_vector_labels: Vec<L>, // e.g., Vec<i8> for +1/-1
}

impl<F, L> SvmClassifier<F, L>
where
    F: Float + Zero + One + Debug + Clone + std::iter::Sum + Default, // Added Sum, Default
    L: Debug + Clone + PartialEq + Into<F> + Copy,                   // Added Into<F>, Copy
{
    pub fn new(
        C: F,
        kernel_config: KernelConfig<F>,
        tolerance: F,
        max_passes: usize,
    ) -> Self {
        if C <= F::zero() {
            panic!("Regularization parameter C must be strictly positive.");
        }
        if tolerance <= F::zero() {
            panic!("Tolerance must be strictly positive.");
        }
        if max_passes == 0 {
            panic!("max_passes must be greater than 0.");
        }

        SvmClassifier {
            C,
            kernel_config,
            tolerance,
            max_passes,
            alphas: Vec::new(), // Initialized empty, populated by fit()
            b: F::default(),       // Initialized to default (e.g., 0.0), adjusted by fit()
            support_vectors: Vec::new(), // Initialized empty, populated by fit()
        }
    }

    fn compute_kernel(&self, x1: &[F], x2: &[F]) -> F {
        if x1.len() != x2.len() {
            // This should ideally be caught earlier or handled by ensuring all features have same dim.
            panic!("Feature vectors must have the same dimensionality for kernel computation.");
        }
        match &self.kernel_config {
            KernelConfig::Linear => {
                // Dot product: sum(x1_i * x2_i)
                x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum()
            }
            KernelConfig::Polynomial { gamma, degree, coef0 } => {
                let dot_product: F = x1.iter().zip(x2.iter()).map(|(&a, &b)| a * b).sum();
                (*gamma * dot_product + *coef0).powi(*degree) 
            }
            KernelConfig::Rbf { gamma } => {
                // exp(-gamma * ||x1 - x2||^2)
                // ||x1 - x2||^2 = sum((x1_i - x2_i)^2)
                let mut norm_sq = F::zero();
                for i in 0..x1.len() {
                    let diff = x1[i] - x2[i];
                    norm_sq = norm_sq + diff * diff;
                }
                (-(*gamma) * norm_sq).exp()
            }
        }
    }

    pub fn fit(&mut self, training_data: &[DataPoint<F, L>]) {
        if training_data.is_empty() {
            // Or return an error
            panic!("Training data cannot be empty.");
        }
        let n_samples = training_data.len();
        self.alphas = vec![F::zero(); n_samples];
        self.b = F::zero();
        self.support_vectors = Vec::new(); // Clear previous support vectors

        let mut passes = 0;
        while passes < self.max_passes {
            let mut num_changed_alphas = 0;
            for i in 0..n_samples {
                // Calculate E_i = f(x_i) - y_i
                // f(x_i) = sum_j (alpha_j * y_j * K(x_j, x_i)) + b
                let mut f_xi = self.b;
                for j in 0..n_samples {
                    if self.alphas[j] > F::zero() { // Only consider non-zero alphas
                        f_xi = f_xi + self.alphas[j] * training_data[j].label.into() * self.compute_kernel(&training_data[j].features, &training_data[i].features);
                    }
                }
                let y_i: F = training_data[i].label.into();
                let e_i = f_xi - y_i;

                // Check if alpha_i violates KKT conditions
                // (y_i * E_i < -tolerance && alpha_i < C) || (y_i * E_i > tolerance && alpha_i > 0)
                if (y_i * e_i < -self.tolerance && self.alphas[i] < self.C) ||
                   (y_i * e_i >  self.tolerance && self.alphas[i] > F::zero()) {
                    
                    // Select j != i randomly (simplest selection, can be improved)
                    let mut j = rand::random::<usize>() % n_samples;
                    while j == i {
                        j = rand::random::<usize>() % n_samples;
                    }

                    // --- SMO Sub-problem for (alpha_i, alpha_j) ---
                    // This is the complex part:
                    // 1. Calculate E_j
                    // 2. Store old alphas: alpha_i_old, alpha_j_old
                    // 3. Compute L and H (bounds for new alpha_j)
                    // 4. Compute eta = 2*K_ij - K_ii - K_jj
                    // 5. If eta >= 0, continue to next pair (or handle differently)
                    // 6. Compute new alpha_j = alpha_j_old - y_j * (E_i - E_j) / eta
                    // 7. Clip alpha_j to [L, H]
                    // 8. If |alpha_j - alpha_j_old| < some_small_epsilon, continue
                    // 9. Compute new alpha_i = alpha_i_old + y_i*y_j*(alpha_j_old - alpha_j)
                    // 10. Compute b1, b2, and update self.b
                    // 11. If successfully updated, num_changed_alphas += 1

                    // Placeholder for the actual SMO update logic
                    // For now, let's just increment num_changed_alphas to simulate progress
                    // num_changed_alphas += 1; // This line would be inside the actual update logic
                }
            }

            if num_changed_alphas == 0 {
                passes += 1;
            } else {
                passes = 0; // Reset passes if alphas changed, to ensure full convergence
            }
        }
        // After loops, populate self.support_vectors based on non-zero alphas
        for i in 0..n_samples {
            if self.alphas[i] > F::epsilon() { // Use a small epsilon for float comparison
                self.support_vectors.push(training_data[i].clone());
                // Note: You'll also need to store the corresponding alpha and label for prediction.
                // A common approach is to store Vec<F> for alphas of SVs, Vec<L> for labels of SVs,
                // and Vec<Vec<F>> for features of SVs, or a Vec<SupportVectorInfo>.
                // For simplicity now, `self.alphas` stores all alphas, and `self.support_vectors` stores DataPoints.
                // This will need refinement for efficient prediction.
            }
        }
    }

    pub fn predict_numeric_output(&self, features: &[F]) -> F {
        if self.support_vectors.is_empty() && self.alphas.iter().all(|&a| a == F::zero()) {
            // Model not trained or no support vectors found.
            // Behavior here can be to panic, return a default, or Option/Result.
            // For now, returning b, which might be 0 if not trained.
            return self.b;
        }
        let mut decision_value = self.b;
        // In a more refined implementation, self.alphas would only store alphas for support vectors.
        // Here, we iterate through all original training alphas and match with stored support vectors.
        // This is inefficient and needs to be improved by storing SV-specific alphas.
        for i in 0..self.support_vectors.len() { // This loop is conceptually wrong if alphas is for all training data
                                                 // and support_vectors is a subset.
                                                 // Let's assume for now `alphas` and `support_vectors` are aligned
                                                 // and only contain SV info. This needs to be fixed in `fit`.
            // This part needs `self.alphas` to correspond to `self.support_vectors`
            // A better structure in `fit` would be to store `sv_alphas`, `sv_labels`, `sv_features`.
            // For now, this is a placeholder for the correct logic:
            // decision_value = decision_value + sv_alpha_i * sv_label_i.into() * self.compute_kernel(&sv_features_i, features);
        }

        // Placeholder: actual prediction logic needs refined storage from `fit`.
        // For now, just returning b as a stub.
        // decision_value

        // Corrected (conceptual) loop if self.alphas and self.support_vectors are aligned post-fit:
        // This assumes self.alphas was trimmed to only contain SV alphas during fit.
        for i in 0..self.support_vectors.len() {
            if i < self.alphas.len() && self.alphas[i] > F::epsilon() { // Check if alpha is non-zero
                let sv = &self.support_vectors[i];
                let sv_label_numeric: F = sv.label.into();
                decision_value = decision_value + self.alphas[i] * sv_label_numeric * self.compute_kernel(&sv.features, features);
            }
        }

        if decision_value >= F::zero() {
            F::one() // Positive class
        } else {
            -F::one() // Negative class
        }
    }
}

// --- Unit tests for SvmClassifier ---
#[cfg(test)]
mod tests {
    use super::*; // Import items from the parent module (svm.rs)

    // Helper to create a default Linear KernelConfig for f64
    fn default_linear_kernel_f64() -> KernelConfig<f64> {
        KernelConfig::Linear
    }

    // Mock label type for testing that converts to f64
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MockLabelF64(i32);
    impl Into<f64> for MockLabelF64 {
        fn into(self) -> f64 {
            self.0 as f64
        }
    }
    

    #[test]
    fn test_svm_new_valid_parameters() {
        let classifier = SvmClassifier::<f64, i32>::new(
            1.0, // C
            default_linear_kernel_f64(),
            1e-3, // tolerance
            1000, // max_passes
        );
        assert_eq!(classifier.C, 1.0);
        assert!(matches!(classifier.kernel_config, KernelConfig::Linear));
        assert_eq!(classifier.tolerance, 1e-3);
        assert_eq!(classifier.max_passes, 1000);
        assert!(classifier.alphas.is_empty());
        assert_eq!(classifier.b, 0.0); // Default for f64
        assert!(classifier.support_vectors.is_empty());
    }

    #[test]
    #[should_panic(expected = "Regularization parameter C must be strictly positive.")]
    fn test_svm_new_invalid_c_zero() {
        SvmClassifier::<f64, i32>::new(0.0, default_linear_kernel_f64(), 1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "Regularization parameter C must be strictly positive.")]
    fn test_svm_new_invalid_c_negative() {
        SvmClassifier::<f64, i32>::new(-1.0, default_linear_kernel_f64(), 1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be strictly positive.")]
    fn test_svm_new_invalid_tolerance_zero() {
        SvmClassifier::<f64, i32>::new(1.0, default_linear_kernel_f64(), 0.0, 1000);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be strictly positive.")]
    fn test_svm_new_invalid_tolerance_negative() {
        SvmClassifier::<f64, i32>::new(1.0, default_linear_kernel_f64(), -1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "max_passes must be greater than 0.")]
    fn test_svm_new_invalid_max_passes() {
        SvmClassifier::<f64, i32>::new(1.0, default_linear_kernel_f64(), 1e-3, 0);
    }

    #[test]
    fn test_svm_new_rbf_kernel() {
        let rbf_kernel = KernelConfig::Rbf { gamma: 0.5 };
        let classifier = SvmClassifier::<f64, MockLabelF64>::new( // Use MockLabelF64
            10.0,
            rbf_kernel.clone(), // Clone if you want to assert against the original
            1e-4,
            500,
        );
        assert_eq!(classifier.C, 10.0);
        if let KernelConfig::Rbf { gamma } = classifier.kernel_config {
            assert_eq!(gamma, 0.5);
        } else {
            panic!("KernelConfig should be RBF");
        }
    }

    // Mock label type for testing that converts to f32
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct MockLabelF32(i32); // Can also be bool if you prefer
    impl Into<f32> for MockLabelF32 {
        fn into(self) -> f32 {
            self.0 as f32 // Example: if it was bool: if self.0 { 1.0 } else { -1.0 }
        }
    }

    #[test]
    fn test_svm_new_polynomial_kernel() {
        let poly_kernel = KernelConfig::Polynomial {
            gamma: 0.1,
            degree: 3,
            coef0: 1.0,
        };
        let classifier = SvmClassifier::<f32, MockLabelF32>::new( // Use MockLabelF32
            0.5,
            poly_kernel.clone(),
            1e-2_f32,
            200,
        );
        assert_eq!(classifier.C, 0.5_f32);
        if let KernelConfig::Polynomial { gamma, degree, coef0 } = classifier.kernel_config {
            assert_eq!(gamma, 0.1_f32);
            assert_eq!(degree, 3);
            assert_eq!(coef0, 1.0_f32);
        } else {
            panic!("KernelConfig should be Polynomial");
        }
    }

    // Test for predict_numeric_output with a non-trained classifier
    #[test]
    fn test_predict_numeric_output_not_trained() {
        // Test with f32 and a custom label type that converts to f32
        let classifier = SvmClassifier::<f32, MockLabelF32>::new(
            1.0,
            KernelConfig::Linear,
            1e-3_f32, // Ensure tolerance is f32
            100
        );
        let features = vec![1.0_f32, 2.0_f32];
        // Expects b (default 0.0_f32) for a non-trained classifier
        assert_eq!(classifier.predict_numeric_output(&features), 0.0_f32); 
    }
}
