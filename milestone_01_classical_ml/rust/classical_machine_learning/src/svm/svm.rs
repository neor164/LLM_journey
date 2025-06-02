//! Core SVM classifier structure and kernel definitions.

use crate::common_types::DataPoint;
use rand; // For random number generation in SMO
use num_traits::{Float, Zero, One, NumCast, FromPrimitive}; // Added FromPrimitive
use std::fmt::Debug;
// num_traits::NumCast is already imported above

#[derive(Debug, Clone)]
pub enum KernelConfig<F: Float + Debug> { // Added Debug bound for F
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
#[derive(Debug)] // L removed from struct definition
pub struct SvmClassifier<F>
where
    F: Float + Zero + One + Debug + Clone + std::iter::Sum + NumCast + FromPrimitive, // Added FromPrimitive
{
   
    pub C: F,
    /// Kernel configuration. Specifies the type of kernel to be used and its parameters.
    pub kernel_config: KernelConfig<F>,
    /// Tolerance for stopping criterion.
    pub tolerance: F,
    /// Maximum number of iterations for the SMO algorithm.
    pub max_passes: usize,

    // --- Learned parameters (populated by fit method) ---
    /// Lagrange multipliers (alpha_i) for each training sample, used during fitting.
    /// This will be cleared or shrunk after fitting if we store SV-specific alphas.
    alphas_workspace: Vec<F>, // Renamed for clarity, used as workspace during fit
    /// Bias term (also known as intercept).
    b: F,
    /// Lagrange multipliers for the support vectors.
    sv_alphas: Vec<F>,
    /// Features of the support vectors.
    sv_features: Vec<Vec<F>>,
    /// Numeric labels (e.g., +1.0, -1.0) of the support vectors.
    sv_labels: Vec<F>,
}

impl<F> SvmClassifier<F> // L removed from impl block signature
where
    F: Float + Zero + One + Debug + Clone + std::iter::Sum + Default + NumCast + FromPrimitive, // Added FromPrimitive
{
    pub fn new(
        C: F,
        kernel_config: KernelConfig<F>,
        tolerance: F,
        max_passes: usize,
    ) -> Self { // L is no longer part of Self's generics here
        if C <= F::zero() {
            panic!("Regularization parameter C must be strictly positive.");
        }
        if tolerance <= F::zero() {
            panic!("Tolerance must be strictly positive.");
        }
        if max_passes == 0 {
            panic!("max_passes must be greater than 0.");
        }

        SvmClassifier { // No need for ::<F, L> turbofish
            C,
            kernel_config,
            tolerance,
            max_passes,
            alphas_workspace: Vec::new(), 
            b: F::default(),     
            // Initialize support vector storage
            sv_alphas: Vec::new(),
            sv_features: Vec::new(),
            sv_labels: Vec::new(),
        }
    }

    fn compute_kernel(&self, x1: &[F], x2: &[F]) -> F {
        if x1.len() != x2.len() {
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

    // L is now a generic parameter only on the fit method
    pub fn fit<L>(&mut self, training_data: &[DataPoint<F, L>])
    where
        L: Debug + Clone + PartialEq + Into<F> + Copy, // Bounds for L specific to fit
    {
        if training_data.is_empty() {
            // Or return an error
            panic!("Training data cannot be empty.");
        }
        let n_samples = training_data.len();
        self.alphas_workspace = vec![F::zero(); n_samples]; // Use workspace alphas
        self.b = F::zero();
        
        // Clear previous support vector information
        self.sv_alphas.clear();
        self.sv_features.clear();
        self.sv_labels.clear();

        let mut passes = 0;
        // A small constant for checking significant alpha changes, can be tuned.
        // Use num_traits::NumCast for conversion from f64 literal
        let alpha_change_epsilon = F::from_f64(1e-5).unwrap_or_else(F::epsilon); // Ensure NumCast is in scope if F::from_f64 is used

        while passes < self.max_passes {
            let mut num_changed_alphas = 0; // Made mutable
            for i in 0..n_samples {
                // f(x_i) = sum_j (alpha_j * y_j * K(x_j, x_i)) + b
                let mut f_xi = self.b;
                for j in 0..n_samples {
                    if self.alphas_workspace[j] > F::zero() { 
                        f_xi = f_xi + self.alphas_workspace[j] * training_data[j].label.into() * self.compute_kernel(&training_data[j].features, &training_data[i].features);
                    }
                }
                let y_i: F = training_data[i].label.into();
                let e_i = f_xi - y_i; // Error for point i
 
                // Check if alpha_i violates KKT conditions
                // (y_i * E_i < -tolerance && alpha_i < C) || (y_i * E_i > tolerance && alpha_i > 0)
                if (y_i * e_i < -self.tolerance && self.alphas_workspace[i] < self.C) ||
                   (y_i * e_i >  self.tolerance && self.alphas_workspace[i] > F::zero()) {
                    
                    // Select j != i randomly (simplest selection, can be improved)
                    let mut j = rand::random::<usize>() % n_samples;
                    while j == i {
                        j = rand::random::<usize>() % n_samples;
                    }

                    // --- SMO Sub-problem for (alpha_i, alpha_j) ---

                    // 1. Calculate E_j
                    let mut f_xj = self.b;
                    for k_idx in 0..n_samples {
                        if self.alphas_workspace[k_idx] > F::zero() {
                            f_xj = f_xj + self.alphas_workspace[k_idx] * training_data[k_idx].label.into() * self.compute_kernel(&training_data[k_idx].features, &training_data[j].features);
                        }
                    }
                    let y_j: F = training_data[j].label.into();
                    let e_j = f_xj - y_j; // Error for point j

                    // 2. Store old workspace alphas
                    let alpha_i_old = self.alphas_workspace[i];
                    let alpha_j_old = self.alphas_workspace[j];

                    // 3. Compute L and H (bounds for new alpha_j)
                    let L: F;
                    let H: F;
                    if y_i != y_j {
                        L = (alpha_j_old - alpha_i_old).max(F::zero());
                        H = (self.C + alpha_j_old - alpha_i_old).min(self.C);
                    } else {
                        L = (alpha_i_old + alpha_j_old - self.C).max(F::zero());
                        H = (alpha_i_old + alpha_j_old).min(self.C);
                    }

                    if L >= H { // Using >= to catch L==H, which means no room to change alpha_j
                        continue; // Skip to next i
                    }

                    // 4. Compute eta = 2*K_ij - K_ii - K_jj
                    let k_ii = self.compute_kernel(&training_data[i].features, &training_data[i].features);
                    let k_jj = self.compute_kernel(&training_data[j].features, &training_data[j].features);
                    let k_ij = self.compute_kernel(&training_data[i].features, &training_data[j].features);
                    let eta = F::from_f64(2.0).unwrap() * k_ij - k_ii - k_jj;

                    // 5. If eta >= 0 (or very close to 0), objective function is not increasing.
                    //    The original SMO paper suggests a more complex handling for eta=0.
                    //    For simplicity, if eta is not strictly negative, we skip.
                    if eta >= -F::epsilon() { // Using -epsilon for a small tolerance around zero
                        continue; // Skip to next i
                    }

                    // 6. Compute new alpha_j (unclipped)
                    let mut alpha_j_new = alpha_j_old - (y_j * (e_i - e_j)) / eta;

                    // 7. Clip alpha_j_new to [L, H]
                    if alpha_j_new > H {
                        alpha_j_new = H;
                    } else if alpha_j_new < L {
                        alpha_j_new = L;
                    }

                    // 8. If |alpha_j_new - alpha_j_old| < some_small_epsilon, skip update.
                    if (alpha_j_new - alpha_j_old).abs() < alpha_change_epsilon {
                        continue; // Skip to next i
                    }

                    // 9. Compute new alpha_i
                    let alpha_i_new = alpha_i_old + y_i * y_j * (alpha_j_old - alpha_j_new);

                    // 10. Compute b1, b2, and update self.b
                    let b1 = self.b - e_i - y_i * (alpha_i_new - alpha_i_old) * k_ii - y_j * (alpha_j_new - alpha_j_old) * k_ij;
                    let b2 = self.b - e_j - y_i * (alpha_i_new - alpha_i_old) * k_ij - y_j * (alpha_j_new - alpha_j_old) * k_jj;

                    if F::zero() < alpha_i_new && alpha_i_new < self.C {
                        self.b = b1;
                    } else if F::zero() < alpha_j_new && alpha_j_new < self.C {
                        self.b = b2;
                    } else {
                        self.b = (b1 + b2) / F::from_f64(2.0).unwrap();
                    }

                    // 11. Update workspace alphas and num_changed_alphas
                    self.alphas_workspace[i] = alpha_i_new;
                    self.alphas_workspace[j] = alpha_j_new;
                    num_changed_alphas += 1;
                }
            }

            if num_changed_alphas == 0 {
                passes += 1;
            } else {
                passes = 0; // Reset passes if alphas changed, to ensure full convergence
            }
        }
        // After loops, populate self.support_vectors based on non-zero alphas
        self.sv_alphas.clear();
        self.sv_features.clear();
        self.sv_labels.clear();

        for i in 0..n_samples {
            if self.alphas_workspace[i] > F::epsilon() { // Use a small epsilon for float comparison
                self.sv_alphas.push(self.alphas_workspace[i]);
                self.sv_features.push(training_data[i].features.clone());
                self.sv_labels.push(training_data[i].label.into());
            }
        }
        // Optional: Clear the workspace alphas if memory is a concern
        // self.alphas_workspace.clear();
        // self.alphas_workspace.shrink_to_fit();
    }

    pub fn predict_numeric_output(&self, features: &[F]) -> F {
        if self.sv_features.is_empty() { // Check if model is trained by looking at support vectors
            // Model not trained or no support vectors found.
            // Returning b, which might be 0.0 if not trained or if b is indeed 0.
            // Consider panic or Option/Result for more robust error handling.
            return self.b;
        }

        let mut decision_value = self.b;
        for i in 0..self.sv_features.len() {
            let alpha_i = self.sv_alphas[i];
            let y_i = self.sv_labels[i]; // Already numeric F
            let sv_features_i = &self.sv_features[i];
            decision_value = decision_value + alpha_i * y_i * self.compute_kernel(sv_features_i, features);
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
        let classifier = SvmClassifier::<f64>::new( // L removed from type instantiation
            1.0, // C
            default_linear_kernel_f64(),
            1e-3, // tolerance
            1000, // max_passes
        );
        assert_eq!(classifier.C, 1.0);
        assert!(matches!(classifier.kernel_config, KernelConfig::Linear));
        assert_eq!(classifier.tolerance, 1e-3);
        assert_eq!(classifier.max_passes, 1000);
        assert!(classifier.alphas_workspace.is_empty());
        assert_eq!(classifier.b, 0.0); // Default for f64
        assert!(classifier.sv_alphas.is_empty());
        assert!(classifier.sv_features.is_empty());
        assert!(classifier.sv_labels.is_empty());
    }

    #[test]
    #[should_panic(expected = "Regularization parameter C must be strictly positive.")]
    fn test_svm_new_invalid_c_zero() {
        SvmClassifier::<f64>::new(0.0, default_linear_kernel_f64(), 1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "Regularization parameter C must be strictly positive.")]
    fn test_svm_new_invalid_c_negative() {
        SvmClassifier::<f64>::new(-1.0, default_linear_kernel_f64(), 1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be strictly positive.")]
    fn test_svm_new_invalid_tolerance_zero() {
        SvmClassifier::<f64>::new(1.0, default_linear_kernel_f64(), 0.0, 1000);
    }

    #[test]
    #[should_panic(expected = "Tolerance must be strictly positive.")]
    fn test_svm_new_invalid_tolerance_negative() {
        SvmClassifier::<f64>::new(1.0, default_linear_kernel_f64(), -1e-3, 1000);
    }

    #[test]
    #[should_panic(expected = "max_passes must be greater than 0.")]
    fn test_svm_new_invalid_max_passes() {
        SvmClassifier::<f64>::new(1.0, default_linear_kernel_f64(), 1e-3, 0);
    }

    #[test]
    fn test_svm_new_rbf_kernel() {
        let rbf_kernel = KernelConfig::Rbf { gamma: 0.5 };
        let classifier = SvmClassifier::<f64>::new( // L removed
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
        let classifier = SvmClassifier::<f32>::new( // L removed
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
        let classifier = SvmClassifier::<f32>::new( // L removed
            1.0,
            KernelConfig::Linear,
            1e-3_f32, // Ensure tolerance is f32
            100
        );
        let features = vec![1.0_f32, 2.0_f32];
        // Expects b (default 0.0_f32) for a non-trained classifier
        assert_eq!(classifier.predict_numeric_output(&features), 0.0_f32); 
    }

    #[test]
    fn test_fit_and_predict_simple_linear_separable() {
        let mut classifier = SvmClassifier::<f64>::new( // L removed
            1.0, // C
            KernelConfig::Linear,
            1e-4, // tolerance
            1000, // max_passes
        );

        let training_data = vec![
            DataPoint::new(vec![1.0, 1.0], MockLabelF64(1)),  // Class +1
            DataPoint::new(vec![1.5, 1.5], MockLabelF64(1)),
            DataPoint::new(vec![2.0, 2.0], MockLabelF64(1)),
            DataPoint::new(vec![5.0, 5.0], MockLabelF64(-1)), // Class -1
            DataPoint::new(vec![5.5, 5.5], MockLabelF64(-1)),
            DataPoint::new(vec![6.0, 6.0], MockLabelF64(-1)),
        ];

        classifier.fit(&training_data);

        // Check if some support vectors were found (specific number depends on data and C)
        assert!(!classifier.sv_alphas.is_empty(), "Should have some support vector alphas");
        assert_eq!(classifier.sv_alphas.len(), classifier.sv_features.len(), "SV alphas and features count mismatch");
        assert_eq!(classifier.sv_alphas.len(), classifier.sv_labels.len(), "SV alphas and labels count mismatch");

        // Test predictions (exact decision boundary depends on SMO convergence)
        // We expect points clearly in class +1 to be predicted as +1.0
        // and points clearly in class -1 to be predicted as -1.0

        // Point clearly in class +1
        let pred_plus1 = classifier.predict_numeric_output(&[1.2, 1.2]);
        assert_eq!(pred_plus1, 1.0, "Prediction for (1.2, 1.2) should be +1");

        // Point clearly in class -1
        let pred_minus1 = classifier.predict_numeric_output(&[5.7, 5.7]);
        assert_eq!(pred_minus1, -1.0, "Prediction for (5.7, 5.7) should be -1");

        // A point near the boundary (e.g., (3.0, 3.0)) might be harder to assert without knowing exact b
        // For this simple test, we'll stick to clear cases.

        // Verify predictions on training data (should be mostly correct for separable data)
        let mut correct_train_preds = 0;
        for dp in &training_data {
            if classifier.predict_numeric_output(&dp.features) == dp.label.into() {
                correct_train_preds += 1;
            }
        }
        // For linearly separable data and sufficient passes, we expect high accuracy.
        assert!(correct_train_preds >= training_data.len() - 1, "SVM should correctly classify most/all training points for this simple case. Got {}/{}", correct_train_preds, training_data.len());
    }
}
