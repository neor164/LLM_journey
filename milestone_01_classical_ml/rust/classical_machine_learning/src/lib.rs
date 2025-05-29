
// Declare your algorithm modules
pub mod knn;
// pub mod svm; // Commented out unused modules
// pub mod decision_tree;
// pub mod random_forest;
// pub mod gradient_boosting;

use knn::knn::{DataPoint, KnnClassifier, SearchStrategy as RustSearchStrategy};
use knn::KnnDistance;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// --- Helper functions for vector operations ---

/// Calculates the dot product of two vectors.
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Calculates the magnitude (L2 norm) of a vector.
fn magnitude(vec: &[f64]) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Calculates the Cosine distance between two vectors of f64.
/// Cosine Distance = 1 - Cosine Similarity
#[pyfunction]
fn cosine_distance_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() { // Unsafe block removed
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Input vectors must have the same length.",
            
        ));
    }
    if a.is_empty() { // Handles both a and b being empty if lengths must match
        return Ok(0.0); // Or PyValueError if distance between empty vectors is undefined for your use case
    }

    let dot = dot_product(&a, &b);
    let mag_a = magnitude(&a);
    let mag_b = magnitude(&b);

    if mag_a == 0.0 || mag_b == 0.0 {
        return Ok(1.0); // Conventionally, if one vector is zero, distance is 1 (unless both are zero)
    }
    Ok(1.0 - (dot / (mag_a * mag_b)))
}

/// Calculates the Euclidean distance between two vectors of f64.
#[pyfunction]
fn euclidean_distance_py(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() { // Unsafe block removed
        // pyo3::exceptions::PyValueError is a good choice for this type of error
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            
            "Input vectors must have the same length.",
        ));
    }
    let sum_sq_diff: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    Ok(sum_sq_diff.sqrt())
    
}

/// Python-friendly representation of KnnDistance
#[pyclass(name = "KnnDistance")]
#[derive(Clone)]
enum PyKnnDistance {
    Euclidean,
    Manhattan,
    Cosine,
    // Minkowski, // Add if you want to expose Minkowski with p-value from Python
}

impl From<PyKnnDistance> for KnnDistance {
    fn from(val: PyKnnDistance) -> Self {
        match val {
            PyKnnDistance::Euclidean => KnnDistance::Euclidean,
            PyKnnDistance::Manhattan => KnnDistance::Manhattan,
            PyKnnDistance::Cosine => KnnDistance::Cosine,
        }
    }
}

/// Python-friendly representation of SearchStrategy
#[pyclass(name = "SearchStrategy")]
#[derive(Clone, Debug)]
enum PySearchStrategy {
    BruteForce,
    KdTree,
    BallTree,
}

impl From<RustSearchStrategy> for PySearchStrategy {
    fn from(val: RustSearchStrategy) -> Self {
        match val {
            RustSearchStrategy::BruteForce => PySearchStrategy::BruteForce,
            RustSearchStrategy::KdTree => PySearchStrategy::KdTree,
            RustSearchStrategy::BallTree => PySearchStrategy::BallTree,
        }
    }
}

impl From<PySearchStrategy> for RustSearchStrategy {
    fn from(val: PySearchStrategy) -> Self {
        match val {
            PySearchStrategy::BruteForce => RustSearchStrategy::BruteForce,
            PySearchStrategy::KdTree => RustSearchStrategy::KdTree,
            PySearchStrategy::BallTree => RustSearchStrategy::BallTree,
        }
    }
}


#[pyclass(name = "KnnClassifier")]
struct PyKnnClassifier {
    classifier: KnnClassifier<f64, String>,
    search_strategy_override: Option<RustSearchStrategy>, // Stores the user's choice from Python
}

#[pymethods]
impl PyKnnClassifier {
    #[new]
    #[pyo3(signature = (k, distance_metric, search_strategy_override = None))]
    fn new(k: usize, distance_metric: PyKnnDistance, search_strategy_override: Option<PySearchStrategy>) -> Self {
        let rust_search_strategy_override = search_strategy_override.map(|s| s.into());
        PyKnnClassifier {
            classifier: KnnClassifier::new(k, distance_metric.into()), // Call the 2-argument Rust constructor
            search_strategy_override: rust_search_strategy_override,    // Store the Python user's choice
        }
    }

    fn fit(&mut self, training_data_py: &Bound<'_, PyList>) -> PyResult<()> {
        let mut training_data_rust: Vec<DataPoint<f64, String>> = Vec::new();

        // The search_strategy_override (if any) chosen during PyKnnClassifier instantiation
        // will be passed to the Rust KnnClassifier's fit method.
        for item_py in training_data_py {
            // Use `let` binding to extend the lifetime of the temporary `PyAny` result
            // Expecting each item to be a dictionary like {'features': [1.0, 2.0], 'label': 'A'}
            // or a tuple like ([1.0, 2.0], 'A')
            if let Ok(dict) = item_py.downcast::<PyDict>() {
                let features_item_any = dict.get_item("features")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'features' key"))?
                    ; // Semicolon to end the statement for features_item_any
                let features_py = features_item_any.downcast::<PyList>()?;
                
                let label_item_any = dict.get_item("label")? // Use ? for PyResult propagation
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'label' key"))?
                    ; // Semicolon
                let label_py = label_item_any.extract::<String>()?;
                
                let features_rust: Vec<f64> = features_py.extract()?;
                training_data_rust.push(DataPoint { features: features_rust, label: label_py });

            } else if let Ok(tuple) = item_py.extract::<(Vec<f64>, String)>() {
                 training_data_rust.push(DataPoint { features: tuple.0, label: tuple.1 });
            }
             else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Training data items must be dictionaries {'features': [...], 'label': '...'} or tuples ([...], '...')",
                ));
            }
        }

        // Call the Rust KnnClassifier's fit method, passing the training data
        // and the stored search strategy override.
        self.classifier.fit(training_data_rust, self.search_strategy_override.clone());
        Ok(())
    }

    fn predict_single(&self, test_sample_features: Vec<f64>) -> PyResult<String> {
        if test_sample_features.is_empty() {
             return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Features cannot be empty for prediction."));
        }
        // Note: The Rust predict_single panics if not fit or k=0.
        // Consider adding checks here or ensuring fit() is called.
        Ok(self.classifier.predict_single(&test_sample_features))
    }
    fn predict(&self, test_data: Vec<Vec<f64>>) -> PyResult<Vec<String>> {
        if test_data.is_empty() {
            return Ok(Vec::new());
        }
        // Similar panic considerations as predict_single
        Ok(self.classifier.predict(&test_data))
    }
    #[getter]
    fn search_strategy(&self) -> PyResult<PySearchStrategy> {
        Ok(self.classifier.get_search_strategy().into())
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` in `Cargo.toml` (with `_py` suffix removed if you prefer,
/// but maturin handles it well).
#[pymodule]
fn classical_machine_learning_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> { // Use Bound<'_, PyModule>
    m.add_function(wrap_pyfunction!(euclidean_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_distance_py, m)?)?;
    m.add_class::<PyKnnDistance>()?;
    m.add_class::<PySearchStrategy>()?;
    m.add_class::<PyKnnClassifier>()?;
    Ok(())
}