# /home/neor/Projects/LLM_journey/milestone_01_classical_ml/evaluate.py
import math
import time 
from sklearn.neighbors import KNeighborsClassifier # Import scikit-learn KNN
from sklearn.datasets import load_iris, make_classification # Added make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np # For synthetic dataset label generation

try:

    import classical_machine_learning_py
    from classical_machine_learning_py import KnnClassifier, KnnDistance, SearchStrategy 

    # Now we can also import the classes defined in Rust

except ImportError as e:
    print(f"Failed to import the Rust module: {e}")
    print("Make sure you have run 'maturin develop' in the Rust crate directory")
    print("and that your virtual environment is activated.")
    exit(1)

def test_euclidean_distance():
    print("\n--- Testing Euclidean Distance ---")
    vec_a: list[float] = [1.0, 2.0, 3.0]
    vec_b: list[float] = [4.0, 5.0, 6.0]
    expected_dist_ab: float = math.sqrt(27.0)

    try:
        dist = classical_machine_learning_py.euclidean_distance_py(vec_a, vec_b)
        print(f"Euclidean distance between {vec_a} and {vec_b} is: {dist}")
        assert math.isclose(dist, expected_dist_ab), \
            f"Expected distance {expected_dist_ab}, but got {dist}"
        print(f"Assertion passed: Distance is approximately {expected_dist_ab:.3f}")
    except ValueError as e:
        print(f"Error calculating distance: {e}")
    except AssertionError as e:
        print(f"AssertionError: {e}")

    vec_c: list[float] = [1.0, 2.0]
    error_correctly_raised: bool = False
    try:
        _ = classical_machine_learning_py.euclidean_distance_py(vec_a, vec_c)
    except ValueError as e:
        print(f"Correctly caught error for mismatched lengths: {e}")
        error_correctly_raised = True
    assert error_correctly_raised, "ValueError for mismatched lengths was not raised."
    print("Assertion passed: ValueError for mismatched lengths correctly raised.")

def test_knn_with_sklearn_dataset():
    print("\n--- Testing KNN Classifier with scikit-learn Iris Dataset ---")
    k = 3 # You can experiment with different k values
    py_distance_metric = KnnDistance.Euclidean # Or KnnDistance.Manhattan
    
    # Define all search strategies to test
    search_strategies_to_test = [
        SearchStrategy.BruteForce,
        SearchStrategy.KdTree,
        SearchStrategy.BallTree,
        # None # To test auto-selection if your Rust code handles it
    ]

    # Variables for final summary
    all_rust_results_summary = []
    sklearn_baseline_results = {} # Will be populated if/when sklearn baseline is run

    # 1. Generate a synthetic dataset (once)
    try:
        # Using make_classification to create a larger dataset
        # suitable for testing KdTree and BallTree strategies.
        n_samples = 2000  # Number of samples
        n_features = 30   # Number of features (dimensionality)
                          # KdTree might be stressed, BallTree should be fine.
        n_classes = 3     # Number of classes
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features - 5, # Number of informative features
            n_redundant=5,          # Number of redundant features
            n_classes=n_classes,
            random_state=42         # For reproducibility
        )
        test_size = 0.3 # Percentage of data for testing
        random_state = 42 # For reproducible split
        # Generate target names for the synthetic dataset
        target_names = [f"Class_{i}" for i in range(n_classes)]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 2. Prepare training data for Rust KNN (list of tuples or dicts) - (once)
        # We need to convert integer labels to strings for the Rust implementation
        training_data_rust = []
        for features, label_int in zip(X_train, y_train):
            label_str = target_names[label_int]
            # Using tuple format ([float, ...], 'string')
            training_data_rust.append((features.tolist(), label_str)) # .tolist() converts NumPy array row to Python list

        # Prepare test data for Rust KNN (list of lists of floats) - (once)
        test_data_rust = X_test.tolist() # Convert NumPy array to list of lists
        
        # Create a mapping from string label back to integer label - (once)
        label_map = {name: i for i, name in enumerate(target_names)}

    except ImportError:
         print("\nSkipping scikit-learn dataset generation: scikit-learn not installed or make_classification failed.")
         return # Exit if scikit-learn is not available for dataset loading
    except Exception as e:
        print(f"An error occurred during dataset preparation: {e}")
        import traceback
        traceback.print_exc()
        return


    for py_search_strategy in search_strategies_to_test:
        print(f"\n--- Testing with Search Strategy: {py_search_strategy} ---")
        try:
            # --- Rust KNN Evaluation ---
            print("\nEvaluating Rust KNN:")
            # 3. Instantiate and fit the Rust KNN classifier (Timing fit)
            classifier = KnnClassifier(
                k, 
                py_distance_metric, 
                py_search_strategy # Pass the chosen strategy as a positional argument
            )
            print(f"Rust KNN instantiated with k={k}, distance={py_distance_metric}, requested_strategy={py_search_strategy}")
            start_time = time.time()
            classifier.fit(training_data_rust)
            end_time = time.time()
            fit_time_rust = end_time - start_time
            print(f"Rust KNN Classifier fitted with {len(training_data_rust)} samples in {fit_time_rust:.6f} seconds.")
            print(f"Search strategy chosen by Rust: {classifier.search_strategy}")

            # 5. Make predictions using the Rust KNN
            start_time = time.time()
            predictions_rust_str = classifier.predict(test_data_rust)
            end_time = time.time()
            predict_time_rust = end_time - start_time
            print(f"Rust KNN Prediction on {len(test_data_rust)} samples took {predict_time_rust:.6f} seconds.")

            # 6. Convert Rust predictions (strings) back to integer labels for scikit-learn metrics
            predictions_rust_int = [label_map[label_str] for label_str in predictions_rust_str]

            # 7. Evaluate accuracy using scikit-learn for Rust predictions
            accuracy_rust = accuracy_score(y_test, predictions_rust_int)
            print(f"Rust KNN Accuracy: {accuracy_rust:.4f}")

            # Store results for the final summary table
            all_rust_results_summary.append({
                "requested_strategy": str(py_search_strategy),
                "actual_strategy": str(classifier.search_strategy),
                "fit_time": fit_time_rust,
                "predict_time": predict_time_rust,
                "accuracy": accuracy_rust
            })

            if py_search_strategy == search_strategies_to_test[0]: # Run sklearn comparison only for the first strategy run
                print("\nEvaluating scikit-learn KNN (baseline):")
                sklearn_classifier = KNeighborsClassifier(n_neighbors=k, metric='euclidean') # 'algorithm' can be 'auto', 'ball_tree', 'kd_tree', 'brute'

                start_time = time.time()
                sklearn_classifier.fit(X_train, y_train)
                end_time = time.time()
                fit_time_sklearn = end_time - start_time
                print(f"Scikit-learn KNN fitted in {fit_time_sklearn:.6f} seconds.")

                start_time = time.time()
                predictions_sklearn_int = sklearn_classifier.predict(X_test)
                end_time = time.time()
                predict_time_sklearn = end_time - start_time
                print(f"Scikit-learn KNN Prediction on {len(X_test)} samples took {predict_time_sklearn:.6f} seconds.")
                
                accuracy_sklearn = accuracy_score(y_test, predictions_sklearn_int)
                print(f"Scikit-learn KNN Accuracy: {accuracy_sklearn:.4f}")
                
                # Populate sklearn_baseline_results for use in per-strategy and final summaries
                sklearn_baseline_results['fit_time'] = fit_time_sklearn
                sklearn_baseline_results['predict_time'] = predict_time_sklearn
                sklearn_baseline_results['accuracy'] = accuracy_sklearn
            
            # --- Comparison Summary for current strategy ---
            print("\n--- Comparison Summary ---")
            print(f"Metric: {py_distance_metric}, k: {k}, Requested Strategy: {py_search_strategy}, Actual Rust Strategy: {classifier.search_strategy}")
            print(f"Dataset: Synthetic, Samples={n_samples}, Features={n_features}, Classes={n_classes}")
            print(f"Train/Test Split: Train={len(X_train)}, Test={len(X_test)}")
            print(f"{'':<15} | {'Fit Time (s)':<15} | {'Predict Time (s)':<15} | {'Accuracy':<10}")
            print("-" * 70)
            print(f"{'Rust KNN':<15} | {fit_time_rust:<15.6f} | {predict_time_rust:<15.6f} | {accuracy_rust:<10.4f}")
            if sklearn_baseline_results: # Check if baseline results are available
                 print(f"{'Sklearn KNN':<15} | {sklearn_baseline_results['fit_time']:<15.6f} | {sklearn_baseline_results['predict_time']:<15.6f} | {sklearn_baseline_results['accuracy']:<10.4f}")


            # Optional: Add an assertion for a minimum acceptable accuracy (adjust threshold for synthetic data)
            # assert accuracy_rust > 0.7, f"Rust KNN Accuracy on synthetic dataset ({accuracy_rust:.4f}) is too low!"
            # Optional: Assert that Rust accuracy is close to scikit-learn accuracy
            # assert math.isclose(accuracy_rust, accuracy_sklearn, abs_tol=0.01), "Rust KNN accuracy significantly differs from scikit-learn."

        except ImportError:
             print("\nSkipping scikit-learn comparison test: scikit-learn not installed.")
             # This specific exception might be better handled outside the loop if it's about loading iris.
             # If it's about accuracy_score, then it's fine here.

        except Exception as e:
            print(f"An error occurred during testing with strategy {py_search_strategy}: {e}")
            import traceback
            traceback.print_exc()
        print("-" * 70) # Separator for different strategy runs

def test_knn_classifier():
    print("\n--- Testing KNN Classifier ---")
    k = 3
    try:
        # Test with dict input
        classifier_dict_input = KnnClassifier(
            k, 
            KnnDistance.Euclidean, 
            SearchStrategy.BruteForce # Pass as positional argument
        )
        print(f"Classifier created with k={k}, distance=Euclidean, requested_strategy=BruteForce")

        training_data_dict = [
            {'features': [1.0, 1.0], 'label': 'A'},
            {'features': [1.0, 2.0], 'label': 'A'},
            {'features': [2.0, 1.5], 'label': 'A'},
            {'features': [5.0, 5.0], 'label': 'B'},
            {'features': [6.0, 5.0], 'label': 'B'},
            {'features': [5.5, 6.0], 'label': 'B'},
        ]
        classifier_dict_input.fit(training_data_dict)
        print("Classifier fitted with dictionary-style data.")
        print(f"Search strategy chosen: {classifier_dict_input.search_strategy}")


        # Test with tuple input
        classifier_tuple_input = KnnClassifier(
            k, 
            KnnDistance.Manhattan, 
            SearchStrategy.KdTree # Pass as positional argument
        )
        print(f"\nClassifier created with k={k}, distance=Manhattan, requested_strategy=KdTree")
        training_data_tuple = [
            ([10.0, 10.0], 'C'),
            ([10.0, 12.0], 'C'),
            ([12.0, 11.5], 'C'),
            ([15.0, 15.0], 'D'),
            ([16.0, 15.0], 'D'),
            ([15.5, 16.0], 'D'),
        ]
        # Add enough points to potentially trigger non-brute-force
        for i in range(100): # Assuming N_THRESHOLD_FOR_BRUTE_FORCE is around 100-1000
             training_data_tuple.append(([float(i), float(i)], 'X'))

        classifier_tuple_input.fit(training_data_tuple)
        print("Classifier fitted with tuple-style data.")
        print(f"Search strategy chosen: {classifier_tuple_input.search_strategy}")


        # Predictions
        test_sample_A = [1.5, 1.5]
        prediction_A = classifier_dict_input.predict_single(test_sample_A)
        print(f"Prediction for {test_sample_A}: {prediction_A}")
        assert prediction_A == 'A'

        test_sample_B = [5.1, 5.1]
        prediction_B = classifier_dict_input.predict_single(test_sample_B)
        print(f"Prediction for {test_sample_B}: {prediction_B}")
        assert prediction_B == 'B'

        print("Single predictions assertions passed.")

        # Batch prediction
        batch_samples = [
            [0.5, 0.5], # Expect A
            [6.5, 6.5]  # Expect B
        ]
        batch_predictions = classifier_dict_input.predict(batch_samples)
        print(f"Batch predictions for {batch_samples}: {batch_predictions}")
        assert batch_predictions == ['A', 'B']
        print("Batch predictions assertions passed.")


    except Exception as e:
        print(f"An error occurred during KNN classifier testing: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    test_euclidean_distance()
    test_knn_classifier()
    test_knn_with_sklearn_dataset() 

if __name__ == "__main__":
    main()
