#!/usr/bin/env python3
"""
Create minimal test data for pipeline validation

This script creates small test datasets to validate the pipeline
without requiring the full dataset.
"""

import pandas as pd
import numpy as np
import os

def create_test_acoustic_features():
    """Create a minimal test acoustic features file."""
    print("Creating test acoustic features...")
    
    # Create minimal test data
    np.random.seed(42)
    participants = [f"TEST_PARTICIPANT_{i:02d}" for i in range(1, 11)]
    
    # Create some dummy acoustic features
    feature_names = [
        'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean',
        'spectral_centroid_mean', 'spectral_rolloff_mean',
        'zero_crossing_rate_mean', 'tempo',
        'chroma_1_mean', 'chroma_2_mean',
        'rms_energy_mean'
    ]
    
    data = {'ID': participants}
    
    # Add realistic-looking acoustic features
    for feature in feature_names:
        if 'mfcc' in feature:
            data[feature] = np.random.normal(0, 1, 10)
        elif 'spectral' in feature:
            data[feature] = np.random.exponential(2, 10)
        elif 'tempo' in feature:
            data[feature] = np.random.normal(120, 20, 10)
        else:
            data[feature] = np.random.uniform(0, 1, 10)
    
    df = pd.DataFrame(data)
    
    # Create test directory
    test_dir = "AllFeats_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Save as multiple files to match expected pattern
    for i, participant in enumerate(participants):
        test_file = os.path.join(test_dir, f"{participant}_converted.csv")
        participant_data = df.iloc[[i]]
        participant_data.to_csv(test_file, index=False)
    
    print(f"Created {len(participants)} test acoustic feature files in {test_dir}/")
    return test_dir

def create_test_perceptual_data():
    """Create test perceptual data file."""
    print("Creating test perceptual data...")
    
    np.random.seed(42)
    participants = [f"TEST_PARTICIPANT_{i:02d}" for i in range(1, 11)]
    
    # Create perceptual features
    perceptual_features = [
        'Overall_Severity', 'Articulatory_Precision', 'Resonance',
        'Vocal_Quality', 'Respiratory_Support', 'Rate',
        'Rhythm', 'Stress_Patterns', 'Intelligibility'
    ]
    
    data = {'ID': participants}
    
    # Add realistic perceptual ratings (1-7 scale typically)
    for feature in perceptual_features:
        data[feature] = np.random.randint(1, 8, 10)
    
    df = pd.DataFrame(data)
    df.to_csv("feature_weights_perceptual_test.csv", index=False)
    
    print("Created feature_weights_perceptual_test.csv")
    return "feature_weights_perceptual_test.csv"

def create_test_demographics():
    """Create test demographics data."""
    print("Creating test demographics data...")
    
    np.random.seed(42)
    participants = [f"TEST_PARTICIPANT_{i:02d}" for i in range(1, 11)]
    
    data = {
        'ID': participants,
        'Age': np.random.randint(40, 80, 10),
        'Sex': np.random.choice(['M', 'F'], 10),
        'Disease_Duration_Months': np.random.randint(6, 60, 10),
        'ALSFRS_Total': np.random.randint(20, 48, 10),
        'Bulbar_Onset': np.random.choice([0, 1], 10),
        'Education_Years': np.random.randint(12, 20, 10)
    }
    
    df = pd.DataFrame(data)
    df.to_csv("Clustering_all_results_combined_test.csv", index=False)
    
    print("Created Clustering_all_results_combined_test.csv")
    return "Clustering_all_results_combined_test.csv"

def create_test_config():
    """Create a test configuration file that points to test data."""
    print("Creating test configuration...")
    
    config_content = '''"""
Test Configuration for ALS Clustering Analysis Pipeline

This configuration uses minimal test data for validation.
"""

import os

# Test data paths
ACOUSTIC_FEATURES_PATH = "AllFeats_test/*_converted.csv"
PERCEPTUAL_DATA_PATH = "feature_weights_perceptual_test.csv"
DEMOGRAPHICS_PATH = "Clustering_all_results_combined_test.csv"

# Analysis parameters (reduced for testing)
N_COMPONENTS_RANGE = [2, 3]  # Smaller range for testing
N_RUNS = 3  # Fewer runs for testing
RANDOM_STATE = 42

# Output directories
OUTPUT_DIR = "outputs_test"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Visualization settings
FIGSIZE = (10, 8)
DPI = 100  # Lower DPI for testing
'''
    
    with open("src/config_test.py", "w") as f:
        f.write(config_content)
    
    print("Created src/config_test.py")
    return "src/config_test.py"

def main():
    """Create all test files."""
    print("=" * 60)
    print("CREATING TEST DATA FOR PIPELINE VALIDATION")
    print("=" * 60)
    
    # Create test data files
    acoustic_dir = create_test_acoustic_features()
    perceptual_file = create_test_perceptual_data()
    demographics_file = create_test_demographics()
    config_file = create_test_config()
    
    print("\n✅ Test data creation complete!")
    print("\nTo test with this data:")
    print("1. Temporarily rename src/config.py to src/config_original.py")
    print("2. Rename src/config_test.py to src/config.py")
    print("3. Run: python test_pipeline.py")
    print("4. After testing, restore original config file")
    
    print(f"\nTest files created:")
    print(f"  📁 {acoustic_dir}/ (acoustic features)")
    print(f"  📄 {perceptual_file} (perceptual data)")
    print(f"  📄 {demographics_file} (demographics)")
    print(f"  📄 {config_file} (test config)")

if __name__ == "__main__":
    main()
