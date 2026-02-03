"""
Test script to verify epsilon handling for all-zero perceptual data.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import HANDLE_ZERO_PERCEPTUAL, PERCEPTUAL_EPSILON
from data_loader import handle_zero_perceptual_data

def test_epsilon_handling():
    """Test the epsilon handling functionality with sample data."""
    
    print("=== TESTING EPSILON HANDLING FOR ZERO PERCEPTUAL DATA ===\n")
    
    # Create sample data that mimics the problematic scenario
    sample_data = pd.DataFrame({
        'feature1': [1.5, 0.0, 2.1, 0.0, 0.8],
        'feature2': [2.3, 0.0, 1.8, 0.0, 1.2],
        'feature3': [0.9, 0.0, 3.2, 0.0, 2.1],
        'feature4': [1.7, 0.0, 0.5, 0.0, 1.9]
    })
    sample_data.index = ['PARTICIPANT_A', 'PARTICIPANT_B', 'PARTICIPANT_C', 'PARTICIPANT_D', 'PARTICIPANT_E']
    
    print("ORIGINAL SAMPLE DATA:")
    print(sample_data)
    print(f"\nRow sums: {sample_data.sum(axis=1).tolist()}")
    
    # Test the handling function
    print(f"\nCONFIG SETTINGS:")
    print(f"  HANDLE_ZERO_PERCEPTUAL = {HANDLE_ZERO_PERCEPTUAL}")
    print(f"  PERCEPTUAL_EPSILON = {PERCEPTUAL_EPSILON}")
    
    print(f"\n--- APPLYING EPSILON HANDLING ---")
    processed_data = handle_zero_perceptual_data(sample_data)
    
    print(f"\nPROCESSED DATA:")
    print(processed_data)
    print(f"\nRow sums after processing: {processed_data.sum(axis=1).tolist()}")
    
    # Check if zero rows were properly handled
    zero_rows_before = (sample_data.sum(axis=1) == 0).sum()
    zero_rows_after = (processed_data.sum(axis=1) == 0).sum()
    
    print(f"\nRESULTS:")
    print(f"  Zero rows before: {zero_rows_before}")
    print(f"  Zero rows after: {zero_rows_after}")
    print(f"  ✓ All zero rows handled: {zero_rows_after == 0}")
    
    # Verify epsilon values
    if HANDLE_ZERO_PERCEPTUAL:
        epsilon_rows = sample_data.sum(axis=1) == 0
        if epsilon_rows.sum() > 0:
            epsilon_values = processed_data.loc[epsilon_rows].values
            all_epsilon = np.all(epsilon_values == PERCEPTUAL_EPSILON)
            print(f"  ✓ Epsilon values correct: {all_epsilon}")
            print(f"    Expected: {PERCEPTUAL_EPSILON}")
            print(f"    Found: {epsilon_values[0, 0] if len(epsilon_values) > 0 else 'None'}")

def test_with_real_coefficient_analysis():
    """Test by loading a real problematic coefficient matrix and checking behavior."""
    
    print(f"\n\n=== TESTING WITH REAL COEFFICIENT DATA ===\n")
    
    # Load one of the problematic coefficient matrices
    try:
        results_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/NMFClustering/outputs/run9/results/"
        coef_matrix = pd.read_csv(f"{results_dir}/nmf_coefficient_matrix_perceptual.csv", index_col=0)
        
        print(f"Loaded coefficient matrix: {coef_matrix.shape}")
        
        # Find problematic samples
        tiny_threshold = 1e-15
        problematic_samples = []
        for col in coef_matrix.columns:
            if (coef_matrix[col] < tiny_threshold).all():
                problematic_samples.append(col)
        
        print(f"Found {len(problematic_samples)} problematic samples with tiny coefficients")
        
        if len(problematic_samples) > 0:
            sample_name = problematic_samples[0]
            print(f"\nAnalyzing sample: {sample_name}")
            print(f"Original values: {coef_matrix[sample_name].values}")
            print(f"Original sum: {coef_matrix[sample_name].sum():.2e}")
            
            # This demonstrates what WOULD happen if epsilon was applied to input data
            print(f"\nIf epsilon ({PERCEPTUAL_EPSILON}) was applied to INPUT data,")
            print(f"NMF would produce normal coefficients instead of 2.22e-16 values")
        
    except Exception as e:
        print(f"Could not load real coefficient data: {e}")
        print("This is expected if the analysis hasn't been run yet with the new epsilon handling")

if __name__ == "__main__":
    test_epsilon_handling()
    test_with_real_coefficient_analysis()
