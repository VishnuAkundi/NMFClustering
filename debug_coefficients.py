"""
Debug script to analyze coefficient matrices and understand the normalization process.
"""

import pandas as pd
import numpy as np
from src.config import RESULTS_DIR

def analyze_coefficient_matrices():
    """Analyze both raw and normalized coefficient matrices to understand the values."""
    
    print("=== COEFFICIENT MATRIX ANALYSIS ===\n")
    
    # Load matrices
    res_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/NMFClustering/outputs/run9/results/"
    try:
        # Raw coefficient matrices (from NMF, before any normalization)
        raw_acoustic = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_acoustic.csv", index_col=0)
        raw_perceptual = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_perceptual.csv", index_col=0)
        
        print("✓ Raw coefficient matrices loaded")
        print(f"  Raw acoustic shape: {raw_acoustic.shape}")
        print(f"  Raw perceptual shape: {raw_perceptual.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load raw matrices: {e}")
        return
    
    try:
        # Normalized coefficient matrices (used for column similarity)
        norm_acoustic = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_normalized_acoustic.csv", index_col=0)
        norm_perceptual = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_normalized_perceptual.csv", index_col=0)
        
        print("✓ Normalized coefficient matrices loaded")
        print(f"  Normalized acoustic shape: {norm_acoustic.shape}")
        print(f"  Normalized perceptual shape: {norm_perceptual.shape}")
        
    except FileNotFoundError:
        print("❌ Normalized matrices not found - run analysis with column_similarity method first")
        norm_acoustic = None
        norm_perceptual = None
    
    # Analyze problematic samples
    problematic_samples = ['Sample_15', 'Sample_16']
    
    for sample in problematic_samples:
        # Check ACOUSTIC coefficients
        if sample in raw_acoustic.columns:
            print(f"\n--- ANALYSIS FOR {sample} (ACOUSTIC) ---")
            
            # Raw values
            raw_values = raw_acoustic[sample]
            print(f"Raw coefficient values:")
            for comp, val in raw_values.items():
                print(f"  {comp}: {val:.6e}")
            print(f"Raw sum: {raw_values.sum():.6f}")
            print(f"Raw max: {raw_values.max():.6f}")
            print(f"Raw min: {raw_values.min():.6e}")
            
            # Manual row normalization (what should happen)
            manual_norm = raw_values / raw_values.sum()
            print(f"\nManual row normalization:")
            for comp, val in manual_norm.items():
                print(f"  {comp}: {val:.6f}")
            print(f"Manual normalized sum: {manual_norm.sum():.6f}")
            
            # Saved normalized values (if available)
            if norm_acoustic is not None and sample in norm_acoustic.columns:
                saved_norm = norm_acoustic[sample]
                print(f"\nSaved normalized values:")
                for comp, val in saved_norm.items():
                    print(f"  {comp}: {val:.6f}")
                print(f"Saved normalized sum: {saved_norm.sum():.6f}")
                
                # Check if they match
                diff = np.abs(manual_norm - saved_norm).max()
                print(f"Max difference between manual and saved: {diff:.6e}")
        
        # Check PERCEPTUAL coefficients for the same samples
        if sample in raw_perceptual.columns:
            print(f"\n--- ANALYSIS FOR {sample} (PERCEPTUAL) ---")
            
            # Raw values
            raw_values = raw_perceptual[sample]
            print(f"Raw coefficient values:")
            for comp, val in raw_values.items():
                print(f"  {comp}: {val:.6e}")
            print(f"Raw sum: {raw_values.sum():.6f}")
            print(f"Raw max: {raw_values.max():.6f}")
            print(f"Raw min: {raw_values.min():.6e}")
            print(f"All values < 1e-15: {(raw_values < 1e-15).all()}")
            
            # Manual row normalization (what should happen)
            if raw_values.sum() > 1e-15:  # Only normalize if sum is not near zero
                manual_norm = raw_values / raw_values.sum()
                print(f"\nManual row normalization:")
                for comp, val in manual_norm.items():
                    print(f"  {comp}: {val:.6f}")
                print(f"Manual normalized sum: {manual_norm.sum():.6f}")
            else:
                print(f"\nCannot normalize - sum too close to zero!")
            
            # Saved normalized values (if available)
            if norm_perceptual is not None and sample in norm_perceptual.columns:
                saved_norm = norm_perceptual[sample]
                print(f"\nSaved normalized values:")
                for comp, val in saved_norm.items():
                    print(f"  {comp}: {val:.6f}")
                print(f"Saved normalized sum: {saved_norm.sum():.6f}")
                
                if raw_values.sum() > 1e-15:
                    # Check if they match
                    diff = np.abs(manual_norm - saved_norm).max()
                    print(f"Max difference between manual and saved: {diff:.6e}")
        else:
            print(f"\n--- {sample} (PERCEPTUAL) ---")
            print("  Sample not found in perceptual matrix")
    
    # Check for problematic samples
    print(f"\n--- PROBLEMATIC SAMPLE DETECTION ---")
    
    # Raw coefficient analysis
    raw_sums_acoustic = raw_acoustic.sum(axis=0)
    raw_sums_perceptual = raw_perceptual.sum(axis=0)
    
    print(f"Raw acoustic coefficient sums:")
    print(f"  Mean: {raw_sums_acoustic.mean():.6f}")
    print(f"  Std: {raw_sums_acoustic.std():.6f}")
    print(f"  Min: {raw_sums_acoustic.min():.6f}")
    print(f"  Max: {raw_sums_acoustic.max():.6f}")
    
    print(f"Raw perceptual coefficient sums:")
    print(f"  Mean: {raw_sums_perceptual.mean():.6f}")
    print(f"  Std: {raw_sums_perceptual.std():.6f}")
    print(f"  Min: {raw_sums_perceptual.min():.6f}")
    print(f"  Max: {raw_sums_perceptual.max():.6f}")
    
    # Find samples with very small coefficients
    acoustic_min_vals = raw_acoustic.min(axis=0)
    perceptual_min_vals = raw_perceptual.min(axis=0)
    
    tiny_threshold = 1e-15
    tiny_acoustic = acoustic_min_vals[acoustic_min_vals < tiny_threshold]
    tiny_perceptual = perceptual_min_vals[perceptual_min_vals < tiny_threshold]
    
    print(f"\nSamples with tiny coefficients (< {tiny_threshold}):")
    print(f"  Acoustic: {len(tiny_acoustic)} samples")
    if len(tiny_acoustic) > 0:
        print(f"    {list(tiny_acoustic.index)}")
    print(f"  Perceptual: {len(tiny_perceptual)} samples")
    if len(tiny_perceptual) > 0:
        print(f"    {list(tiny_perceptual.index)}")
    
    # Check for all-zeros or near-zero samples
    near_zero_threshold = 1e-10
    acoustic_near_zero = raw_sums_acoustic[raw_sums_acoustic < near_zero_threshold]
    perceptual_near_zero = raw_sums_perceptual[raw_sums_perceptual < near_zero_threshold]
    
    print(f"\nSamples with near-zero total coefficients (< {near_zero_threshold}):")
    print(f"  Acoustic: {len(acoustic_near_zero)} samples")
    if len(acoustic_near_zero) > 0:
        print(f"    {list(acoustic_near_zero.index)}")
    print(f"  Perceptual: {len(perceptual_near_zero)} samples")
    if len(perceptual_near_zero) > 0:
        print(f"    {list(perceptual_near_zero.index)}")

if __name__ == "__main__":
    analyze_coefficient_matrices()
