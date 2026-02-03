"""
Test script to verify coefficient matrix saving logic.
"""

import pandas as pd
import numpy as np

def test_coefficient_saving():
    """Test to see what's actually happening with coefficient matrix saving."""
    
    # Load the saved coefficient matrices
    res_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/NMFClustering/outputs/run9/results/"
    
    print("=== TESTING COEFFICIENT MATRIX SAVING ===\n")
    
    # Load all coefficient-related files
    files_to_check = [
        "nmf_coefficient_matrix_acoustic.csv",
        "nmf_coefficient_matrix_perceptual.csv", 
        "nmf_coefficient_matrix_normalized_acoustic.csv",
        "nmf_coefficient_matrix_normalized_perceptual.csv"
    ]
    
    for filename in files_to_check:
        filepath = f"{res_dir}/{filename}"
        try:
            df = pd.read_csv(filepath, index_col=0)
            print(f"\n--- {filename} ---")
            print(f"Shape: {df.shape}")
            print(f"Sample_15 values (first 4 components):")
            if 'Sample_15' in df.columns:
                sample_15 = df['Sample_15']
                for i, (comp, val) in enumerate(sample_15.items()):
                    print(f"  {comp}: {val:.6e}")
                    if i >= 3:  # Only show first 4 components
                        break
                print(f"  Sum: {sample_15.sum():.6f}")
                print(f"  All values < 1e-15: {(sample_15 < 1e-15).all()}")
                print(f"  Min: {sample_15.min():.6e}")
                print(f"  Max: {sample_15.max():.6e}")
            else:
                print("  Sample_15 not found in this file")
                
            print(f"Sample_16 values (first 4 components):")
            if 'Sample_16' in df.columns:
                sample_16 = df['Sample_16']
                for i, (comp, val) in enumerate(sample_16.items()):
                    print(f"  {comp}: {val:.6e}")
                    if i >= 3:  # Only show first 4 components
                        break
                print(f"  Sum: {sample_16.sum():.6f}")
                print(f"  All values < 1e-15: {(sample_16 < 1e-15).all()}")
                print(f"  Min: {sample_16.min():.6e}")
                print(f"  Max: {sample_16.max():.6e}")
            else:
                print("  Sample_16 not found in this file")
                
        except FileNotFoundError:
            print(f"\n--- {filename} ---")
            print("  File not found")
        except Exception as e:
            print(f"\n--- {filename} ---")
            print(f"  Error loading: {e}")
    
    # Now let's specifically check for problematic perceptual samples
    print(f"\n=== PERCEPTUAL SAMPLE ANALYSIS ===")
    try:
        perceptual_df = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_perceptual.csv", index_col=0)
        
        # Check samples that debug script identified as problematic
        problematic_samples = ['Sample_6', 'Sample_7', 'Sample_10', 'Sample_15', 'Sample_16']
        
        for sample in problematic_samples:
            if sample in perceptual_df.columns:
                sample_data = perceptual_df[sample]
                print(f"\n{sample}:")
                print(f"  Values: {sample_data.values}")
                print(f"  Sum: {sample_data.sum():.6e}")
                print(f"  All values < 1e-15: {(sample_data < 1e-15).all()}")
            else:
                print(f"\n{sample}: Not found in perceptual matrix")
                
    except Exception as e:
        print(f"Error analyzing perceptual samples: {e}")

if __name__ == "__main__":
    test_coefficient_saving()
