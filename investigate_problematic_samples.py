"""
Script to investigate problematic perceptual samples and suggest solutions.
"""

import pandas as pd
import numpy as np

def investigate_problematic_samples():
    """Investigate samples with problematic perceptual coefficients."""
    
    print("=== INVESTIGATING PROBLEMATIC PERCEPTUAL SAMPLES ===\n")
    
    # Load the data
    res_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/NMFClustering/outputs/run9/results/"
    
    try:
        # Load raw coefficient matrices
        raw_perceptual = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_perceptual.csv", index_col=0)
        print("✓ Raw perceptual coefficient matrix loaded")
        
        # Load the original data if available
        try:
            # Try to load the combined data
            combined_data_path = f"{res_dir}/../all_data_combined.csv"  # This might exist from your pipeline
            combined_data = pd.read_csv(combined_data_path, index_col=0)
            print("✓ Combined data loaded")
        except:
            print("⚠ Combined data not found - will work with coefficient matrices only")
            combined_data = None
            
    except FileNotFoundError as e:
        print(f"❌ Could not load matrices: {e}")
        return
    
    # Identify problematic samples
    perceptual_sums = raw_perceptual.sum(axis=0)
    near_zero_threshold = 1e-10
    problematic_samples = perceptual_sums[perceptual_sums < near_zero_threshold].index
    
    print(f"Found {len(problematic_samples)} problematic samples:")
    print(f"  {list(problematic_samples)}")
    
    # Analyze each problematic sample
    print(f"\n--- DETAILED ANALYSIS OF PROBLEMATIC SAMPLES ---")
    
    for i, sample in enumerate(problematic_samples[:5]):  # Show first 5 for brevity
        print(f"\n{sample}:")
        sample_coeffs = raw_perceptual[sample]
        print(f"  Coefficient values: {sample_coeffs.values}")
        print(f"  Sum: {sample_coeffs.sum():.2e}")
        print(f"  Max: {sample_coeffs.max():.2e}")
        print(f"  All values < 1e-15: {(sample_coeffs < 1e-15).all()}")
        
        # If we have the original data, show the raw feature values
        if combined_data is not None:
            try:
                sample_idx = int(sample.replace('Sample_', ''))
                if sample_idx < len(combined_data):
                    sample_data = combined_data.iloc[sample_idx]
                    # Assuming perceptual features come first (adjust as needed)
                    perceptual_features = sample_data.iloc[:10]  # Adjust this range
                    print(f"  Raw perceptual features (first 10): {perceptual_features.values}")
                    print(f"  Feature sum: {perceptual_features.sum():.6f}")
                    print(f"  Feature mean: {perceptual_features.mean():.6f}")
                    print(f"  Zero features: {(perceptual_features == 0).sum()}")
            except:
                pass
    
    if len(problematic_samples) > 5:
        print(f"\n... and {len(problematic_samples) - 5} more problematic samples")
    
    # Summary statistics
    print(f"\n--- SUMMARY STATISTICS ---")
    print(f"Total samples: {len(perceptual_sums)}")
    print(f"Problematic samples: {len(problematic_samples)} ({len(problematic_samples)/len(perceptual_sums)*100:.1f}%)")
    
    # Distribution of coefficient sums
    print(f"\nPerceptual coefficient sum distribution:")
    print(f"  Min: {perceptual_sums.min():.6e}")
    print(f"  25th percentile: {perceptual_sums.quantile(0.25):.6f}")
    print(f"  Median: {perceptual_sums.median():.6f}")
    print(f"  75th percentile: {perceptual_sums.quantile(0.75):.6f}")
    print(f"  Max: {perceptual_sums.max():.6f}")
    print(f"  Mean: {perceptual_sums.mean():.6f}")
    print(f"  Std: {perceptual_sums.std():.6f}")
    
    # Recommendations
    print(f"\n--- RECOMMENDATIONS ---")
    print(f"1. FILTER PROBLEMATIC SAMPLES:")
    print(f"   - Remove {len(problematic_samples)} samples with near-zero coefficients")
    print(f"   - This will improve numerical stability of the analysis")
    
    print(f"\n2. CHECK DATA PREPROCESSING:")
    print(f"   - Verify MinMax scaling is working correctly for perceptual data")
    print(f"   - Consider different preprocessing (StandardScaler, RobustScaler)")
    print(f"   - Check for missing values or extreme outliers")
    
    print(f"\n3. NMF PARAMETER TUNING:")
    print(f"   - Try different initialization methods (nndsvd, nndsvda, nndsvdar)")
    print(f"   - Increase max_iter if convergence is an issue")
    print(f"   - Try different number of components")
    
    print(f"\n4. ALTERNATIVE APPROACHES:")
    print(f"   - Consider using only samples with good coefficients for both modalities")
    print(f"   - Use robust NMF variants that handle outliers better")
    
    # Generate a filtered sample list
    good_samples = perceptual_sums[perceptual_sums >= near_zero_threshold].index
    print(f"\nSUGGESTED SAMPLE FILTER:")
    print(f"  Keep {len(good_samples)} samples out of {len(perceptual_sums)}")
    print(f"  Filter criterion: perceptual coefficient sum >= {near_zero_threshold}")
    
    # Save filtered sample list
    filtered_samples_path = f"{res_dir}/filtered_sample_list.txt"
    with open(filtered_samples_path, 'w') as f:
        f.write("# Samples to KEEP (good perceptual coefficients)\n")
        for sample in good_samples:
            f.write(f"{sample}\n")
        f.write(f"\n# Samples to EXCLUDE (problematic perceptual coefficients)\n")
        for sample in problematic_samples:
            f.write(f"# {sample}\n")
    
    print(f"  Saved filtered sample list to: {filtered_samples_path}")

if __name__ == "__main__":
    investigate_problematic_samples()
