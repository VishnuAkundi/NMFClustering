"""
Script to identify all perceptual samples with 2.22E-16 coefficients and map to participant IDs.
"""

import pandas as pd
import numpy as np

def find_problematic_perceptual_samples():
    """Find all samples with all-tiny coefficient values and map to participant IDs."""
    
    print("=== FINDING ALL PROBLEMATIC PERCEPTUAL SAMPLES ===\n")
    
    # Load the data
    res_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/NMFClustering/outputs/run9/results/"
    
    try:
        # Load raw perceptual coefficient matrix
        raw_perceptual = pd.read_csv(f"{res_dir}/nmf_coefficient_matrix_perceptual.csv", index_col=0)
        print("✓ Raw perceptual coefficient matrix loaded")
        print(f"  Shape: {raw_perceptual.shape}")
        print(f"  Components: {list(raw_perceptual.index)}")
        print(f"  Samples: {raw_perceptual.shape[1]} total samples")
        
    except FileNotFoundError as e:
        print(f"❌ Could not load perceptual matrix: {e}")
        return
    
    # Try to load original data to get participant mapping
    try:
        # Look for cluster assignment files that have participant IDs
        cluster_file_path = f"{res_dir}/../tables/cluster_assignments_acoustic.csv"
        cluster_data = pd.read_csv(cluster_file_path)
        participant_ids = cluster_data['Participant_ID'].tolist()
        print(f"✓ Participant IDs loaded from cluster assignments - {len(participant_ids)} participants")
        has_participant_mapping = True
    except:
        print("⚠ Cluster assignment data not found - will only show sample indices")
        has_participant_mapping = False
        participant_ids = None
    
    print(f"\n--- ANALYZING ALL PERCEPTUAL SAMPLES ---")
    
    # Define thresholds
    tiny_threshold = 1e-15  # Values essentially zero
    near_zero_sum_threshold = 1e-10  # Sum essentially zero
    
    # Storage for results
    all_tiny_samples = []
    near_zero_sum_samples = []
    sample_analysis = []
    
    # Check each sample
    for sample_col in raw_perceptual.columns:
        sample_coeffs = raw_perceptual[sample_col]
        
        # Calculate metrics
        sample_sum = sample_coeffs.sum()
        sample_max = sample_coeffs.max()
        sample_min = sample_coeffs.min()
        all_tiny = (sample_coeffs < tiny_threshold).all()
        
        # Store analysis
        analysis = {
            'sample': sample_col,
            'sum': sample_sum,
            'max': sample_max,
            'min': sample_min,
            'all_tiny': all_tiny,
            'near_zero_sum': sample_sum < near_zero_sum_threshold,
            'coefficients': sample_coeffs.values.tolist()
        }
        
        # Get participant ID if available
        if has_participant_mapping:
            try:
                sample_idx = int(sample_col.replace('Sample_', ''))
                if sample_idx < len(participant_ids):
                    participant_id = participant_ids[sample_idx]
                    analysis['participant_id'] = participant_id
                else:
                    analysis['participant_id'] = f"INDEX_OUT_OF_RANGE_{sample_idx}"
            except:
                analysis['participant_id'] = f"PARSE_ERROR_{sample_col}"
        else:
            analysis['participant_id'] = "UNKNOWN"
        
        sample_analysis.append(analysis)
        
        # Categorize problematic samples
        if all_tiny:
            all_tiny_samples.append(analysis)
        if sample_sum < near_zero_sum_threshold:
            near_zero_sum_samples.append(analysis)
    
    # Report findings
    print(f"\n--- SUMMARY RESULTS ---")
    print(f"Total samples analyzed: {len(sample_analysis)}")
    print(f"Samples with ALL coefficients < {tiny_threshold}: {len(all_tiny_samples)}")
    print(f"Samples with sum < {near_zero_sum_threshold}: {len(near_zero_sum_samples)}")
    
    # Detailed reporting for all-tiny samples
    if len(all_tiny_samples) > 0:
        print(f"\n--- SAMPLES WITH ALL COEFFICIENTS ≈ 2.22E-16 ---")
        print(f"Found {len(all_tiny_samples)} samples where ALL coefficients are essentially zero:\n")
        
        for i, sample in enumerate(all_tiny_samples):
            print(f"{i+1:2d}. {sample['sample']}")
            print(f"    Participant ID: {sample['participant_id']}")
            print(f"    Coefficient values: {[f'{x:.2e}' for x in sample['coefficients']]}")
            print(f"    Sum: {sample['sum']:.2e}")
            print(f"    Max: {sample['max']:.2e}")
            print("")
        
        # Create a summary list
        all_tiny_participant_ids = [s['participant_id'] for s in all_tiny_samples]
        all_tiny_sample_names = [s['sample'] for s in all_tiny_samples]
        
        print(f"PARTICIPANT IDs with all-tiny coefficients:")
        print(f"  {all_tiny_participant_ids}")
        print(f"\nSAMPLE NAMES with all-tiny coefficients:")
        print(f"  {all_tiny_sample_names}")
    
    # Detailed reporting for near-zero sum samples (but not all-tiny)
    near_zero_but_not_all_tiny = [s for s in near_zero_sum_samples if not s['all_tiny']]
    if len(near_zero_but_not_all_tiny) > 0:
        print(f"\n--- SAMPLES WITH NEAR-ZERO SUM (but not all tiny) ---")
        print(f"Found {len(near_zero_but_not_all_tiny)} additional samples with very small sums:\n")
        
        for i, sample in enumerate(near_zero_but_not_all_tiny):
            print(f"{i+1:2d}. {sample['sample']}")
            print(f"    Participant ID: {sample['participant_id']}")
            print(f"    Coefficient values: {[f'{x:.2e}' for x in sample['coefficients']]}")
            print(f"    Sum: {sample['sum']:.2e}")
            print(f"    Max: {sample['max']:.2e}")
            print("")
    
    # Save results to file
    results_file = f"{res_dir}/problematic_perceptual_samples_detailed.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("PROBLEMATIC PERCEPTUAL SAMPLES ANALYSIS\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total samples analyzed: {len(sample_analysis)}\n")
        f.write(f"Samples with ALL coefficients < {tiny_threshold}: {len(all_tiny_samples)}\n")
        f.write(f"Samples with sum < {near_zero_sum_threshold}: {len(near_zero_sum_samples)}\n\n")
        
        f.write("SAMPLES WITH ALL COEFFICIENTS ~= 2.22E-16:\n")
        f.write("-" * 40 + "\n")
        for sample in all_tiny_samples:
            f.write(f"Sample: {sample['sample']}\n")
            f.write(f"Participant ID: {sample['participant_id']}\n")
            f.write(f"Coefficients: {[f'{x:.2e}' for x in sample['coefficients']]}\n")
            f.write(f"Sum: {sample['sum']:.2e}\n\n")
        
        f.write("\nPARTICIPANT IDs TO EXCLUDE:\n")
        f.write("-" * 30 + "\n")
        for pid in all_tiny_participant_ids:
            f.write(f"{pid}\n")
    
    print(f"\n✓ Detailed results saved to: {results_file}")
    
    # Return the problematic participant IDs for further use
    return {
        'all_tiny_samples': all_tiny_samples,
        'all_tiny_participant_ids': all_tiny_participant_ids,
        'all_tiny_sample_names': all_tiny_sample_names,
        'near_zero_sum_samples': near_zero_sum_samples
    }

if __name__ == "__main__":
    results = find_problematic_perceptual_samples()
