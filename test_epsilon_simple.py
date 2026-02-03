"""
Simple test to verify epsilon handling configuration and demonstrate the logic.
"""

import pandas as pd
import numpy as np
import os

def test_epsilon_handling_simple():
    """Test the epsilon handling logic with sample data."""
    
    print("=== TESTING EPSILON HANDLING CONFIGURATION ===\n")
    
    # Load config values
    config_path = "src/config.py"
    if os.path.exists(config_path):
        # Read config file to get values
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Extract config values
        handle_zero = "HANDLE_ZERO_PERCEPTUAL = True" in config_content
        
        # Extract epsilon value
        epsilon_line = [line for line in config_content.split('\n') if 'PERCEPTUAL_EPSILON =' in line]
        if epsilon_line:
            epsilon_str = epsilon_line[0].split('=')[1].strip()
            epsilon_value = float(epsilon_str)
        else:
            epsilon_value = 1e-10
            
        print(f"✓ Configuration loaded:")
        print(f"  HANDLE_ZERO_PERCEPTUAL = {handle_zero}")
        print(f"  PERCEPTUAL_EPSILON = {epsilon_value}")
    else:
        print(f"❌ Config file not found at {config_path}")
        return
    
    # Create sample data that mimics the problematic scenario
    print(f"\n--- CREATING SAMPLE DATA ---")
    sample_data = pd.DataFrame({
        'feature1': [1.5, 0.0, 2.1, 0.0, 0.8],
        'feature2': [2.3, 0.0, 1.8, 0.0, 1.2],
        'feature3': [0.9, 0.0, 3.2, 0.0, 2.1],
        'feature4': [1.7, 0.0, 0.5, 0.0, 1.9]
    })
    sample_data.index = ['HEALTHY_A', 'HEALTHY_B', 'MILD_ALS_C', 'HEALTHY_D', 'MODERATE_ALS_E']
    
    print("Original sample data:")
    print(sample_data)
    print(f"\nRow sums: {sample_data.sum(axis=1).values}")
    
    # Identify zero rows
    zero_threshold = 1e-12
    row_sums = sample_data.sum(axis=1)
    zero_rows = row_sums < zero_threshold
    
    print(f"\n--- ZERO ROW IDENTIFICATION ---")
    print(f"Zero threshold: {zero_threshold}")
    print(f"Participants with all-zero data: {list(sample_data.index[zero_rows])}")
    print(f"Number of zero participants: {zero_rows.sum()}")
    
    # Apply epsilon handling
    if handle_zero and zero_rows.sum() > 0:
        print(f"\n--- APPLYING EPSILON HANDLING ---")
        processed_data = sample_data.copy()
        processed_data.loc[zero_rows, :] = epsilon_value
        
        print(f"Added epsilon ({epsilon_value}) to {zero_rows.sum()} participants")
        print("\nProcessed data:")
        print(processed_data)
        print(f"\nRow sums after epsilon: {processed_data.sum(axis=1).values}")
        
        # Verify no more zero rows
        zero_rows_after = (processed_data.sum(axis=1) < zero_threshold).sum()
        print(f"\n✓ Zero rows eliminated: {zero_rows_after == 0}")
        
    else:
        print(f"\n--- EPSILON HANDLING DISABLED ---")
        print("No changes made to data")

def demonstrate_nmf_impact():
    """Demonstrate why epsilon is needed for NMF."""
    
    print(f"\n\n=== WHY EPSILON IS NEEDED FOR NMF ===\n")
    
    print("NMF (Non-negative Matrix Factorization) works by decomposing:")
    print("  Data ≈ W × H")
    print("  where W = features/basis, H = coefficients")
    print()
    print("When a row is all zeros:")
    print("  [0, 0, 0, 0] ≈ W₁×H₁ + W₂×H₂ + W₃×H₃")
    print("  This forces H₁=H₂=H₃=0 (or near-zero like 2.22e-16)")
    print()
    print("With epsilon:")
    print("  [1e-10, 1e-10, 1e-10, 1e-10] ≈ W₁×H₁ + W₂×H₂ + W₃×H₃")
    print("  This allows normal coefficient values representing 'minimal symptoms'")
    print()
    print("Clinical interpretation:")
    print("  • Zero values = 'undefined/missing' (problematic for NMF)")
    print("  • Epsilon values = 'virtually no symptoms' (valid for clustering)")
    print("  • This preserves healthy participants as a reference group")

def check_existing_results():
    """Check if the problematic 2.22e-16 values exist in current results."""
    
    print(f"\n\n=== CHECKING EXISTING RESULTS ===\n")
    
    results_dir = "outputs/run9/results/"
    coef_file = f"{results_dir}nmf_coefficient_matrix_perceptual.csv"
    
    if os.path.exists(coef_file):
        try:
            coef_matrix = pd.read_csv(coef_file, index_col=0)
            print(f"✓ Loaded existing coefficient matrix: {coef_matrix.shape}")
            
            # Check for tiny values
            tiny_threshold = 1e-15
            tiny_count = 0
            problematic_samples = []
            
            for col in coef_matrix.columns:
                if (coef_matrix[col] < tiny_threshold).all():
                    tiny_count += 1
                    problematic_samples.append(col)
            
            print(f"Found {tiny_count} samples with all coefficients < {tiny_threshold}")
            
            if tiny_count > 0:
                print(f"These represent the {tiny_count} healthy participants")
                print("After implementing epsilon handling, these will have normal coefficients")
                print(f"First few problematic samples: {problematic_samples[:5]}")
            else:
                print("No problematic tiny coefficients found")
                
        except Exception as e:
            print(f"Could not analyze coefficient matrix: {e}")
    else:
        print(f"No existing results found at {coef_file}")
        print("Run the analysis with epsilon handling to generate new results")

if __name__ == "__main__":
    test_epsilon_handling_simple()
    demonstrate_nmf_impact()
    check_existing_results()
