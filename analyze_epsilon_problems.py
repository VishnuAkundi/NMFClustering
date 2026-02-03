"""
Comparison script to demonstrate different approaches for handling zero perceptual data.
Shows why epsilon approach produces inconsistent tiny coefficients.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import NMF

def demonstrate_epsilon_problems():
    """Demonstrate why epsilon approach produces inconsistent results."""
    
    print("=== DEMONSTRATING EPSILON APPROACH PROBLEMS ===\n")
    
    # Create sample data mimicking your situation
    normal_data = np.array([
        [2.5, 1.8, 3.2, 1.1],  # Participant with moderate symptoms
        [1.2, 0.9, 1.5, 0.7],  # Participant with mild symptoms  
        [3.8, 2.9, 4.1, 2.2],  # Participant with severe symptoms
    ])
    
    zero_data = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Healthy participant (all zeros)
    ])
    
    epsilon_data = np.array([
        [1e-10, 1e-10, 1e-10, 1e-10],  # Same participant with epsilon
    ])
    
    print("Normal participants data:")
    print(normal_data)
    print("\nHealthy participant (original):")
    print(zero_data)
    print("\nHealthy participant (with epsilon):")
    print(epsilon_data)
    
    # Test NMF with normal data only
    print(f"\n--- NMF WITH NORMAL DATA ONLY ---")
    nmf_normal = NMF(n_components=3, random_state=42, max_iter=1000)
    normal_coeffs = nmf_normal.fit_transform(normal_data)
    print("Normal data coefficients:")
    print(normal_coeffs)
    print(f"Coefficient ranges: {normal_coeffs.min():.2e} to {normal_coeffs.max():.2e}")
    
    # Test NMF with epsilon data included
    print(f"\n--- NMF WITH EPSILON DATA INCLUDED ---")
    combined_data = np.vstack([normal_data, epsilon_data])
    nmf_epsilon = NMF(n_components=3, random_state=42, max_iter=1000)
    epsilon_coeffs = nmf_epsilon.fit_transform(combined_data)
    
    print("Combined data coefficients:")
    print(epsilon_coeffs)
    print(f"Coefficient ranges: {epsilon_coeffs.min():.2e} to {epsilon_coeffs.max():.2e}")
    
    # Show the problematic epsilon coefficients
    epsilon_participant_coeffs = epsilon_coeffs[-1, :]  # Last row (epsilon participant)
    print(f"\nEpsilon participant coefficients: {epsilon_participant_coeffs}")
    print(f"These are similar to your 2.06E-11, 3.67E-11 values!")
    
    # Show how different random seeds produce different tiny coefficients
    print(f"\n--- EPSILON INSTABILITY ACROSS RANDOM SEEDS ---")
    seeds = [42, 123, 456, 789]
    for seed in seeds:
        nmf_test = NMF(n_components=3, random_state=seed, max_iter=1000)
        test_coeffs = nmf_test.fit_transform(combined_data)
        epsilon_coeffs_test = test_coeffs[-1, :]
        print(f"Seed {seed}: {epsilon_coeffs_test}")
    
    print("\n❌ PROBLEM: Epsilon coefficients are unpredictable and seed-dependent!")

def demonstrate_better_approaches():
    """Show better approaches for handling zero data."""
    
    print(f"\n\n=== BETTER APPROACHES ===\n")
    
    # Sample perceptual data
    perceptual_data = pd.DataFrame({
        'feature1': [2.5, 0.0, 3.2, 1.1, 0.0],
        'feature2': [1.8, 0.0, 2.1, 0.9, 0.0], 
        'feature3': [3.1, 0.0, 4.1, 1.5, 0.0],
        'feature4': [2.2, 0.0, 2.9, 1.2, 0.0]
    })
    perceptual_data.index = ['ALS_MILD', 'HEALTHY_A', 'ALS_SEVERE', 'ALS_MODERATE', 'HEALTHY_B']
    
    print("Original perceptual data:")
    print(perceptual_data)
    
    # Identify zero rows
    zero_rows = (perceptual_data.sum(axis=1) == 0)
    print(f"\nHealthy participants (zero rows): {list(perceptual_data.index[zero_rows])}")
    
    # Approach 1: Exclude zero rows
    print(f"\n--- APPROACH 1: EXCLUDE ZERO ROWS ---")
    data_excluded = perceptual_data.loc[~zero_rows]
    print(f"Data for NMF clustering:")
    print(data_excluded)
    print(f"Result: Clean NMF clustering of {len(data_excluded)} participants with symptoms")
    print("Healthy participants analyzed separately or used as reference group")
    
    # Approach 2: Scaled epsilon  
    print(f"\n--- APPROACH 2: SCALED EPSILON ---")
    data_scaled = perceptual_data.copy()
    mean_non_zero = data_excluded.values[data_excluded.values > 0].mean()
    scaled_epsilon = mean_non_zero * 0.01  # 1% of mean
    data_scaled.loc[zero_rows, :] = scaled_epsilon
    
    print(f"Mean non-zero value: {mean_non_zero:.3f}")
    print(f"Scaled epsilon (1%): {scaled_epsilon:.3e}")
    print(f"Data with scaled epsilon:")
    print(data_scaled)
    print("Result: Epsilon is proportional to actual data scale")
    
    # Approach 3: Separate analysis workflow
    print(f"\n--- APPROACH 3: SEPARATE ANALYSIS WORKFLOW ---")
    print("1. Run NMF on non-zero participants to find symptom patterns")
    print("2. Assign healthy participants to 'Cluster 0: Healthy/Reference'")
    print("3. Compare symptom clusters against healthy reference")
    print("4. Analyze disease progression patterns")
    print("Result: Preserves clinical meaning without mathematical artifacts")

def recommend_best_approach():
    """Recommend the best approach for your specific case."""
    
    print(f"\n\n=== RECOMMENDATION FOR YOUR ANALYSIS ===\n")
    
    print("Given your findings of 2.06E-11, 3.67E-11 coefficient values:")
    print()
    print("❌ AVOID: Fixed epsilon approach")
    print("   • Produces unpredictable tiny coefficients (as you observed)")
    print("   • May bias NMF basis vectors")
    print("   • Doesn't represent meaningful 'healthy' state")
    print()
    print("✅ RECOMMENDED: Exclude approach with separate analysis")
    print("   • Run NMF clustering on participants with symptoms only")
    print("   • Treat healthy participants as reference/control group")
    print("   • Compare cluster characteristics against healthy baseline")
    print("   • Clinically interpretable and mathematically sound")
    print()
    print("✅ ALTERNATIVE: Scaled epsilon approach")
    print("   • Use epsilon = 1% of mean non-zero perceptual value")
    print("   • More proportional to actual data scale")
    print("   • Still may produce small coefficients but more predictable")
    print()
    print("IMPLEMENTATION:")
    print("   Set ZERO_PERCEPTUAL_HANDLING = 'exclude' in config.py")
    print("   This will give you clean, interpretable results")

if __name__ == "__main__":
    demonstrate_epsilon_problems()
    demonstrate_better_approaches() 
    recommend_best_approach()
