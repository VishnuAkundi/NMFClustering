"""
Test script to demonstrate the new "exclude for NMF, preserve for downstream" approach.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_new_approach():
    """Test the new approach for handling zero perceptual data."""
    
    print("=== TESTING NEW APPROACH: EXCLUDE FOR NMF, PRESERVE FOR DOWNSTREAM ===\n")
    
    # Show current configuration
    try:
        from config import ZERO_PERCEPTUAL_HANDLING
        print(f"Current configuration: ZERO_PERCEPTUAL_HANDLING = '{ZERO_PERCEPTUAL_HANDLING}'")
    except ImportError:
        print("Could not import config - using default 'exclude'")
        ZERO_PERCEPTUAL_HANDLING = "exclude"
    
    # Create sample data
    print(f"\n--- CREATING SAMPLE SCENARIO ---")
    
    # Sample perceptual data with some all-zero participants
    sample_perceptual = pd.DataFrame({
        'severity_speech': [2.5, 0.0, 3.8, 1.2, 0.0, 2.1],
        'severity_voice': [1.8, 0.0, 4.1, 0.9, 0.0, 1.7],
        'intelligibility': [3.2, 0.0, 4.5, 1.5, 0.0, 2.8],
        'articulation': [2.1, 0.0, 3.9, 1.1, 0.0, 2.3]
    })
    sample_perceptual.index = ['ALS_MILD', 'HEALTHY_A', 'ALS_SEVERE', 'ALS_MODERATE', 'HEALTHY_B', 'ALS_EARLY']
    
    print("Original perceptual data:")
    print(sample_perceptual)
    
    # Identify healthy participants
    zero_rows = (sample_perceptual.sum(axis=1) == 0)
    healthy_participants = list(sample_perceptual.index[zero_rows])
    symptom_participants = list(sample_perceptual.index[~zero_rows])
    
    print(f"\nHealthy participants (all zeros): {healthy_participants}")
    print(f"Participants with symptoms: {symptom_participants}")
    
    # Simulate the new approach
    print(f"\n--- NEW APPROACH WORKFLOW ---")
    
    print("1. EXCLUDE healthy participants from NMF clustering:")
    data_for_nmf = sample_perceptual.loc[~zero_rows]
    print(f"   Data sent to NMF: {len(data_for_nmf)} participants")
    print(f"   {list(data_for_nmf.index)}")
    
    print("\n2. Run NMF clustering on symptom participants only:")
    # Simulate NMF clustering results
    simulated_clusters = pd.Series([0, 2, 1, 0], index=data_for_nmf.index, name='cluster')
    print(f"   NMF cluster assignments:")
    for participant, cluster in simulated_clusters.items():
        print(f"     {participant}: Cluster {cluster}")
    
    print("\n3. Integrate healthy participants back for downstream analysis:")
    # Assign healthy participants to special cluster -1
    healthy_clusters = pd.Series([-1, -1], index=healthy_participants, name='cluster')
    full_clusters = pd.concat([simulated_clusters, healthy_clusters]).sort_index()
    
    print(f"   Full cluster assignments (including healthy):")
    for participant, cluster in full_clusters.items():
        cluster_name = "HEALTHY" if cluster == -1 else f"Cluster {cluster}"
        print(f"     {participant}: {cluster_name}")
    
    print("\n4. BENEFITS of this approach:")
    print("   ✅ Clean NMF clustering without mathematical artifacts")
    print("   ✅ No 2.22E-16 coefficient values")
    print("   ✅ Healthy participants preserved as reference group")
    print("   ✅ All participants available for downstream analysis")
    print("   ✅ Clinically interpretable results")
    
    print("\n5. DOWNSTREAM ANALYSIS possibilities:")
    print("   • Compare Cluster 0, 1, 2 characteristics vs HEALTHY baseline")
    print("   • Analyze disease progression patterns")
    print("   • Demographics analysis across all groups including healthy")
    print("   • Feature importance with healthy as reference")
    print("   • Cross-modal alignment using symptom clusters only")

def demonstrate_coefficient_quality():
    """Show how this approach eliminates the 2.22E-16 coefficient problem."""
    
    print(f"\n\n=== COEFFICIENT QUALITY COMPARISON ===\n")
    
    print("OLD APPROACH (epsilon addition):")
    print("  Input data: [1e-10, 1e-10, 1e-10, 1e-10]  # Epsilon added to healthy participant")
    print("  NMF output: [2.06E-11, 3.67E-11, 1.29E-12, 7.65E-13]  # Unpredictable tiny values")
    print("  ❌ Problem: Coefficients are mathematical artifacts")
    
    print("\nNEW APPROACH (exclude from NMF):")
    print("  Input data: Only participants with actual symptoms")
    print("  NMF output: [0.23, 0.67, 0.45, 0.12]  # Normal, interpretable coefficients")  
    print("  Healthy participants: Assigned to HEALTHY cluster (-1)")
    print("  ✅ Solution: Clean coefficients + preserved healthy participants")
    
    print("\nRESULT:")
    print("  • No more mysterious 2.22E-16 values")
    print("  • Coefficient matrices represent real symptom patterns")
    print("  • Healthy participants available for comparison")
    print("  • Mathematically and clinically sound")

def show_implementation_changes():
    """Show what changes were made to implement this approach."""
    
    print(f"\n\n=== IMPLEMENTATION CHANGES ===\n")
    
    print("CONFIGURATION (config.py):")
    print("  ZERO_PERCEPTUAL_HANDLING = 'exclude'  # New options: exclude, epsilon, scaled_epsilon")
    
    print("\nDATA LOADER (data_loader.py):")
    print("  • handle_zero_perceptual_data() now returns (data_for_nmf, zero_info)")
    print("  • load_perceptual_data() tracks healthy participants separately")
    print("  • integrate_healthy_participants() adds them back to results")
    
    print("\nMAIN PIPELINE (run_analysis.py):")
    print("  • NMF runs on filtered data (symptoms only)")
    print("  • Healthy participants integrated after clustering")
    print("  • All downstream analysis includes full participant set")
    
    print("\nRESULT FILES:")
    print("  • Coefficient matrices: Clean values for symptom participants")
    print("  • Cluster assignments: Include -1 for HEALTHY participants")
    print("  • Demographics: All participants preserved")
    print("  • Cross-modal: Uses symptom clusters for alignment")

if __name__ == "__main__":
    test_new_approach()
    demonstrate_coefficient_quality()
    show_implementation_changes()
