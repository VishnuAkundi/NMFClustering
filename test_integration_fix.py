"""
Quick test to verify the integration fix works.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_integration_fix():
    """Test that the integration works with pandas Series."""
    
    print("=== TESTING INTEGRATION FIX ===\n")
    
    # Simulate NMF cluster results as pandas Series (new format)
    nmf_participants = ['ALS_MILD', 'ALS_SEVERE', 'ALS_MODERATE', 'ALS_EARLY']
    
    cluster_results = {
        'acoustic': pd.Series([0, 2, 1, 0], index=nmf_participants, name='cluster'),
        'perceptual': pd.Series([1, 3, 2, 1], index=nmf_participants, name='cluster')
    }
    
    print("NMF cluster results (pandas Series):")
    for data_type, clusters in cluster_results.items():
        print(f"{data_type}:")
        print(f"  {dict(clusters)}")
    
    # Simulate healthy participants info
    healthy_participants = ['HEALTHY_A', 'HEALTHY_B']
    zero_participants_info = pd.DataFrame({
        'is_healthy_baseline': [True, True],
        'perceptual_cluster': ['HEALTHY', 'HEALTHY'],
        'exclusion_reason': ['all_zero_perceptual_data', 'all_zero_perceptual_data']
    }, index=healthy_participants)
    
    print(f"\nHealthy participants to integrate:")
    print(f"  {list(zero_participants_info.index)}")
    
    # Test integration
    try:
        from data_loader import integrate_healthy_participants
        
        print(f"\n--- RUNNING INTEGRATION ---")
        integrated_results = integrate_healthy_participants(cluster_results, zero_participants_info)
        
        print(f"\nINTEGRATED RESULTS:")
        for data_type, clusters in integrated_results.items():
            print(f"{data_type}:")
            for participant, cluster in clusters.items():
                cluster_name = "HEALTHY" if cluster == -1 else f"Cluster {cluster}"
                print(f"  {participant}: {cluster_name}")
        
        print(f"\n✅ SUCCESS: Integration completed without errors!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Integration test failed")

if __name__ == "__main__":
    test_integration_fix()
