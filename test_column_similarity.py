#!/usr/bin/env python3
"""
Test script demonstrating the new column similarity alignment method.

This script shows how to configure and use the new "column_similarity" 
alignment method in your NMF clustering pipeline.

Author: Vishnu Akundi and Leif Simmatis
Date: 2025
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import config

def demonstrate_column_similarity_config():
    """Demonstrate how to configure the column similarity method."""
    
    print("COLUMN SIMILARITY CONFIGURATION EXAMPLE")
    print("=" * 50)
    print()
    
    print("1. To use the default JSD + Hungarian alignment method:")
    print("   Set in config.py:")
    print("   ALIGNMENT_METHOD = 'jsd_hungarian'")
    print()
    
    print("2. To use the new column similarity alignment method:")
    print("   Set in config.py:")
    print("   ALIGNMENT_METHOD = 'column_similarity'")
    print("   COLUMN_SIMILARITY_METRIC = 'cosine'  # or 'correlation'")
    print()
    
    print("CURRENT CONFIGURATION:")
    print(f"   ALIGNMENT_METHOD = '{config.ALIGNMENT_METHOD}'")
    
    if hasattr(config, 'COLUMN_SIMILARITY_METRIC'):
        print(f"   COLUMN_SIMILARITY_METRIC = '{config.COLUMN_SIMILARITY_METRIC}'")
    else:
        print("   COLUMN_SIMILARITY_METRIC = Not configured (will use default)")
    print()
    
    print("COLUMN SIMILARITY METHOD OVERVIEW:")
    print("=" * 40)
    print("When 'column_similarity' is selected, the pipeline will:")
    print()
    print("1. Take coefficient matrices W_A and W_B from NMF of each modality")
    print("2. Row-normalize them so each patient's memberships sum to 1")
    print("3. Compute similarity matrix C where:")
    print("   C[i,j] = similarity between A's component i and B's component j")
    print("4. Use cosine similarity or correlation as the similarity metric")
    print("5. If kA == kB, apply Hungarian algorithm for optimal 1-to-1 mapping")
    print("6. Save the mapping and similarity matrix to results folder")
    print("7. Generate a heatmap visualization of the similarity matrix")
    print()
    
    print("OUTPUT FILES (when using column similarity):")
    print("=" * 40)
    print("Results folder will contain:")
    print("• similarity_matrix_{metric}.csv - Full similarity matrix")
    print("• cluster_mapping_{metric}.csv - Optimal cluster mapping")
    print("• similarity_matrix_metadata_{metric}.json - Analysis metadata")
    print()
    print("Plots folder will contain:")
    print("• similarity_matrix_heatmap_{metric}.png - Similarity heatmap")
    print()
    print("Where {metric} is either 'cosine' or 'correlation'")
    print()


def show_similarity_matrix_interpretation():
    """Show how to interpret the similarity matrix results."""
    
    print("INTERPRETING SIMILARITY MATRIX RESULTS")
    print("=" * 40)
    print()
    
    print("The similarity matrix C has dimensions kA × kB where:")
    print("• kA = number of acoustic components")
    print("• kB = number of perceptual components")
    print()
    
    print("Each element C[i,j] represents:")
    print("• Similarity between acoustic component i and perceptual component j")
    print("• Computed across all shared patients")
    print("• Range depends on similarity metric:")
    print("  - Cosine similarity: [0, 1] (higher = more similar)")
    print("  - Correlation: [-1, 1] (closer to ±1 = more similar)")
    print()
    
    print("The heatmap visualization shows:")
    print("• Rows = Acoustic components (A0, A1, A2, ...)")
    print("• Columns = Perceptual components (P0, P1, P2, ...)")
    print("• Color intensity = similarity strength")
    print("• Red borders = optimal mapping (when kA == kB)")
    print()
    
    print("QUICK INSPECTION TIPS:")
    print("=" * 25)
    print("1. Look for high similarity values along diagonal patterns")
    print("2. Check if optimal mappings make intuitive sense")
    print("3. Examine similarity distribution across the matrix")
    print("4. Compare results between cosine and correlation metrics")
    print()


if __name__ == "__main__":
    demonstrate_column_similarity_config()
    print()
    show_similarity_matrix_interpretation()
    
    print("To run analysis with column similarity:")
    print("1. Edit src/config.py to set ALIGNMENT_METHOD = 'column_similarity'")
    print("2. Choose similarity metric: COLUMN_SIMILARITY_METRIC = 'cosine' or 'correlation'")
    print("3. Run: python run_analysis.py")
    print()
    print("The pipeline will automatically use the new method and generate all outputs!")
