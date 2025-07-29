#!/usr/bin/env python3
"""
ALS Clustering Analysis - Main Execution Script

This script runs the complete clustering analysis pipeline:
1. Load and preprocess data
2. Perform NMF clustering on acoustic and perceptual features
3. Generate visualizations
4. Conduct cross-modal analysis
5. Export results to organized folders

Author: Vishnu Akundi and Leif Simmatis
Date: 2025
"""

import sys
import os
import time
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config import *
from src.data_loader import (
    merge_all_data, filter_demographics, get_demographic_stats
)
from src.nmf_clustering import run_nmf_clustering, get_top_features
from src.visualization import (
    setup_output_directories, plot_pca_clusters, plot_cluster_distributions,
    plot_feature_importance, plot_confusion_matrix, plot_similarity_distributions,
    visualize_cluster_features
)
from src.cross_modal_analysis import run_cross_modal_analysis
from src.results_export import export_all_results

def main():
    """Main execution function."""
    print("=" * 80)
    print("ALS CLUSTERING ANALYSIS - AUTOMATED PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    start_time = time.time()
    
    try:
        # Step 1: Setup and Data Loading
        print("STEP 1: Setting up output directories and loading data...")
        setup_output_directories()
        
        all_data, clinical_summaries, demographics, first_acoustic = merge_all_data()
        demographics_filtered, clinical_dem_filtered = filter_demographics(demographics, clinical_summaries)
        demographic_stats = get_demographic_stats(demographics_filtered)
        
        print(f"✓ Data loaded successfully")
        print(f"  - {all_data.shape[0]} participants")
        print(f"  - {all_data.shape[1]} total features")
        print(f"  - {first_acoustic} perceptual features")
        print(f"  - {all_data.shape[1] - first_acoustic} acoustic features")
        print()
        
        # Step 2: NMF Clustering Analysis
        print("STEP 2: Performing NMF clustering analysis...")
        all_clusters, all_feature_maps, coef_dists, basis_dists, clinical_summaries_filtered = run_nmf_clustering(
            all_data, clinical_summaries, first_acoustic
        )
        
        # Extract top features for each cluster
        top_features_dict = {}
        for data_type in ["acoustic", "perceptual"]:
            if data_type == "acoustic":
                columns = all_data.columns[first_acoustic:]
            else:
                columns = all_data.columns[:first_acoustic]
            
            top_features_dict[data_type] = get_top_features(
                basis_dists[data_type], columns, data_type
            )
        
        print("✓ NMF clustering completed")
        print()
        
        # Step 3: Generate Visualizations
        print("STEP 3: Generating visualizations...")
        
        # PCA cluster plots
        for data_type in ["acoustic", "perceptual"]:
            if data_type == "acoustic":
                V = all_data.iloc[:, first_acoustic:].values
            else:
                V = all_data.iloc[:, :first_acoustic].values
            
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            V_scaled = mms.fit_transform(V)
            
            save_path = f"{PLOTS_DIR}/pca_clusters_{data_type}.png"
            plot_pca_clusters(V_scaled, all_clusters[data_type], data_type, save_path)
        
        # Cluster distribution plots
        for data_type in ["acoustic", "perceptual"]:
            plot_cluster_distributions(
                all_clusters[data_type], clinical_summaries_filtered, data_type, PLOTS_DIR
            )
        
        # Feature importance plots
        plot_feature_importance(basis_dists, all_data, first_acoustic, PLOTS_DIR)
        
        print("✓ Basic visualizations completed")
        print()
        
        # Step 4: Cross-Modal Analysis
        print("STEP 4: Conducting cross-modal analysis...")
        analysis_results, similarity_metrics = run_cross_modal_analysis(
            all_clusters, all_feature_maps, coef_dists, clinical_summaries_filtered
        )
        
        # Generate cross-modal visualizations
        # Confusion matrix
        perceptual_clusters_df = analysis_results['combined_clusters'][['perceptual']]
        acoustic_clusters_df = analysis_results['combined_clusters'][['acoustic']]
        
        confusion_path = f"{PLOTS_DIR}/confusion_matrix.png"
        cramers_v = plot_confusion_matrix(perceptual_clusters_df, acoustic_clusters_df, confusion_path)
        
        # Similarity distributions
        similarity_path = f"{PLOTS_DIR}/similarity_distributions.png"
        plot_similarity_distributions(
            similarity_metrics['js_divergences'],
            similarity_metrics['cosine_similarities'],
            similarity_metrics['bhattacharyya_coefficients'],
            similarity_path
        )
        
        print("✓ Cross-modal analysis completed")
        print(f"  - Cramer's V: {cramers_v:.4f}")
        print()
        
        # Step 5: Detailed Feature Analysis Plots
        print("STEP 5: Generating detailed feature analysis plots...")
        
        catted = analysis_results['combined_clusters']
        for data_type in ["acoustic", "perceptual"]:
            if data_type == "acoustic":
                columns = all_data.columns[first_acoustic:]
            else:
                columns = all_data.columns[:first_acoustic]
            
            temp_data = all_data[columns]
            top_features = top_features_dict[data_type]
            
            for cluster_idx, features in top_features.items():
                feature_names = features["Feature Names"]
                feature_scores = features["Scores"]
                
                # Calculate statistics for this cluster
                cluster_data = temp_data.loc[catted[data_type] == cluster_idx, feature_names]
                means = cluster_data.mean()
                stds = cluster_data.std()
                medians = cluster_data.median()
                q1 = cluster_data.quantile(0.25)
                q3 = cluster_data.quantile(0.75)
                
                # Generate plot
                save_path = f"{PLOTS_DIR}/cluster_features_{data_type}_cluster_{cluster_idx}.png"
                visualize_cluster_features(
                    cluster_idx, feature_names, means.values, stds.values,
                    medians.values, q1.values, q3.values, feature_scores,
                    data_type, save_path
                )
        
        print("✓ Detailed feature analysis plots completed")
        print()
        
        # Step 6: Export Results
        print("STEP 6: Exporting results to organized files...")
        export_all_results(
            all_clusters, all_data, first_acoustic, analysis_results,
            similarity_metrics, demographic_stats, top_features_dict
        )
        
        print("✓ Results export completed")
        print()
        
        # Final Summary
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("OUTPUT SUMMARY:")
        print(f"📊 Plots saved to: {PLOTS_DIR}")
        print(f"📋 Tables saved to: {TABLES_DIR}")
        print(f"📁 Results saved to: {RESULTS_DIR}")
        print()
        print("KEY FINDINGS:")
        print(f"  • Identified {len(analysis_results['combined_clusters']['acoustic'].unique())} acoustic clusters")
        print(f"  • Identified {len(analysis_results['combined_clusters']['perceptual'].unique())} perceptual clusters")
        print(f"  • Cross-modal agreement (Cramer's V): {cramers_v:.4f}")
        print(f"  • Mean cross-modal similarity: {similarity_metrics['cosine_similarities'].mean():.4f}")
        print()
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        print("Analysis failed. Please check the error message above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
