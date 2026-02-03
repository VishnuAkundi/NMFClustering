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
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from src.config import *
from src import config
from src.data_loader import (
    merge_all_data, filter_demographics, get_demographic_stats
)
from src.nmf_clustering import run_nmf_clustering, get_top_features
from src.visualization import (
    setup_output_directories, plot_pca_clusters, plot_cluster_distributions,
    plot_feature_importance, plot_confusion_matrix, plot_similarity_distributions,
    visualize_cluster_feature_statistics, visualize_cluster_feature_importance
)
from src.cross_modal_analysis import run_cross_modal_analysis
from src.results_export import export_all_results


def save_feature_importances_and_values(all_data, first_acoustic, top_features_dict, catted):
    """
    Save feature importances and descriptive statistics for all clusters in both modalities.
    
    Args:
        all_data: Full dataset
        first_acoustic: Index where acoustic features start
        top_features_dict: Dictionary with top features per cluster per modality
        catted: Combined clusters dataframe with cluster assignments
    """
    import pandas as pd
    
    combined_results = []
    
    for data_type in ["acoustic", "perceptual"]:
        # Get appropriate columns for this data type
        if data_type == "acoustic":
            columns = all_data.columns[first_acoustic:]
        else:
            columns = all_data.columns[:first_acoustic]
        
        temp_data = all_data[columns]
        top_features = top_features_dict[data_type]
        
        for cluster_idx, features in top_features.items():
            feature_names = features["Feature Names"]
            feature_scores = features["Scores"]  # NMF importance scores
            
            # Calculate statistics for this cluster
            cluster_data = temp_data.loc[catted[data_type] == cluster_idx, feature_names]
            means = cluster_data.mean()
            stds = cluster_data.std()
            medians = cluster_data.median()
            q1 = cluster_data.quantile(0.25)
            q3 = cluster_data.quantile(0.75)
            
            # Get cluster size
            cluster_size = len(cluster_data)
            
            # Create row for each feature in this cluster
            for i, feature_name in enumerate(feature_names):
                row = {
                    'modality': data_type,
                    'cluster_id': cluster_idx,
                    'cluster_size': cluster_size,
                    'feature_name': feature_name,
                    'feature_rank': i + 1,  # Rank within cluster (1 = most important)
                    'nmf_importance_score': feature_scores[i],
                    'mean_value': means.iloc[i],
                    'std_value': stds.iloc[i],
                    'median_value': medians.iloc[i],
                    'q1_value': q1.iloc[i],
                    'q3_value': q3.iloc[i]
                }
                combined_results.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(combined_results)
    
    # Sort by modality, cluster, then by importance rank
    results_df = results_df.sort_values(['modality', 'cluster_id', 'feature_rank'])
    
    # Save to CSV
    output_path = f"{config.RESULTS_DIR}/feature_importances_and_values.csv"
    results_df.to_csv(output_path, index=False)
    
    print(f"✓ Feature importances and values saved to: {output_path}")
    print(f"  - Total rows: {len(results_df)}")
    print(f"  - Acoustic clusters: {len(top_features_dict['acoustic'])}")
    print(f"  - Perceptual clusters: {len(top_features_dict['perceptual'])}")
    
    return results_df

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
        
        all_data, clinical_summaries, demographics, first_acoustic, zero_perceptual_participants = merge_all_data()
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
        all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists, clinical_summaries_filtered = run_nmf_clustering(
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
                original_V = V.copy()  # For acoustic, original and scaled are from same source
                zero_participants_for_plot = None
            else:
                V = all_data.iloc[:, :first_acoustic].values
                # For perceptual, create a mask based on the zero_perceptual_participants index
                zero_participants_mask = None
                if len(zero_perceptual_participants) > 0:
                    # Create boolean mask for zero participants
                    zero_participants_mask = all_data.index.isin(zero_perceptual_participants)
                    print(f"DEBUG: Creating mask for {len(zero_perceptual_participants)} zero participants")
                    print(f"DEBUG: Mask sum: {zero_participants_mask.sum()}")
                
                # Pass the original data and mask for zero detection
                original_V = V.copy()
                zero_participants_for_plot = zero_participants_mask
            
            from sklearn.preprocessing import MinMaxScaler
            mms = MinMaxScaler()
            V_scaled = mms.fit_transform(V)
            
            save_path = f"{config.PLOTS_DIR}/pca_clusters_{data_type}.png"
            plot_pca_clusters(V_scaled, all_clusters[data_type], data_type, save_path, original_V, zero_participants_for_plot)
        
        # Cluster distribution plots
        for data_type in ["acoustic", "perceptual"]:
            plot_cluster_distributions(
                all_clusters[data_type], clinical_summaries_filtered, data_type, config.PLOTS_DIR
            )
        
        # Feature importance plots
        plot_feature_importance(basis_dists, all_data, first_acoustic, config.PLOTS_DIR)
        
        print("✓ Basic visualizations completed")
        print()
        
        # Step 4: Cross-Modal Analysis
        print("STEP 4: Conducting cross-modal analysis...")
        analysis_results, similarity_metrics = run_cross_modal_analysis(
            all_clusters, all_feature_maps, coef_dists, clinical_summaries_filtered, raw_coef_dists
        )
        
        # Generate cross-modal visualizations
        # Confusion matrix
        perceptual_clusters_df = analysis_results['combined_clusters'][['perceptual']]
        acoustic_clusters_df = analysis_results['combined_clusters'][['acoustic']]
        
        confusion_path = f"{config.PLOTS_DIR}/confusion_matrix.png"
        cramers_v = plot_confusion_matrix(perceptual_clusters_df, acoustic_clusters_df, confusion_path)
        
        # Similarity distributions
        similarity_path = f"{config.PLOTS_DIR}/similarity_distributions.png"
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
                
                # Generate separate plots for statistics and importance
                stats_save_path = f"{config.PLOTS_DIR}/cluster_features_{data_type}_cluster_{cluster_idx}_statistics.png"
                importance_save_path = f"{config.PLOTS_DIR}/cluster_features_{data_type}_cluster_{cluster_idx}_importance.png"
                
                visualize_cluster_feature_statistics(
                    cluster_idx, feature_names, means.values, stds.values,
                    medians.values, q1.values, q3.values, data_type, stats_save_path
                )
                
                visualize_cluster_feature_importance(
                    cluster_idx, feature_names, feature_scores, data_type, importance_save_path
                )
        
        print("✓ Detailed feature analysis plots completed")
        print()
        

         # Step 5.5: Save combined feature importances and statistics
        print("STEP 5.5: Saving feature importances and statistics...")
        feature_results_df = save_feature_importances_and_values(
            all_data, first_acoustic, top_features_dict, catted
        )
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
        print(f"📊 Plots saved to: {config.PLOTS_DIR}")
        print(f"📋 Tables saved to: {config.TABLES_DIR}")
        print(f"📁 Results saved to: {config.RESULTS_DIR}")
        print()
        print("KEY FINDINGS:")
        print(f"  • Identified {len(analysis_results['combined_clusters']['acoustic'].unique())} acoustic clusters")
        print(f"  • Identified {len(analysis_results['combined_clusters']['perceptual'].unique())} perceptual clusters")
        print(f"  • Alignment method: {analysis_results.get('alignment_method', 'jsd_hungarian')}")
        
        if analysis_results.get('alignment_method') == 'column_similarity':
            similarity_metric = analysis_results.get('similarity_metric', 'cosine')
            print(f"  • Similarity metric: {similarity_metric}")
            
            if 'similarity_matrix' in analysis_results:
                similarity_matrix = analysis_results['similarity_matrix']
                avg_similarity = np.mean([similarity_matrix[k, v] for k, v in analysis_results['acoustic_to_perceptual_mapping'].items()])
                print(f"  • Average optimal alignment similarity: {avg_similarity:.4f}")
        else:
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
