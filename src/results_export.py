"""
Results export module for saving analysis outputs to organized files.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any
from .config import *
from . import config


def save_demographic_stats(demographic_stats: Dict, save_path: str) -> None:
    """
    Save demographic statistics to CSV.
    
    Args:
        demographic_stats: Dictionary of demographic statistics
        save_path: Path to save the CSV file
    """
    df_stats = pd.DataFrame(demographic_stats).T
    df_stats.to_csv(save_path)
    print(f"Demographic statistics saved: {save_path}")


def save_cluster_results(all_clusters: Dict, all_data: pd.DataFrame, 
                        first_acoustic: int, save_dir: str) -> None:
    """
    Save cluster assignments and basic statistics.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        all_data: Combined dataset
        first_acoustic: Index of first acoustic feature
        save_dir: Directory to save results
    """
    for data_type in ["acoustic", "perceptual"]:
        clusters = all_clusters[data_type]
        
        # Create results dataframe
        cluster_df = pd.DataFrame({
            'Participant_ID': all_data.index[:len(clusters[0])],
            'Cluster_Assignment': clusters[0]
        })
        
        # Add cluster statistics
        from collections import Counter
        cluster_counts = Counter(clusters[0])
        
        stats_data = []
        for cluster_id, count in cluster_counts.items():
            stats_data.append({
                'Cluster_ID': cluster_id,
                'Participant_Count': count,
                'Percentage': (count / len(clusters[0])) * 100
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Save files
        cluster_file = f"{save_dir}/cluster_assignments_{data_type}.csv"
        stats_file = f"{save_dir}/cluster_statistics_{data_type}.csv"
        
        cluster_df.to_csv(cluster_file, index=False)
        stats_df.to_csv(stats_file, index=False)
        
        print(f"Cluster assignments saved: {cluster_file}")
        print(f"Cluster statistics saved: {stats_file}")


def save_top_features(top_features_dict: Dict, save_dir: str) -> None:
    """
    Save top features for each cluster to CSV files.
    
    Args:
        top_features_dict: Dictionary of top features per cluster
        save_dir: Directory to save results
    """
    for data_type, features_data in top_features_dict.items():
        all_features_data = []
        
        for cluster_idx, cluster_features in features_data.items():
            for i, (feature, score) in enumerate(zip(
                cluster_features["Feature Names"], 
                cluster_features["Scores"]
            )):
                all_features_data.append({
                    'Cluster_ID': cluster_idx,
                    'Rank': i + 1,
                    'Feature_Name': feature,
                    'Importance_Score': score
                })
        
        df = pd.DataFrame(all_features_data)
        save_path = f"{save_dir}/top_features_{data_type}.csv"
        df.to_csv(save_path, index=False)
        print(f"Top features saved: {save_path}")


def save_cross_modal_results(analysis_results: Dict, similarity_metrics: Dict, 
                           save_dir: str) -> None:
    """
    Save cross-modal analysis results.
    
    Args:
        analysis_results: Dictionary of analysis results
        similarity_metrics: Dictionary of similarity metrics
        save_dir: Directory to save results
    """
    # Save cluster feature mappings
    for mode, mappings in analysis_results['cluster_mappings'].items():
        mapping_data = []
        for cluster_id, features in mappings.items():
            for feature in features:
                mapping_data.append({
                    'Cluster_ID': cluster_id,
                    'Feature_Name': feature
                })
        
        df = pd.DataFrame(mapping_data)
        save_path = f"{save_dir}/cluster_feature_mapping_{mode}.csv"
        df.to_csv(save_path, index=False)
        print(f"Cluster feature mapping saved: {save_path}")
    
    # Save severity statistics
    severity_data = []
    for mode, mode_stats in analysis_results['severity_stats'].items():
        for cluster_id, stats in mode_stats.items():
            severity_data.append({
                'Data_Type': mode,
                'Cluster_ID': cluster_id,
                'Participant_Count': stats['count'],
                'Median_ALSIBD': stats['median'],
                'Q1_ALSIBD': stats['q1'],
                'Q3_ALSIBD': stats['q3']
            })
    
    severity_df = pd.DataFrame(severity_data)
    severity_path = f"{save_dir}/cluster_severity_statistics.csv"
    severity_df.to_csv(severity_path, index=False)
    print(f"Severity statistics saved: {severity_path}")
    
    # Save acoustic-to-perceptual mapping
    mapping_data = []
    for k, v in analysis_results['acoustic_to_perceptual_mapping'].items():
        mapping_entry = {'Acoustic_Cluster': k, 'Perceptual_Cluster': v}
        
        # Add similarity score if using column similarity method
        if 'similarity_matrix' in analysis_results:
            similarity_matrix = analysis_results['similarity_matrix']
            mapping_entry['Similarity_Score'] = similarity_matrix[k, v]
            
        mapping_data.append(mapping_entry)
    
    mapping_df = pd.DataFrame(mapping_data)
    mapping_path = f"{save_dir}/acoustic_perceptual_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Acoustic-perceptual mapping saved: {mapping_path}")
    
    # Save method-specific results
    alignment_method = analysis_results.get('alignment_method', 'jsd_hungarian')
    
    if alignment_method == 'column_similarity' and 'similarity_matrix' in analysis_results:
        # Save full similarity matrix
        similarity_matrix = analysis_results['similarity_matrix']
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=[f'Acoustic_{i}' for i in range(similarity_matrix.shape[0])],
            columns=[f'Perceptual_{j}' for j in range(similarity_matrix.shape[1])]
        )
        
        similarity_metric = analysis_results.get('similarity_metric', 'cosine')
        similarity_matrix_path = f"{save_dir}/full_similarity_matrix_{similarity_metric}.csv"
        similarity_df.to_csv(similarity_matrix_path)
        print(f"Full similarity matrix saved: {similarity_matrix_path}")
        
        # Save similarity matrix metadata
        metadata = {
            'alignment_method': alignment_method,
            'similarity_metric': similarity_metric,
            'matrix_shape': similarity_matrix.shape,
            'matrix_range': [float(similarity_matrix.min()), float(similarity_matrix.max())],
            'description': f"C[i,j] = {similarity_metric} similarity between acoustic component i and perceptual component j, computed over shared patients"
        }
        
        metadata_path = f"{save_dir}/similarity_matrix_metadata_{similarity_metric}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Similarity matrix metadata saved: {metadata_path}")
    
    # Save similarity metrics (patient-level similarities)
    similarity_df = pd.DataFrame({
        'Participant_Index': range(len(similarity_metrics['js_divergences'])),
        'JS_Divergence': similarity_metrics['js_divergences'],
        'Cosine_Similarity': similarity_metrics['cosine_similarities'],
        'Bhattacharyya_Coefficient': similarity_metrics['bhattacharyya_coefficients']
    })
    similarity_path = f"{save_dir}/cross_modal_similarities.csv"
    similarity_df.to_csv(similarity_path, index=False)
    print(f"Cross-modal similarities saved: {similarity_path}")
    
    # Save combined cluster assignments
    combined_path = f"{save_dir}/combined_cluster_assignments.csv"
    analysis_results['combined_clusters'].to_csv(combined_path)
    print(f"Combined cluster assignments saved: {combined_path}")


def save_summary_report(analysis_results: Dict, similarity_metrics: Dict, 
                       demographic_stats: Dict, save_path: str) -> None:
    """
    Generate and save a summary report of the analysis.
    
    Args:
        analysis_results: Dictionary of analysis results
        similarity_metrics: Dictionary of similarity metrics
        demographic_stats: Dictionary of demographic statistics
        save_path: Path to save the report
    """
    report = []
    report.append("ALS CLUSTERING ANALYSIS - SUMMARY REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Dataset summary
    report.append("DATASET SUMMARY:")
    combined_clusters = analysis_results['combined_clusters']
    report.append(f"Total participants: {len(combined_clusters)}")
    report.append(f"Acoustic clusters: {len(combined_clusters['acoustic'].unique())}")
    report.append(f"Perceptual clusters: {len(combined_clusters['perceptual'].unique())}")
    report.append("")
    
    # Analysis method summary
    alignment_method = analysis_results.get('alignment_method', 'jsd_hungarian')
    report.append("ANALYSIS METHOD:")
    report.append(f"Alignment method: {alignment_method}")
    
    if alignment_method == 'column_similarity':
        similarity_metric = analysis_results.get('similarity_metric', 'cosine')
        report.append(f"Similarity metric: {similarity_metric}")
        
        if 'similarity_matrix' in analysis_results:
            similarity_matrix = analysis_results['similarity_matrix']
            report.append(f"Similarity matrix shape: {similarity_matrix.shape}")
            report.append(f"Similarity range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
            
            # Report optimal mapping scores
            mapping = analysis_results['acoustic_to_perceptual_mapping']
            if len(mapping) == similarity_matrix.shape[0] and len(set(mapping.values())) == len(mapping.values()):
                total_similarity = sum(similarity_matrix[k, v] for k, v in mapping.items())
                avg_similarity = total_similarity / len(mapping)
                report.append(f"Average optimal alignment similarity: {avg_similarity:.4f}")
    
    report.append("")
    
    # Demographic statistics
    report.append("DEMOGRAPHIC STATISTICS:")
    for var, stats in demographic_stats.items():
        if isinstance(stats, dict) and 'median' in stats:
            report.append(f"{var}:")
            report.append(f"  Median: {stats['median']:.2f}")
            report.append(f"  IQR: {stats['iqr']:.2f}")
    report.append("")
    
    # Cluster mapping
    report.append("ACOUSTIC-PERCEPTUAL CLUSTER MAPPING:")
    mapping = analysis_results['acoustic_to_perceptual_mapping']
    
    if alignment_method == 'column_similarity' and 'similarity_matrix' in analysis_results:
        similarity_matrix = analysis_results['similarity_matrix']
        for acoustic, perceptual in mapping.items():
            similarity_score = similarity_matrix[acoustic, perceptual]
            report.append(f"  Acoustic Cluster {acoustic} -> Perceptual Cluster {perceptual} (similarity: {similarity_score:.4f})")
    else:
        for acoustic, perceptual in mapping.items():
            report.append(f"  Acoustic Cluster {acoustic} -> Perceptual Cluster {perceptual}")
    report.append("")
    
    # Cross-modal similarity
    report.append("CROSS-MODAL SIMILARITY METRICS:")
    report.append(f"Mean JS Divergence: {np.mean(similarity_metrics['js_divergences']):.4f}")
    report.append(f"Mean Cosine Similarity: {np.mean(similarity_metrics['cosine_similarities']):.4f}")
    report.append(f"Mean Bhattacharyya Coefficient: {np.mean(similarity_metrics['bhattacharyya_coefficients']):.4f}")
    report.append("")
    
    # Severity analysis
    report.append("SEVERITY ANALYSIS BY CLUSTER:")
    for mode in ['acoustic', 'perceptual']:
        report.append(f"{mode.upper()} CLUSTERS:")
        for cluster_id, stats in analysis_results['severity_stats'][mode].items():
            report.append(f"  Cluster {cluster_id}:")
            report.append(f"    Participants: {stats['count']}")
            report.append(f"    Median ALSIBD: {stats['median']:.2f}")
            report.append(f"    Q1-Q3 ALSIBD: {stats['q1']:.2f} - {stats['q3']:.2f}")
        report.append("")
    
    # Feature summary
    report.append("TOP CLUSTER FEATURES:")
    for mode in ['acoustic', 'perceptual']:
        cluster_mappings = analysis_results['cluster_mappings'][mode]
        report.append(f"{mode.upper()} FEATURES:")
        for cluster_id, features in cluster_mappings.items():
            report.append(f"  Cluster {cluster_id}: {', '.join(features[:3])}...")
        report.append("")
    
    # Save report
    with open(save_path, 'w') as f:
        f.write('\\n'.join(report))
    
    print(f"Summary report saved: {save_path}")


def export_all_results(all_clusters: Dict, all_data: pd.DataFrame, first_acoustic: int,
                      analysis_results: Dict, similarity_metrics: Dict, 
                      demographic_stats: Dict, top_features_dict: Dict) -> None:
    """
    Export all analysis results to organized files.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        all_data: Combined dataset
        first_acoustic: Index of first acoustic feature
        analysis_results: Dictionary of analysis results
        similarity_metrics: Dictionary of similarity metrics
        demographic_stats: Dictionary of demographic statistics
        top_features_dict: Dictionary of top features
    """
    print("\\n" + "="*50)
    print("EXPORTING RESULTS")
    print("="*50)
    
    # Create output directories
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save demographic statistics
    demo_path = f"{config.TABLES_DIR}/demographic_statistics.csv"
    save_demographic_stats(demographic_stats, demo_path)
    
    # Save cluster results
    save_cluster_results(all_clusters, all_data, first_acoustic, config.TABLES_DIR)
    
    # Save top features
    save_top_features(top_features_dict, config.TABLES_DIR)
    
    # Save cross-modal analysis results
    save_cross_modal_results(analysis_results, similarity_metrics, config.RESULTS_DIR)
    
    # Generate summary report
    report_path = f"{config.RESULTS_DIR}/analysis_summary_report.txt"
    save_summary_report(analysis_results, similarity_metrics, demographic_stats, report_path)
    
    print("\\nAll results exported successfully!")
    print(f"Tables saved to: {config.TABLES_DIR}")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
