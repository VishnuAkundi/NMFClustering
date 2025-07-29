"""
Cross-modal analysis module for comparing acoustic and perceptual clusters.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, Any
from .config import *


def average_distributions(distributions: np.ndarray, assignments: np.ndarray, 
                        n_clusters: int) -> np.ndarray:
    """
    Compute average probability distribution per cluster.
    
    Args:
        distributions: Probability distributions for each sample
        assignments: Cluster assignments
        n_clusters: Number of clusters
        
    Returns:
        Average distributions per cluster
    """
    assignments = np.asarray(assignments).flatten()
    avg = []
    for k in range(n_clusters):
        cluster_mask = assignments == k
        if np.any(cluster_mask):
            avg.append(np.mean(distributions[cluster_mask], axis=0))
        else:
            avg.append(np.zeros(distributions.shape[1]))
    return np.squeeze(np.array(avg))


def align_clusters(coef_dists: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Align acoustic and perceptual clusters using optimal assignment.
    
    Args:
        coef_dists: Dictionary containing coefficient distributions
        
    Returns:
        Tuple of (aligned_acoustic_probs, perceptual_probs, mapping_dict)
    """
    print("Aligning acoustic and perceptual clusters...")
    
    # Get probability distributions
    probs_acoustic = coef_dists["acoustic"].T
    probs_perceptual = coef_dists["perceptual"].T
    
    print(f"Acoustic probabilities shape: {probs_acoustic.shape}")
    print(f"Perceptual probabilities shape: {probs_perceptual.shape}")
    
    # Hard assignments
    acoustic_labels = np.argmax(probs_acoustic, axis=1).reshape(-1)
    perceptual_labels = np.argmax(probs_perceptual, axis=1).reshape(-1)
    
    n_clusters = probs_acoustic.shape[1]
    
    # Get average distributions per cluster
    acoustic_avg = average_distributions(probs_acoustic, acoustic_labels, n_clusters)
    perceptual_avg = average_distributions(probs_perceptual, perceptual_labels, n_clusters)
    
    print(f"Acoustic average shape: {acoustic_avg.shape}")
    print(f"Perceptual average shape: {perceptual_avg.shape}")
    
    # Build cost matrix of JS divergence
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            cost_matrix[i, j] = jensenshannon(acoustic_avg[i], perceptual_avg[j])**2
    
    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create label mapping
    acoustic_to_perceptual = dict(zip(row_ind, col_ind))
    
    # Permute acoustic probabilities to match perceptual clusters
    perm = [k for k, v in sorted(acoustic_to_perceptual.items(), key=lambda x: x[1])]
    probs_acoustic_aligned = probs_acoustic[:, perm]
    
    print(f"Cluster alignment mapping: {acoustic_to_perceptual}")
    
    return probs_acoustic_aligned, probs_perceptual, acoustic_to_perceptual


def compute_similarity_metrics(probs_acoustic_aligned: np.ndarray, 
                             probs_perceptual: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute similarity metrics between aligned probability distributions.
    
    Args:
        probs_acoustic_aligned: Aligned acoustic probabilities
        probs_perceptual: Perceptual probabilities
        
    Returns:
        Tuple of (js_divergences, cosine_similarities, bhattacharyya_coefficients)
    """
    print("Computing similarity metrics...")
    
    # Jensen-Shannon divergence (symmetrized and squared)
    js_divs = [jensenshannon(np.array(a)[0] if a.ndim > 1 else np.array(a), 
                           np.array(p)[0] if p.ndim > 1 else np.array(p), base=2)**2
              for a, p in zip(probs_acoustic_aligned, probs_perceptual)]
    
    # Cosine similarity
    cos_sims = [cosine_similarity(np.asarray(a).reshape(1, -1), 
                                np.asarray(p).reshape(1, -1))[0, 0]
               for a, p in zip(probs_acoustic_aligned, probs_perceptual)]
    
    # Bhattacharyya coefficient
    def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Bhattacharyya coefficient between two distributions."""
        return np.sum(np.sqrt(p * q))
    
    bhatt_coeffs = [bhattacharyya_coefficient(np.array(a).squeeze(), np.array(p).squeeze())
                   for a, p in zip(probs_acoustic_aligned, probs_perceptual)]
    
    # Convert to numpy arrays
    js_divs = np.array(js_divs)
    cos_sims = np.array(cos_sims)
    bhatt_coeffs = np.array(bhatt_coeffs)
    
    # Print summary statistics
    print(f"Mean JS divergence: {np.mean(js_divs):.4f}")
    print(f"Mean cosine similarity: {np.mean(cos_sims):.4f}")
    print(f"Mean Bhattacharyya coefficient: {np.mean(bhatt_coeffs):.4f}")
    print(f"Std Bhattacharyya coefficient: {np.std(bhatt_coeffs):.4f}")
    
    return js_divs, cos_sims, bhatt_coeffs


def analyze_cluster_features(all_clusters: Dict, all_feature_maps: Dict, 
                           catted: pd.DataFrame) -> Dict:
    """
    Analyze and map features to clusters for each modality.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        all_feature_maps: Dictionary of feature mappings
        catted: Combined cluster assignments dataframe
        
    Returns:
        Dictionary of cluster feature mappings
    """
    cluster_feature_mappings = {}
    
    for mode in ["perceptual", "acoustic"]:
        print(f"\\n--- {mode.upper()} MODE ---")
        clusters = all_clusters[mode]
        features_df = all_feature_maps[mode]
        
        cluster_feature_mapping = {}
        for ct in np.unique(clusters[0]):
            # Get participants in this cluster
            cluster_participants = catted.index[catted[mode] == ct]
            
            # Get features associated with this cluster
            cluster_features = list(features_df.index[features_df.values.ravel() == ct])
            cluster_feature_mapping[ct] = cluster_features
            
            # Clean perceptual feature names
            if mode == "perceptual":
                cleaned_features = []
                for feature in cluster_features:
                    # Clean feature names by removing parenthetical content and extra formatting
                    cleaned = feature.replace("  ", "(").replace(":", "(")
                    if ". " in cleaned:
                        cleaned = cleaned.split(". ")[1]
                    if "(" in cleaned:
                        cleaned = cleaned.split("(")[0]
                    cleaned = cleaned.replace(" ", "")
                    cleaned_features.append(cleaned)
                cluster_feature_mapping[ct] = cleaned_features
        
        cluster_feature_mappings[mode] = cluster_feature_mapping
        
        print(f"Cluster feature mapping for {mode}:")
        for cluster_id, features in cluster_feature_mapping.items():
            print(f"  Cluster {cluster_id}: {features}")
    
    return cluster_feature_mappings


def analyze_cluster_severity(all_clusters: Dict, clinical_summaries: pd.DataFrame, 
                           catted: pd.DataFrame) -> Dict:
    """
    Analyze severity statistics per cluster.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        clinical_summaries: Clinical summary scores
        catted: Combined cluster assignments dataframe
        
    Returns:
        Dictionary of severity statistics per cluster
    """
    severity_stats = {}
    
    for mode in ["perceptual", "acoustic"]:
        clusters = all_clusters[mode]
        mode_stats = {}
        
        print(f"\\nSeverity analysis for {mode.upper()} clusters:")
        for cluster_idx in np.unique(clusters[0]):
            cluster_severities = clinical_summaries.loc[
                :, "ALSIBD Total Score (calculated)"
            ].loc[catted[mode] == cluster_idx]
            
            stats = {
                'median': np.median(cluster_severities.values),
                'q1': np.quantile(cluster_severities.values, 0.25),
                'q3': np.quantile(cluster_severities.values, 0.75),
                'count': len(cluster_severities)
            }
            
            mode_stats[cluster_idx] = stats
            
            print(f"  Cluster {cluster_idx}:")
            print(f"    Count: {stats['count']}")
            print(f"    Median: {stats['median']:.2f}")
            print(f"    Q1: {stats['q1']:.2f}")
            print(f"    Q3: {stats['q3']:.2f}")
        
        severity_stats[mode] = mode_stats
    
    return severity_stats


def run_cross_modal_analysis(all_clusters: Dict, all_feature_maps: Dict, 
                            coef_dists: Dict, clinical_summaries: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Run complete cross-modal analysis between acoustic and perceptual clusters.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        all_feature_maps: Dictionary of feature mappings
        coef_dists: Dictionary of coefficient distributions
        clinical_summaries: Clinical summary scores
        
    Returns:
        Tuple of (analysis_results, similarity_metrics)
    """
    print("\\n" + "="*60)
    print("CROSS-MODAL ANALYSIS")
    print("="*60)
    
    # Create combined cluster dataframe
    perceptual_clusters_df = pd.DataFrame(
        data=all_clusters["perceptual"][0], 
        index=clinical_summaries.index[:len(all_clusters["perceptual"][0])]
    )
    acoustic_clusters_df = pd.DataFrame(
        data=all_clusters["acoustic"][0], 
        index=clinical_summaries.index[:len(all_clusters["acoustic"][0])]
    )
    
    catted = pd.concat([perceptual_clusters_df, acoustic_clusters_df], axis=1)
    catted.columns = ['perceptual', "acoustic"]
    
    # Align clusters and compute similarities
    probs_acoustic_aligned, probs_perceptual, mapping = align_clusters(coef_dists)
    js_divs, cos_sims, bhatt_coeffs = compute_similarity_metrics(
        probs_acoustic_aligned, probs_perceptual
    )
    
    # Analyze cluster features and severity
    cluster_mappings = analyze_cluster_features(all_clusters, all_feature_maps, catted)
    severity_stats = analyze_cluster_severity(all_clusters, clinical_summaries, catted)
    
    # Compile results
    analysis_results = {
        'cluster_mappings': cluster_mappings,
        'severity_stats': severity_stats,
        'acoustic_to_perceptual_mapping': mapping,
        'combined_clusters': catted
    }
    
    similarity_metrics = {
        'js_divergences': js_divs,
        'cosine_similarities': cos_sims,
        'bhattacharyya_coefficients': bhatt_coeffs
    }
    
    return analysis_results, similarity_metrics
