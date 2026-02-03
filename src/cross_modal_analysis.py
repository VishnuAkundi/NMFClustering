"""
Cross-modal analysis module for comparing acoustic and perceptual clusters.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, Any
from .config import *
from . import config


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


def align_clusters(coef_dists: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    Align acoustic and perceptual clusters using optimal assignment.
    
    Args:
        coef_dists: Dictionary containing coefficient distributions
        
    Returns:
        Tuple of (aligned_acoustic_probs, perceptual_probs, mapping_dict, js_divergence_matrix)
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
    
    print(f"JS divergence matrix range: [{cost_matrix.min():.4f}, {cost_matrix.max():.4f}]")
    
    # Solve optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create label mapping
    acoustic_to_perceptual = dict(zip(row_ind, col_ind))
    
    # Permute acoustic probabilities to match perceptual clusters
    perm = [k for k, v in sorted(acoustic_to_perceptual.items(), key=lambda x: x[1])]
    probs_acoustic_aligned = probs_acoustic[:, perm]
    
    print(f"Cluster alignment mapping: {acoustic_to_perceptual}")
    print(f"Hungarian assignment cost: {cost_matrix[row_ind, col_ind].sum():.4f}")
    
    return probs_acoustic_aligned, probs_perceptual, acoustic_to_perceptual, cost_matrix


def align_clusters_column_similarity(coef_dists: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, np.ndarray]:
    """
    Align acoustic and perceptual clusters using column similarity method.
    
    This method compares the coefficient matrices (W_A, W_B) directly by:
    1. Row-normalizing coefficient matrices so each patient's memberships sum to 1
    2. Computing similarity matrix C where C[i,j] = similarity between A's component i and B's component j
    3. Using Hungarian algorithm for optimal 1-to-1 mapping when kA == kB
    
    Args:
        coef_dists: Dictionary containing coefficient distributions from NMF
        
    Returns:
        Tuple of (aligned_acoustic_probs, perceptual_probs, mapping_dict, similarity_matrix)
    """
    print("Aligning acoustic and perceptual clusters using column similarity method...")
    
    # Get coefficient matrices (these are the W matrices from NMF)
    # coef_dists contains the coefficient distributions, shape: (n_components, n_samples)
    W_A = np.asarray(coef_dists["acoustic"].T)  # Shape: (n_samples, kA)
    W_B = np.asarray(coef_dists["perceptual"].T)  # Shape: (n_samples, kB)
    
    print(f"Acoustic coefficient matrix (W_A) shape: {W_A.shape}")
    print(f"Perceptual coefficient matrix (W_B) shape: {W_B.shape}")
    
    # Step 1: Row-normalize so each patient's memberships sum to 1
    W_A_sums = np.asarray(W_A.sum(axis=1)).reshape(-1, 1) + 1e-10  # Add small epsilon to avoid division by zero
    W_B_sums = np.asarray(W_B.sum(axis=1)).reshape(-1, 1) + 1e-10
    W_A_norm = np.asarray(W_A) / W_A_sums
    W_B_norm = np.asarray(W_B) / W_B_sums
    
    print(f"Row-normalized W_A range: [{W_A_norm.min():.4f}, {W_A_norm.max():.4f}]")
    print(f"Row-normalized W_B range: [{W_B_norm.min():.4f}, {W_B_norm.max():.4f}]")
    
    # Save the row-normalized coefficient matrices used for similarity computation
    # Save normalized acoustic coefficient matrix
    W_A_norm_path = f"{config.RESULTS_DIR}/nmf_coefficient_matrix_normalized_acoustic.csv"
    W_A_norm_df = pd.DataFrame(
        W_A_norm.T,  # Transpose to get components x samples format
        index=[f'Component_{i}' for i in range(W_A_norm.shape[1])],
        columns=[f'Sample_{i}' for i in range(W_A_norm.shape[0])]
    )
    W_A_norm_df.to_csv(W_A_norm_path)
    print(f"Row-normalized acoustic coefficient matrix saved to: {W_A_norm_path}")
    
    # Save normalized perceptual coefficient matrix  
    W_B_norm_path = f"{config.RESULTS_DIR}/nmf_coefficient_matrix_normalized_perceptual.csv"
    W_B_norm_df = pd.DataFrame(
        W_B_norm.T,  # Transpose to get components x samples format
        index=[f'Component_{i}' for i in range(W_B_norm.shape[1])],
        columns=[f'Sample_{i}' for i in range(W_B_norm.shape[0])]
    )
    W_B_norm_df.to_csv(W_B_norm_path)
    print(f"Row-normalized perceptual coefficient matrix saved to: {W_B_norm_path}")
    
    # Step 2: Compute similarity matrix C
    # C[i,j] = similarity between A's component i and B's component j, computed over shared patients
    kA = W_A_norm.shape[1]
    kB = W_B_norm.shape[1]
    
    similarity_matrix = np.zeros((kA, kB))
    
    for i in range(kA):
        for j in range(kB):
            # Get component vectors across all patients
            vec_A = W_A_norm[:, i]  # A's component i across all patients
            vec_B = W_B_norm[:, j]  # B's component j across all patients
            
            # Compute similarity based on selected metric
            if config.COLUMN_SIMILARITY_METRIC == "cosine":
                # Cosine similarity
                similarity_matrix[i, j] = cosine_similarity(
                    vec_A.reshape(1, -1), vec_B.reshape(1, -1)
                )[0, 0]
            elif config.COLUMN_SIMILARITY_METRIC == "correlation":
                # Pearson correlation
                similarity_matrix[i, j] = np.corrcoef(vec_A, vec_B)[0, 1]
                # Handle NaN values (can occur with constant vectors)
                if np.isnan(similarity_matrix[i, j]):
                    similarity_matrix[i, j] = 0.0
            else:
                raise ValueError(f"Unknown similarity metric: {config.COLUMN_SIMILARITY_METRIC}")
    
    print(f"Similarity matrix C shape: {similarity_matrix.shape}")
    print(f"Similarity matrix range: [{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]")
    
    # Step 3: Apply Hungarian algorithm if kA == kB
    mapping_dict = {}
    aligned_acoustic_probs = W_A_norm.copy()
    
    if kA == kB:
        print(f"kA == kB ({kA}), applying Hungarian algorithm for optimal 1-to-1 mapping...")
        
        # Hungarian algorithm minimizes cost, so we use -similarity_matrix as cost
        # For correlation, use absolute values for optimization (strong negative correlations are also good matches)
        if config.COLUMN_SIMILARITY_METRIC == "correlation":
            cost_matrix = -np.abs(similarity_matrix)  # Use absolute values for optimization
            print("Using absolute correlation values for Hungarian optimization")
        else:
            cost_matrix = -similarity_matrix  # For cosine similarity, use values directly
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create mapping dictionary
        mapping_dict = dict(zip(row_ind, col_ind))
        
        # Reorder acoustic probabilities to match perceptual clusters
        perm = [mapping_dict[i] for i in range(kA)]
        aligned_acoustic_probs = W_A_norm[:, perm]
        
        print(f"Optimal alignment mapping: {mapping_dict}")
        print(f"Total alignment cost (negative similarity): {cost_matrix[row_ind, col_ind].sum():.4f}")
        print(f"Total alignment similarity: {similarity_matrix[row_ind, col_ind].sum():.4f}")
        
    else:
        print(f"kA != kB ({kA} != {kB}), no 1-to-1 mapping possible. Similarity matrix saved for inspection.")
        # Create a simple mapping for consistency (maps each acoustic cluster to most similar perceptual)
        for i in range(kA):
            if config.COLUMN_SIMILARITY_METRIC == "correlation":
                # For correlation, use absolute values to find best match
                best_match = np.argmax(np.abs(similarity_matrix[i, :]))
            else:
                # For cosine similarity, use values directly
                best_match = np.argmax(similarity_matrix[i, :])
            mapping_dict[i] = best_match
        print(f"Best match mapping (not optimal): {mapping_dict}")
    
    return aligned_acoustic_probs, W_B_norm, mapping_dict, similarity_matrix


def visualize_similarity_matrix(similarity_matrix: np.ndarray, mapping_dict: Dict, 
                              similarity_metric: str, save_path: str) -> None:
    """
    Visualize the similarity matrix as a heatmap.
    
    Args:
        similarity_matrix: kA x kB similarity matrix where C[i,j] = similarity between 
                          A's component i and B's component j
        mapping_dict: Mapping from acoustic to perceptual clusters
        similarity_metric: Type of similarity metric used ("cosine" or "correlation")
        save_path: Path to save the heatmap figure
    """
    print(f"Creating similarity matrix heatmap...")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(
        similarity_matrix,
        annot=True,
        fmt='.3f',
        cmap='viridis',
        center=0 if similarity_metric == "correlation" else 0.5,
        cbar_kws={'label': f'{similarity_metric.capitalize()} Similarity'}    
    )
    # Set color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label(f'{similarity_metric.capitalize()} Similarity', fontsize=20, fontweight='bold')
    # Set color bar tick label font size
    cbar.ax.tick_params(labelsize=18)
    # Set annotation font size
    for t in ax.texts:
        t.set_fontsize(30)
    
    # Customize labels
    kA, kB = similarity_matrix.shape
    ax.set_xlabel('Perceptual Components', fontsize=20, fontweight='bold')
    ax.set_ylabel('Acoustic Components', fontsize=20, fontweight='bold')
    ax.set_title(f'Cross-Modal Component Similarity Matrix\n({similarity_metric.capitalize()} similarity)', 
                fontsize=20, fontweight='bold')
    
    # Set tick labels
    ax.set_xticklabels([f'P{j}' for j in range(kB)], rotation=0, fontsize=20)
    ax.set_yticklabels([f'A{i}' for i in range(kA)], rotation=0, fontsize=20)

    # Highlight optimal mappings if available
    if mapping_dict and len(mapping_dict) == kA and len(set(mapping_dict.values())) == len(mapping_dict.values()):
        # Only highlight if we have a proper 1-to-1 mapping
        for acoustic_idx, perceptual_idx in mapping_dict.items():
            # Add a colored border around optimal mappings
            ax.add_patch(plt.Rectangle((perceptual_idx, acoustic_idx), 1, 1, 
                                     fill=False, edgecolor='red', lw=3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Similarity matrix heatmap saved to: {save_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


def visualize_js_divergence_matrix(js_divergence_matrix: np.ndarray, mapping_dict: Dict, save_path: str) -> None:
    """
    Visualize the JS divergence matrix as a heatmap.
    
    Args:
        js_divergence_matrix: kA x kB JS divergence matrix where D[i,j] = JS divergence between 
                             A's component i and B's component j
        mapping_dict: Mapping from acoustic to perceptual clusters
        save_path: Path to save the heatmap figure
    """
    print(f"Creating JS divergence matrix heatmap...")
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap (reverse colormap since lower divergence is better)
    ax = sns.heatmap(
        js_divergence_matrix,
        annot=True,
        fmt='.4f',
        cmap='viridis_r',  # Reversed colormap: darker = lower divergence (better match)
        cbar_kws={'label': 'Jensen-Shannon Divergence'}
    )
    
    # Customize labels
    kA, kB = js_divergence_matrix.shape
    ax.set_xlabel('Perceptual Components', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acoustic Components', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Modal Component JS Divergence Matrix\n(Lower values = better match)', 
                fontsize=14, fontweight='bold')
    
    # Set tick labels
    ax.set_xticklabels([f'P{j}' for j in range(kB)], rotation=0)
    ax.set_yticklabels([f'A{i}' for i in range(kA)], rotation=0)
    
    # Highlight optimal mappings if available
    if mapping_dict and len(mapping_dict) == kA and len(set(mapping_dict.values())) == len(mapping_dict.values()):
        # Only highlight if we have a proper 1-to-1 mapping
        for acoustic_idx, perceptual_idx in mapping_dict.items():
            # Add a colored border around optimal mappings
            ax.add_patch(plt.Rectangle((perceptual_idx, acoustic_idx), 1, 1, 
                                     fill=False, edgecolor='red', lw=3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"JS divergence matrix heatmap saved to: {save_path}")
    
    if config.SHOW_PLOTS:
        plt.show()
    plt.close()


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
                            coef_dists: Dict, clinical_summaries: pd.DataFrame,
                            raw_coef_dists: Dict = None) -> Tuple[Dict, Dict]:
    """
    Run complete cross-modal analysis between acoustic and perceptual clusters.
    
    Args:
        all_clusters: Dictionary of cluster assignments
        all_feature_maps: Dictionary of feature mappings
        coef_dists: Dictionary of coefficient distributions (softmax normalized)
        clinical_summaries: Clinical summary scores
        raw_coef_dists: Dictionary of raw coefficient matrices (optional, used for column similarity)
        
    Returns:
        Tuple of (analysis_results, similarity_metrics)
    """
    print("\n" + "="*60)
    print("CROSS-MODAL ANALYSIS")
    print("="*60)
    print(f"Using alignment method: {config.ALIGNMENT_METHOD}")
    
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
    
    # Choose alignment method based on configuration
    similarity_matrix = None
    
    if config.ALIGNMENT_METHOD == "column_similarity":
        print(f"Using column similarity method with {config.COLUMN_SIMILARITY_METRIC} metric")
        
        # Choose coefficient matrices based on availability of raw coefficients
        if raw_coef_dists is not None:
            # Use raw coefficients to bypass softmax normalization
            coeffs_for_alignment = raw_coef_dists
            print("Using raw coefficient matrices (bypassing softmax) for column similarity alignment")
        else:
            # Fall back to softmax-normalized coefficients
            coeffs_for_alignment = coef_dists
            print("Warning: Raw coefficients not available, using softmax-normalized coefficients")
            
        probs_acoustic_aligned, probs_perceptual, mapping, similarity_matrix = align_clusters_column_similarity(coeffs_for_alignment)
        
        # Save similarity matrix to results
        similarity_matrix_path = f"{config.RESULTS_DIR}/similarity_matrix_{config.COLUMN_SIMILARITY_METRIC}.csv"
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=[f'Acoustic_{i}' for i in range(similarity_matrix.shape[0])],
            columns=[f'Perceptual_{j}' for j in range(similarity_matrix.shape[1])]
        )
        similarity_df.to_csv(similarity_matrix_path)
        print(f"Similarity matrix saved to: {similarity_matrix_path}")
        
        # Save mapping to results
        mapping_path = f"{config.RESULTS_DIR}/cluster_mapping_{config.COLUMN_SIMILARITY_METRIC}.csv"
        mapping_df = pd.DataFrame([
            {'Acoustic_Cluster': k, 'Perceptual_Cluster': v, 'Similarity': similarity_matrix[k, v]}
            for k, v in mapping.items()
        ])
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Cluster mapping saved to: {mapping_path}")
        
        # Create similarity matrix heatmap
        heatmap_path = f"{config.PLOTS_DIR}/similarity_matrix_heatmap_{config.COLUMN_SIMILARITY_METRIC}.png"
        visualize_similarity_matrix(similarity_matrix, mapping, config.COLUMN_SIMILARITY_METRIC, heatmap_path)
        
    else:
        # Default: Use existing JSD + Hungarian alignment method
        print("Using default JSD + Hungarian alignment method")
        probs_acoustic_aligned, probs_perceptual, mapping, js_divergence_matrix = align_clusters(coef_dists)
        
        # Convert JS divergence to similarity (similarity = 1 - divergence)
        similarity_matrix = 1 - js_divergence_matrix
        
        # Save JS divergence matrix to results
        divergence_matrix_path = f"{config.RESULTS_DIR}/js_divergence_matrix_hungarian.csv"
        divergence_df = pd.DataFrame(
            js_divergence_matrix,
            index=[f'Acoustic_{i}' for i in range(js_divergence_matrix.shape[0])],
            columns=[f'Perceptual_{j}' for j in range(js_divergence_matrix.shape[1])]
        )
        divergence_df.to_csv(divergence_matrix_path)
        print(f"JS divergence matrix saved to: {divergence_matrix_path}")
        
        # Save similarity matrix (1 - divergence) to results
        similarity_matrix_path = f"{config.RESULTS_DIR}/similarity_matrix_hungarian.csv"
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=[f'Acoustic_{i}' for i in range(similarity_matrix.shape[0])],
            columns=[f'Perceptual_{j}' for j in range(similarity_matrix.shape[1])]
        )
        similarity_df.to_csv(similarity_matrix_path)
        print(f"Similarity matrix saved to: {similarity_matrix_path}")
        
        # Save mapping to results
        mapping_path = f"{config.RESULTS_DIR}/cluster_mapping_hungarian.csv"
        mapping_df = pd.DataFrame([
            {'Acoustic_Cluster': k, 'Perceptual_Cluster': v, 'JS_Divergence': js_divergence_matrix[k, v], 'Similarity': similarity_matrix[k, v]}
            for k, v in mapping.items()
        ])
        mapping_df.to_csv(mapping_path, index=False)
        print(f"Cluster mapping saved to: {mapping_path}")
        
        # Create similarity matrix heatmap
        heatmap_path = f"{config.PLOTS_DIR}/similarity_matrix_heatmap_hungarian.png"
        visualize_similarity_matrix(similarity_matrix, mapping, "hungarian_similarity", heatmap_path)
        
        # Create JS divergence matrix heatmap
        divergence_heatmap_path = f"{config.PLOTS_DIR}/js_divergence_matrix_heatmap_hungarian.png"
        visualize_js_divergence_matrix(js_divergence_matrix, mapping, divergence_heatmap_path)
    
    # Compute similarity metrics between aligned distributions
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
        'combined_clusters': catted,
        'alignment_method': config.ALIGNMENT_METHOD
    }
    
    # Add similarity matrix to results if using column similarity method
    if similarity_matrix is not None:
        analysis_results['similarity_matrix'] = similarity_matrix
        analysis_results['similarity_metric'] = config.COLUMN_SIMILARITY_METRIC
    
    similarity_metrics = {
        'js_divergences': js_divs,
        'cosine_similarities': cos_sims,
        'bhattacharyya_coefficients': bhatt_coeffs
    }
    
    return analysis_results, similarity_metrics
