"""
Visualization module for clustering analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from collections import Counter
from typing import Dict, Tuple, Any
from .config import *


def setup_output_directories():
    """Create output directories if they don't exist."""
    import os
    for directory in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)


def plot_kde(V: np.ndarray, ax: plt.Axes, colour: str = 'k', 
             label: str = None, limits: float = 2) -> plt.Axes:
    """
    Plot kernel density estimation contours on given axis.
    
    Args:
        V: matrix where columns = variables and rows = samples
        ax: axis to attach the plot to
        colour: color for the contour lines
        label: label for the plot
        limits: limits for the plot area
        
    Returns:
        Modified axes object
    """
    kernel = gaussian_kde(V.T)
    xmin, ymin, xmax, ymax = np.r_[V.min(axis=0) - limits, V.max(axis=0) + limits]
    
    # Perform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    
    ax.contour(xx, yy, f, colors=colour, levels=1, linewidths=2, label=label)
    return ax


def plot_pca_clusters(V: np.ndarray, clusters: np.ndarray, data_type: str, 
                     save_path: str = None) -> None:
    """
    Create PCA visualization of clusters.
    
    Args:
        V: Scaled data matrix
        clusters: Cluster assignments
        data_type: Type of data ("acoustic" or "perceptual")
        save_path: Path to save the plot
    """
    # Create color mapping
    vectorized_func = np.vectorize(lambda x: CLUSTER_COLORS[x])
    string_array = vectorized_func(clusters)
    
    # Plot the clustering with PCA visualization
    fig, ax = plt.subplots(1, 1, figsize=PCA_FIGURE_SIZE)
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(V)
    
    ax.axvline(0, c="grey", ls=":")
    ax.axhline(0, c="grey", ls=":")
    ax.scatter(pca_out[:, 0], pca_out[:, 1], c=string_array[0])
    
    # Add KDE contours for each cluster
    lim = 0.5  # Visual scale of KDE plot
    for c in np.unique(clusters[0]):
        plot_kde(pca_out[clusters[0] == c, :], ax=ax, colour=CLUSTER_COLORS[c], 
                label=None, limits=lim)
    
    ax.set_xlabel('PCA component 1 (visualization only)')
    ax.set_ylabel('PCA component 2 (visualization only)')
    ax.set_title(f'Cluster Visualization - {data_type.capitalize()} Data (PCA with NMF coloring)')
    
    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=CLUSTER_COLORS[c], markersize=10, 
                         label=f"Cluster {c}") for c in np.unique(clusters[0])]
    ax.legend(handles=handles, loc="upper right")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved: {save_path}")
    
    plt.show()


def plot_cluster_distributions(clusters: np.ndarray, clinical_summaries: pd.DataFrame, 
                             data_type: str, save_dir: str = None) -> None:
    """
    Create bar plots showing cluster size distributions and severity ratios.
    
    Args:
        clusters: Cluster assignments
        clinical_summaries: Clinical summary scores
        data_type: Type of data
        save_dir: Directory to save plots
    """
    from collections import Counter
    
    # Compute overall cluster counts
    counts = Counter(clusters[0])
    categories = list(counts.keys())
    values = list(counts.values())
    
    # Compute counts of severe participants per cluster
    severe_mask = clinical_summaries["ALSIBD Total Score (calculated)"] > ALSIBD_THRESHOLD
    severe_clusters = clusters[0][severe_mask]
    counts_severe = Counter(severe_clusters)
    categories_severe = list(counts_severe.keys())
    values_severe = list(counts_severe.values())
    
    # Bar plot 1: Cluster size distribution
    plt.figure(figsize=BAR_FIGURE_SIZE)
    plt.bar(categories, np.array(values) / len(clusters[0]), color='dodgerblue')
    plt.xlabel("Cluster ID")
    plt.ylabel("Proportion of Participants")
    plt.title(f"Distribution of Participants Across Clusters - {data_type.capitalize()}")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/cluster_distribution_{data_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster distribution plot saved: {save_path}")
    
    plt.show()
    
    # Bar plot 2: Severe participants per cluster (global proportion)
    plt.figure(figsize=BAR_FIGURE_SIZE)
    plt.bar(categories_severe, np.array(values_severe) / len(clusters[0]), color='firebrick')
    plt.xlabel("Cluster ID")
    plt.ylabel("Proportion of All Participants")
    plt.title(f"Severe Participants per Cluster - {data_type.capitalize()} (Global Proportion)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    if save_dir:
        save_path = f"{save_dir}/severe_global_{data_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Severe participants plot saved: {save_path}")
    
    plt.show()
    
    # Bar plot 3: Severity ratio within each cluster
    if categories_severe:  # Only plot if there are severe participants
        within_cluster_ratios = np.array(values_severe) / np.array([counts[c] for c in categories_severe])
        plt.figure(figsize=BAR_FIGURE_SIZE)
        plt.bar(categories_severe, within_cluster_ratios, color='orange')
        plt.xlabel("Cluster ID")
        plt.ylabel("Proportion of Cluster That Is Severe")
        plt.title(f"Severity Percentage Within Each Cluster - {data_type.capitalize()}")
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()
        
        if save_dir:
            save_path = f"{save_dir}/severity_within_cluster_{data_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Within-cluster severity plot saved: {save_path}")
        
        plt.show()


def plot_feature_importance(basis_dists: Dict, all_data: pd.DataFrame, 
                          first_acoustic: int, save_dir: str = None) -> None:
    """
    Create feature importance plots for each cluster and data type.
    
    Args:
        basis_dists: Dictionary of basis matrices
        all_data: Combined dataset
        first_acoustic: Index of first acoustic feature
        save_dir: Directory to save plots
    """
    for data_type in ["acoustic", "perceptual"]:
        basis = basis_dists[data_type]
        
        if data_type == "acoustic":
            columns = all_data.columns[first_acoustic:]
        elif data_type == "perceptual":
            columns = all_data.columns[:first_acoustic]
        
        # Create DataFrame with proper indexing
        basis_df = pd.DataFrame(basis, index=columns)
        color_palette = plt.cm.get_cmap('tab20', basis_df.shape[0])
        
        # Create bar plots for each cluster
        for cluster_idx in range(basis_df.shape[1]):
            plt.figure(figsize=FEATURE_PLOT_SIZE)
            
            # Sort features by importance for current cluster
            all_features = basis_df.iloc[:, cluster_idx].sort_values(ascending=False)
            feature_names = all_features.index
            
            # Clean perceptual feature names by removing parenthetical content
            if data_type == "perceptual":
                feature_names = [re.sub(r"\\(.*\\)", "", feat).strip() for feat in all_features.index]
            
            # Plot
            plt.bar(feature_names, all_features.values, 
                   color=[color_palette(i) for i in range(len(all_features))])
            plt.title(f"Feature Importances - Cluster {cluster_idx} ({data_type.capitalize()} Data)")
            plt.xlabel("Feature Names")
            plt.ylabel("Importance (Weight)")
            plt.xticks(rotation=90, ha='right')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            if save_dir:
                save_path = f"{save_dir}/feature_importance_{data_type}_cluster_{cluster_idx}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Feature importance plot saved: {save_path}")
            
            plt.show()


def plot_confusion_matrix(perceptual_clusters_df: pd.DataFrame, 
                        acoustic_clusters_df: pd.DataFrame, 
                        save_path: str = None) -> float:
    """
    Create confusion matrix comparing acoustic and perceptual clusters.
    
    Args:
        perceptual_clusters_df: Perceptual cluster assignments
        acoustic_clusters_df: Acoustic cluster assignments
        save_path: Path to save the plot
        
    Returns:
        Cramer's V coefficient
    """
    from sklearn.metrics import confusion_matrix
    from scipy.stats.contingency import association
    
    # Combine cluster assignments
    catted = pd.concat([perceptual_clusters_df, acoustic_clusters_df], axis=1)
    catted.columns = ['perceptual', "acoustic"]
    
    # Create confusion matrix
    observed = confusion_matrix(catted['acoustic'], catted['perceptual'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(observed, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=np.unique(catted['perceptual']),
                yticklabels=np.unique(catted['acoustic']))
    plt.xlabel("Perceptual Clusters")
    plt.ylabel("Acoustic Clusters")
    plt.title("Confusion Matrix: Acoustic vs Perceptual Clusters")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved: {save_path}")
    
    plt.show()
    
    # Calculate Cramer's V for association strength
    cramers_v = association(observed[:4, :], method="cramer")
    print(f"Cramer's V: {cramers_v:.4f}")
    
    # Calculate accuracy
    accuracy = np.trace(observed) / np.sum(observed)
    print(f"Accuracy (trace/sum): {accuracy:.4f}")
    print(f"Total observations: {np.sum(observed)}")
    
    return cramers_v


def plot_similarity_distributions(js_divs: np.ndarray, cos_sims: np.ndarray, 
                                bhatt_coeffs: np.ndarray, save_path: str = None) -> None:
    """
    Create histograms of similarity metrics between acoustic and perceptual clusters.
    
    Args:
        js_divs: Jensen-Shannon divergences
        cos_sims: Cosine similarities  
        bhatt_coeffs: Bhattacharyya coefficients
        save_path: Path to save the plot
    """
    plt.figure(figsize=SIMILARITY_PLOT_SIZE)
    
    plt.subplot(1, 3, 1)
    plt.hist(js_divs.flatten(), bins=50, color='skyblue', edgecolor='black')
    plt.title("JS Divergence per Patient")
    plt.xlabel("Jensen-Shannon Divergence")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 2)
    plt.hist(cos_sims.flatten(), bins=50, color='lightgreen', edgecolor='black')
    plt.title("Cosine Similarity per Patient")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    
    plt.subplot(1, 3, 3)
    plt.hist(bhatt_coeffs, bins=50, color='salmon', edgecolor='black')
    plt.title("Bhattacharyya Similarity per Patient")
    plt.xlabel("Bhattacharyya Coefficient")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Similarity distributions plot saved: {save_path}")
    
    plt.show()


def visualize_cluster_features(cluster_idx: int, feature_names: list, means: np.ndarray, 
                             stds: np.ndarray, medians: np.ndarray, q1: np.ndarray, 
                             q3: np.ndarray, scores: np.ndarray, data_type: str, 
                             save_path: str = None) -> None:
    """
    Create detailed feature analysis plots for a specific cluster.
    
    Args:
        cluster_idx: Cluster index
        feature_names: Names of features
        means: Mean values
        stds: Standard deviations
        medians: Median values
        q1: First quartile values
        q3: Third quartile values
        scores: Importance scores
        data_type: Type of data
        save_path: Path to save the plot
    """
    # Clean perceptual feature names
    if data_type == "perceptual":
        feature_names = [re.sub(r"\\(.*\\)", "", feat).strip() for feat in feature_names]
    
    fig, axes = plt.subplots(2, 1, figsize=CLUSTER_FEATURE_SIZE)
    fig.suptitle(f"Feature Analysis - Cluster {cluster_idx} ({data_type.capitalize()} Data)", 
                fontsize=16)
    
    # Plot statistical summary
    axes[0].bar(feature_names, means, yerr=stds, capsize=5, label='Mean ± Std')
    axes[0].scatter(feature_names, medians, color='orange', label='Median', s=50)
    axes[0].scatter(feature_names, q1, color='green', marker='x', label='Q1', s=50)
    axes[0].scatter(feature_names, q3, color='red', marker='x', label='Q3', s=50)
    axes[0].set_ylabel("Feature Values")
    axes[0].legend()
    axes[0].set_title("Statistical Summary of Features")
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot importance scores
    axes[1].bar(feature_names, scores, color='skyblue')
    axes[1].set_ylabel("Importance Scores")
    axes[1].set_title("Feature Importance Scores")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster feature analysis saved: {save_path}")
    
    plt.show()
