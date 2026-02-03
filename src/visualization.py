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
    """Create output directories, with versioning if enabled."""
    import os
    from . import config
    
    if USE_VERSIONED_OUTPUTS:
        # Find the next available run number
        run_number = 1
        while os.path.exists(f"{OUTPUT_BASE_DIR}/run{run_number}"):
            run_number += 1
        
        # Update config paths to use versioned directory
        config.OUTPUT_DIR = f"{OUTPUT_BASE_DIR}/run{run_number}"
        config.PLOTS_DIR = f"{config.OUTPUT_DIR}/plots"
        config.TABLES_DIR = f"{config.OUTPUT_DIR}/tables"
        config.RESULTS_DIR = f"{config.OUTPUT_DIR}/results"
        
        print(f"📁 Creating versioned output directory: run{run_number}")
    
    # Create all directories
    for directory in [config.OUTPUT_DIR, config.PLOTS_DIR, config.TABLES_DIR, config.RESULTS_DIR]:
        os.makedirs(directory, exist_ok=True)
        
    print(f"📊 Output directories ready:")
    print(f"   Plots: {config.PLOTS_DIR}")
    print(f"   Tables: {config.TABLES_DIR}")
    print(f"   Results: {config.RESULTS_DIR}")


def show_plot_if_enabled():
    """Show plot only if SHOW_PLOTS is enabled in config."""
    if SHOW_PLOTS:
        plt.show()


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
                     save_path: str = None, original_data: np.ndarray = None, 
                     zero_participants: pd.Index = None) -> None:
    """
    Create PCA visualization of clusters.
    
    Args:
        V: Scaled data matrix
        clusters: Cluster assignments
        data_type: Type of data ("acoustic" or "perceptual")
        save_path: Path to save the plot
        original_data: Original unscaled data matrix (for detecting all-zero perceptual data)
        zero_participants: Either pd.Index of participant IDs or boolean mask for zero participants
    """
    # Create color mapping
    vectorized_func = np.vectorize(lambda x: CLUSTER_COLORS[x])
    string_array = vectorized_func(clusters)
    
    # Detect all-zero perceptual data if enabled and data_type is perceptual
    zero_data_mask = None
    if (data_type == "perceptual" and HIGHLIGHT_ZERO_PERCEPTUAL):
        if zero_participants is not None:
            # Check if zero_participants is a boolean mask or an index
            if isinstance(zero_participants, (pd.Index, list, np.ndarray)) and len(zero_participants) > 0:
                if isinstance(zero_participants, pd.Series) or hasattr(zero_participants, 'dtype'):
                    # It's a boolean mask
                    if zero_participants.dtype == bool:
                        zero_data_mask = zero_participants.values if hasattr(zero_participants, 'values') else zero_participants
                        print(f"Using boolean mask: {zero_data_mask.sum()} zero participants identified")
                    else:
                        # It's an index, convert to boolean mask by checking original data
                        if original_data is not None:
                            epsilon_threshold = 1e-9
                            row_sums = np.sum(original_data, axis=1)
                            zero_data_mask = row_sums < epsilon_threshold
                            print(f"Converted index to mask: {zero_data_mask.sum()} zero participants detected")
                else:
                    # It's an index or list, convert to boolean mask
                    if original_data is not None:
                        epsilon_threshold = 1e-9
                        row_sums = np.sum(original_data, axis=1)
                        zero_data_mask = row_sums < epsilon_threshold
                        print(f"Using index with {len(zero_participants)} participants, detected {zero_data_mask.sum()} zero rows")
            elif hasattr(zero_participants, 'dtype') and zero_participants.dtype == bool:
                # It's already a boolean mask
                zero_data_mask = zero_participants
                print(f"Using provided boolean mask: {zero_data_mask.sum()} zero participants")
            else:
                print(f"DEBUG: zero_participants type: {type(zero_participants)}, length: {len(zero_participants) if hasattr(zero_participants, '__len__') else 'N/A'}")
                
        elif original_data is not None:
            # Fallback: Identify rows that are all zeros (or effectively zero) from original data
            zero_threshold = 1e-9  # Relaxed threshold to catch near-zero values after preprocessing
            row_sums = np.sum(original_data, axis=1)
            zero_data_mask = row_sums < zero_threshold
            
            if zero_data_mask.sum() > 0:
                print(f"Highlighting {zero_data_mask.sum()} participants with all-zero perceptual data (auto-detected)")
        else:
            print(f"DEBUG: No zero participants info or original data provided for perceptual highlighting")
    else:
        pass  # Not checking for zero data for acoustic or when highlighting is disabled
    
    # Plot the clustering with PCA visualization
    fig, ax = plt.subplots(1, 1, figsize=PCA_FIGURE_SIZE)
    pca = PCA(n_components=2)
    pca_out = pca.fit_transform(V)
    
    ax.axvline(0, c="grey", ls=":")
    ax.axhline(0, c="grey", ls=":")
    
    # Plot regular data points
    if zero_data_mask is not None:
        # Plot non-zero data points
        non_zero_mask = ~zero_data_mask
        ax.scatter(pca_out[non_zero_mask, 0], pca_out[non_zero_mask, 1], 
                  c=string_array[0][non_zero_mask], alpha=0.8, s=60)
        
        # Plot zero data points with special highlighting
        if zero_data_mask.sum() > 0:
            ax.scatter(pca_out[zero_data_mask, 0], pca_out[zero_data_mask, 1], 
                      c=string_array[0][zero_data_mask], alpha=0.8, s=60, 
                      edgecolors='black', linewidths=2, marker='s',
                      label='All-zero perceptual data')
    else:
        # Plot all data points normally
        ax.scatter(pca_out[:, 0], pca_out[:, 1], c=string_array[0], alpha=0.8, s=60)
    
    # Add KDE contours for each cluster
    lim = 0.5  # Visual scale of KDE plot
    for c in np.unique(clusters[0]):
        plot_kde(pca_out[clusters[0] == c, :], ax=ax, colour=CLUSTER_COLORS[c], 
                label=None, limits=lim)
    #font size = 15 for both x and y labels
    ax.set_xlabel('PCA component 1 (visualization only)', fontsize=30)
    ax.set_ylabel('PCA component 2 (visualization only)', fontsize=30)
    title = f'Cluster Visualization - {data_type.capitalize()} Data (PCA with NMF coloring)'
    if zero_data_mask is not None and zero_data_mask.sum() > 0:
        title += f'\n({zero_data_mask.sum()} all-zero data points highlighted with squares)'
    ax.set_title(title, fontsize=35)
    
    #Set x and y tick font sizes
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    # Short label for legend based on modality
    cluster_prefix = "A" if data_type == "acoustic" else "P"

    # Create legend
    # handles = [plt.Line2D([0], [0], marker='o', color='w', 
    #                      markerfacecolor=CLUSTER_COLORS[c], markersize=10, 
    #                      label=f"Cluster {c}") for c in np.unique(clusters[0])]
    handles = [
        plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=CLUSTER_COLORS[c],
            markersize=10,
            label=f"{cluster_prefix}{c}"
        )
        for c in np.unique(clusters[0])
    ]

    
    # Add legend entry for all-zero data points if they exist
    if zero_data_mask is not None and zero_data_mask.sum() > 0:
        handles.append(plt.Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor='gray', markeredgecolor='black',
                                 markeredgewidth=2, markersize=10, 
                                 label='All-zero perceptual data'))
    
    ax.legend(handles=handles, loc="upper right", fontsize=30)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA plot saved: {save_path}")
    
    show_plot_if_enabled()


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
    from matplotlib.colors import LinearSegmentedColormap


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

    # Custom purple and green gradient palettes

    # Purple gradient for cluster distribution
    purple_cmap = LinearSegmentedColormap.from_list("purple_grad", ["#b39ddb", "#512da8"], N=len(categories))
    bar_colors = [purple_cmap(i / max(len(categories)-1, 1)) for i in range(len(categories))]

    # Green gradient for severe participants
    green_cmap = LinearSegmentedColormap.from_list("green_grad", ["#a5d6a7", "#388e3c"], N=len(categories_severe))
    bar_colors_severe = [green_cmap(i / max(len(categories_severe)-1, 1)) for i in range(len(categories_severe))]

    # Bar plot 1: Cluster size distribution
    plt.figure(figsize=BAR_FIGURE_SIZE)
    cluster_proportions = np.array(values) / len(clusters[0])
    bars = plt.bar(categories, cluster_proportions, color=bar_colors)
    plt.xlabel("Cluster ID", fontsize=25)
    plt.ylabel("Proportion of Participants", fontsize=25)
    plt.title(f"Distribution of Participants Across Clusters - {data_type.capitalize()}", fontsize=28)
    plt.ylim(0, 1)
    plt.xticks(categories, fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Add counts above bars
    for bar, count in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(count),
                 ha='center', va='bottom', fontsize=22, fontweight='bold')

    if save_dir:
        save_path = f"{save_dir}/cluster_distribution_{data_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster distribution plot saved: {save_path}")
        
        # Save cluster distribution percentages to CSV
        cluster_dist_df = pd.DataFrame({
            'cluster_id': categories,
            'count': values,
            'proportion': cluster_proportions
        })
        csv_path = f"{save_dir}/cluster_distribution_{data_type}.csv"
        cluster_dist_df.to_csv(csv_path, index=False)
        print(f"Cluster distribution data saved: {csv_path}")

    show_plot_if_enabled()

    # Bar plot 2: Severe participants per cluster (global proportion)
    plt.figure(figsize=BAR_FIGURE_SIZE)
    severe_proportions = np.array(values_severe) / len(clusters[0])
    bars_severe = plt.bar(categories_severe, severe_proportions, color=bar_colors_severe)
    plt.xlabel("Cluster ID", fontsize=20)
    plt.ylabel("Proportion of All Participants", fontsize=20)
    plt.title(f"Severe Participants per Cluster - {data_type.capitalize()} (Global Proportion)", fontsize=20)
    plt.ylim(0, 1)
    plt.xticks(categories_severe, fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Add counts above bars
    for bar, count in zip(bars_severe, values_severe):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(count),
                 ha='center', va='bottom', fontsize=22, fontweight='bold')

    if save_dir:
        save_path = f"{save_dir}/severe_global_{data_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Severe participants plot saved: {save_path}")

    show_plot_if_enabled()

    # Bar plot 3: Severity ratio within each cluster
    if categories_severe:  # Only plot if there are severe participants
        within_cluster_ratios = np.array(values_severe) / np.array([counts[c] for c in categories_severe])
        bars_within = None
        plt.figure(figsize=BAR_FIGURE_SIZE)
        bars_within = plt.bar(categories_severe, within_cluster_ratios, color=bar_colors_severe)
        plt.xlabel("Cluster ID", fontsize=28)
        plt.ylabel("Proportion of Cluster That Is Severe", fontsize=28)
        plt.title(f"Severity Percentage Within Each Cluster - {data_type.capitalize()}", fontsize=28)
        plt.ylim(0, 1)
        plt.xticks(categories_severe, fontsize=28)
        plt.yticks(fontsize=28)
        plt.grid(axis='y', linestyle=':', alpha=0.5)
        plt.tight_layout()

        # Add counts above bars
        for bar, count in zip(bars_within, values_severe):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, str(count),
                     ha='center', va='bottom', fontsize=28, fontweight='bold')

        if save_dir:
            save_path = f"{save_dir}/severity_within_cluster_{data_type}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Within-cluster severity plot saved: {save_path}")

        show_plot_if_enabled()


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
            
            # Clean perceptual feature names for better display
            if data_type == "perceptual":
                display_names = clean_perceptual_feature_names(list(all_features.index))
            else:
                display_names = list(feature_names)
            
            # Plot
            plt.bar(display_names, all_features.values, 
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

            show_plot_if_enabled()


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

    show_plot_if_enabled()

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

    show_plot_if_enabled()


def clean_perceptual_feature_names(feature_names: list) -> list:
    """
    Clean perceptual feature names by removing number prefixes and parenthetical content.
    
    Args:
        feature_names: List of feature names to clean
        
    Returns:
        List of cleaned feature names
    """
    cleaned_names = []
    for feat in feature_names:
        # Remove everything after the opening bracket
        if "(" in feat:
            cleaned = feat.split("(")[0].strip()
        else:
            cleaned = feat.strip()
        
        # Remove number and dot prefix (like "13. ")
        if ". " in cleaned:
            cleaned = cleaned.split(". ", 1)[1]
        
        cleaned_names.append(cleaned)
    
    return cleaned_names


def visualize_cluster_feature_statistics(cluster_idx: int, feature_names: list, means: np.ndarray, 
                                        stds: np.ndarray, medians: np.ndarray, q1: np.ndarray, 
                                        q3: np.ndarray, data_type: str, save_path: str = None) -> None:
    """
    Create statistical summary plot for cluster features.
    
    Args:
        cluster_idx: Cluster index
        feature_names: Names of features
        means: Mean values
        stds: Standard deviations
        medians: Median values
        q1: First quartile values
        q3: Third quartile values
        data_type: Type of data
        save_path: Path to save the plot
    """
    # Clean perceptual feature names for better display
    if data_type == "perceptual":
        display_names = clean_perceptual_feature_names(feature_names)
    else:
        display_names = feature_names
    
    plt.figure(figsize=(10, 6))
    plt.bar(display_names, means, yerr=stds, capsize=5, label='Mean ± Std')
    plt.scatter(display_names, medians, color='orange', label='Median', s=50)
    plt.scatter(display_names, q1, color='green', marker='x', label='Q1', s=50)
    plt.scatter(display_names, q3, color='red', marker='x', label='Q3', s=50)
    plt.ylabel("Feature Values")
    plt.legend()
    plt.title(f"Feature Statistics - Cluster {cluster_idx} ({data_type.capitalize()} Data)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster feature statistics saved: {save_path}")

    show_plot_if_enabled()


def visualize_cluster_feature_importance(cluster_idx: int, feature_names: list, scores: np.ndarray, 
                                       data_type: str, save_path: str = None) -> None:
    """
    Create feature importance plot for a specific cluster.
    
    Args:
        cluster_idx: Cluster index
        feature_names: Names of features
        scores: Importance scores
        data_type: Type of data
        save_path: Path to save the plot
    """
    # Clean perceptual feature names for better display
    if data_type == "perceptual":
        display_names = clean_perceptual_feature_names(feature_names)
    else:
        display_names = feature_names
    
    plt.figure(figsize=(10, 6))
    plt.bar(display_names, scores, color='skyblue')
    plt.ylabel("Importance Scores")
    plt.title(f"Feature Importance - Cluster {cluster_idx} ({data_type.capitalize()} Data)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Cluster feature importance saved: {save_path}")

    show_plot_if_enabled()


def visualize_cluster_features(cluster_idx: int, feature_names: list, means: np.ndarray, 
                             stds: np.ndarray, medians: np.ndarray, q1: np.ndarray, 
                             q3: np.ndarray, scores: np.ndarray, data_type: str, 
                             save_path: str = None) -> None:
    """
    Create detailed feature analysis plots for a specific cluster.
    DEPRECATED: Use visualize_cluster_feature_statistics and visualize_cluster_feature_importance instead.
    
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
    print("⚠️  visualize_cluster_features is deprecated. Use separate functions instead.")
    
    # For backward compatibility, create both plots with modified save paths
    if save_path:
        base_path = save_path.replace('.png', '')
        stats_path = f"{base_path}_statistics.png"
        importance_path = f"{base_path}_importance.png"
        
        visualize_cluster_feature_statistics(cluster_idx, feature_names, means, stds, medians, q1, q3, data_type, stats_path)
        visualize_cluster_feature_importance(cluster_idx, feature_names, scores, data_type, importance_path)
    else:
        visualize_cluster_feature_statistics(cluster_idx, feature_names, means, stds, medians, q1, q3, data_type)
        visualize_cluster_feature_importance(cluster_idx, feature_names, scores, data_type)
