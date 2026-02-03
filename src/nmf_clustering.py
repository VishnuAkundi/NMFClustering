"""
Non-negative Matrix Factorization (NMF) clustering module.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import nimfa
from typing import Tuple, Dict, Any
from .config import *
from . import config


def perform_nmf_analysis(data: np.ndarray, data_type: str) -> Tuple[Any, Any, int]:
    """
    Perform NMF analysis to determine optimal number of components.
    
    Args:
        data: Input data matrix
        data_type: Type of data ("acoustic" or "perceptual")
        
    Returns:
        Tuple of (basis_matrix, coefficient_matrix, optimal_components)
    """
    print(f"\nPerforming NMF analysis for {data_type} data...")
    print(f"Data shape: {data.shape}")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Define search range for NMF components
    ranks = np.arange(*NMF_RANK_RANGE)
    results = {}
    
    # Test different numbers of components
    for rank in ranks:
        nmf = nimfa.Nmf(data.T, rank=rank, max_iter=NMF_MAX_ITER, seed=NMF_SEED)
        fit = nmf()
        
        # Calculate quality measures
        rss = fit.fit.rss()  # Residual Sum of Squares
        evar = fit.fit.evar()  # Explained variance
        
        results[rank] = {
            'rss': rss,
            'evar': evar,
        }
    
    # Generate vector of error values
    errors = [results[r]['rss'] for r in ranks]
    
    # Use Kneedle algorithm to identify optimal number of components
    kneedle = KneeLocator(x=ranks, y=errors, S=0.0, curve="convex", direction="decreasing")
    optimal_components = kneedle.knee
    print(f"Optimal components = {optimal_components}")
    
    # Re-fit using optimal number of components
    print('DEBUG: Data input shape:', data.T.shape)
    nmf = nimfa.Nmf(data.T, rank=optimal_components, max_iter=NMF_MAX_ITER, seed=NMF_SEED)
    fit = nmf()
    basis = fit.fit.basis()  # features
    coef = fit.fit.coef()  # individual clusters
    print(f"DEBUG: Basis shape: {basis.shape}, Coefficient shape: {coef.shape}")
    
    # Validate NMF fit
    V_hat = np.dot(basis, coef)
    rss = np.power((data.T - V_hat), 2)
    # Avoid division by zero - calculate relative error more safely
    total_variance = np.var(data.T)
    if total_variance > 0:
        reconstruction_error = np.mean(rss) / total_variance * 100
        print(f"Reconstruction error (% of original): {reconstruction_error:.2f}%")
    else:
        print("Reconstruction error: Cannot calculate (zero variance in data)")
    
    return basis, coef, optimal_components


def extract_clusters_and_features(basis: np.ndarray, coef: np.ndarray, 
                                columns: pd.Index, data_type: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Extract cluster assignments and feature mappings from NMF results.
    
    Args:
        basis: NMF basis matrix
        coef: NMF coefficient matrix
        columns: Feature column names
        data_type: Type of data ("acoustic" or "perceptual")
        
    Returns:
        Tuple of (clusters, features_dataframe, coefficient_distributions, raw_coefficients)
    """
    # Map basis and coefficient matrices to feature weights and cluster indices
    features = np.array(np.argmax(basis, axis=1))
    features_df = pd.DataFrame(data=features, index=columns)
    clusters = np.array(np.argmax(coef, axis=0))
    
    # Store raw coefficients for column similarity method (before softmax)
    raw_coef = coef.copy()
    
    # Calculate coefficient distributions (softmax)
    coef_dist = np.exp(coef) / np.sum(np.exp(coef), axis=0)
    
    # Save feature weights
    output_path = f"{config.RESULTS_DIR}/feature_weights_{data_type}.csv"
    features_df.to_csv(output_path)
    print(f"Feature weights saved to: {output_path}")
    
    # Save raw NMF matrices
    # Save basis matrix (H) - features x components
    basis_path = f"{config.RESULTS_DIR}/nmf_basis_matrix_{data_type}.csv"
    basis_df = pd.DataFrame(basis, index=columns, columns=[f'Component_{i}' for i in range(basis.shape[1])])
    basis_df.to_csv(basis_path)
    print(f"NMF basis matrix (H) saved to: {basis_path}")
    
    # Save coefficient matrix (W) - components x samples  
    coef_path = f"{config.RESULTS_DIR}/nmf_coefficient_matrix_{data_type}.csv"
    coef_df = pd.DataFrame(coef, index=[f'Component_{i}' for i in range(coef.shape[0])], 
                          columns=[f'Sample_{i}' for i in range(coef.shape[1])])
    coef_df.to_csv(coef_path)
    print(f"NMF coefficient matrix (W) saved to: {coef_path}")
    
    return clusters, features_df, coef_dist, raw_coef


def determine_optimal_features(feature_scores: np.ndarray, cluster_idx: int) -> int:
    """
    Determine optimal number of features using Kneedle algorithm.
    
    Args:
        feature_scores: Array of feature importance scores (sorted descending)
        cluster_idx: Cluster index (for debugging output)
        
    Returns:
        Optimal number of features to select
    """
    if len(feature_scores) < 3:
        return len(feature_scores)
    
    # Create x-axis (feature indices)
    x = np.arange(len(feature_scores))
    
    try:
        # Use KneeLocator to find the elbow point
        kneedle = KneeLocator(
            x, feature_scores, 
            curve="convex", 
            direction="decreasing",
            online=True
        )
        
        optimal_n = kneedle.elbow if kneedle.elbow is not None else min(5, len(feature_scores))
        
        # Ensure we get at least 2 features and at most 10
        optimal_n = max(2, min(optimal_n + 1, min(10, len(feature_scores))))
        
        print(f"    Kneedle selected {optimal_n} features for cluster {cluster_idx}")
        return optimal_n
        
    except Exception as e:
        print(f"    Kneedle failed for cluster {cluster_idx}: {e}, defaulting to 5 features")
        return min(5, len(feature_scores))


def get_top_features(basis: np.ndarray, columns: pd.Index, data_type: str) -> Dict:
    """
    Extract top features for each cluster.
    
    Args:
        basis: NMF basis matrix
        columns: Feature column names
        data_type: Type of data ("acoustic" or "perceptual")
        
    Returns:
        Dictionary of top features per cluster
    """
    basis_df = pd.DataFrame(basis, index=columns)
    top_features_per_cluster = {}
    
    print(f"\nTop features for {data_type} data:")
    for cluster_idx in range(basis_df.shape[1]):
        # Sort features by importance in descending order
        sorted_features = basis_df.iloc[:, cluster_idx].sort_values(ascending=False)
        
        # Determine number of features to select
        if isinstance(TOP_N_FEATURES, str) and TOP_N_FEATURES.lower() == "kneedle":
            n_features = determine_optimal_features(sorted_features.values, cluster_idx)
        else:
            n_features = min(int(TOP_N_FEATURES), len(sorted_features))
        
        # Get top N features
        top_features = sorted_features.head(n_features)
        
        # Store in dictionary
        top_features_per_cluster[cluster_idx] = {
            "Feature Names": list(top_features.index),
            "Scores": list(top_features.values)
        }
        
        # Display feature names and scores
        print(f"\nCluster {cluster_idx} (selected {n_features} features):")
        for feature, score in zip(top_features.index, top_features.values):
            print(f"  {feature}: {score:.4f}")
    
    return top_features_per_cluster


def run_nmf_clustering(all_data: pd.DataFrame, clinical_summaries: pd.DataFrame, 
                      first_acoustic: int) -> Tuple[Dict, Dict, Dict, Dict, Dict, pd.DataFrame]:
    """
    Run complete NMF clustering analysis for both acoustic and perceptual data.
    
    Args:
        all_data: Combined dataset
        clinical_summaries: Clinical summary scores
        first_acoustic: Index of first acoustic feature
        
    Returns:
        Tuple of (all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists, clinical_summaries_filtered)
    """
    all_clusters = {}
    all_feature_maps = {}
    coef_dists = {}
    raw_coef_dists = {}
    basis_dists = {}
    
    # Filter clinical summaries to match all_data participants (do this once at the start)
    clinical_summaries_filtered = clinical_summaries.loc[
        [i in all_data.index for i in clinical_summaries.index], :
    ]
    print(f"Clinical summaries filtered: {clinical_summaries_filtered.shape[0]} participants")
    
    for data_type in ["acoustic", "perceptual"]:
        print(f"\n{'='*50}")
        print(f"Processing {data_type.upper()} data")
        print(f"{'='*50}")
        
        if data_type == "acoustic":
            V = all_data.iloc[:, first_acoustic:]
            columns = all_data.columns[first_acoustic:]
        elif data_type == "perceptual":
            V = all_data.iloc[:, :first_acoustic]
            columns = all_data.columns[:first_acoustic]
        
        # Rescale data for NMF (enforce positivity and uniform scale)
        mms = MinMaxScaler()
        V_scaled = mms.fit_transform(V)
        
        # Perform NMF analysis
        basis, coef, optimal_components = perform_nmf_analysis(V_scaled, data_type)
        
        # Extract clusters and features
        clusters, features_df, coef_dist, raw_coef = extract_clusters_and_features(
            basis, coef, columns, data_type
        )
        
        # Get top features
        top_features = get_top_features(basis, columns, data_type)
        
        # Store results
        all_clusters[data_type] = clusters
        all_feature_maps[data_type] = features_df
        coef_dists[data_type] = coef_dist
        raw_coef_dists[data_type] = raw_coef
        basis_dists[data_type] = basis
        
        print(f"Completed {data_type} analysis with {optimal_components} clusters")
    
    return all_clusters, all_feature_maps, coef_dists, raw_coef_dists, basis_dists, clinical_summaries_filtered
