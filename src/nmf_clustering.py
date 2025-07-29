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
    nmf = nimfa.Nmf(data.T, rank=optimal_components, max_iter=NMF_MAX_ITER, seed=NMF_SEED)
    fit = nmf()
    basis = fit.fit.basis()  # features
    coef = fit.fit.coef()  # individual clusters
    
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
                                columns: pd.Index, data_type: str) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Extract cluster assignments and feature mappings from NMF results.
    
    Args:
        basis: NMF basis matrix
        coef: NMF coefficient matrix
        columns: Feature column names
        data_type: Type of data ("acoustic" or "perceptual")
        
    Returns:
        Tuple of (clusters, features_dataframe, coefficient_distributions)
    """
    # Map basis and coefficient matrices to feature weights and cluster indices
    features = np.array(np.argmax(basis, axis=1))
    features_df = pd.DataFrame(data=features, index=columns)
    clusters = np.array(np.argmax(coef, axis=0))
    
    # Calculate coefficient distributions (softmax)
    coef_dist = np.exp(coef) / np.sum(np.exp(coef), axis=0)
    
    # Save feature weights
    output_path = f"{RESULTS_DIR}/feature_weights_{data_type}.csv"
    features_df.to_csv(output_path)
    print(f"Feature weights saved to: {output_path}")
    
    return clusters, features_df, coef_dist


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
        top_features = basis_df.iloc[:, cluster_idx].nlargest(TOP_N_FEATURES)
        
        # Store in dictionary
        top_features_per_cluster[cluster_idx] = {
            "Feature Names": list(top_features.index),
            "Scores": list(top_features.values)
        }
        
        # Display feature names and scores
        print(f"\nCluster {cluster_idx}:")
        for feature, score in zip(top_features.index, top_features.values):
            print(f"  {feature}: {score:.4f}")
    
    return top_features_per_cluster


def run_nmf_clustering(all_data: pd.DataFrame, clinical_summaries: pd.DataFrame, 
                      first_acoustic: int) -> Tuple[Dict, Dict, Dict, Dict, pd.DataFrame]:
    """
    Run complete NMF clustering analysis for both acoustic and perceptual data.
    
    Args:
        all_data: Combined dataset
        clinical_summaries: Clinical summary scores
        first_acoustic: Index of first acoustic feature
        
    Returns:
        Tuple of (all_clusters, all_feature_maps, coef_dists, basis_dists, clinical_summaries_filtered)
    """
    all_clusters = {}
    all_feature_maps = {}
    coef_dists = {}
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
        clusters, features_df, coef_dist = extract_clusters_and_features(
            basis, coef, columns, data_type
        )
        
        # Get top features
        top_features = get_top_features(basis, columns, data_type)
        
        # Store results
        all_clusters[data_type] = clusters
        all_feature_maps[data_type] = features_df
        coef_dists[data_type] = coef_dist
        basis_dists[data_type] = basis
        
        print(f"Completed {data_type} analysis with {optimal_components} clusters")
    
    return all_clusters, all_feature_maps, coef_dists, basis_dists, clinical_summaries_filtered
