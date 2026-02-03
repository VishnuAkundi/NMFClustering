"""
Configuration file for ALS clustering analysis.
Contains all file paths and analysis parameters.
"""

import os

# Base directory
BASE_DIR = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering"

# Data paths
ACOUSTIC_FEATURES_PATH = f"{BASE_DIR}/WavFilesConvertedALL441k/Features/*.csv"
PERCEPTUAL_DATA_PATH = f"{BASE_DIR}/Data/perceptual_ratings_final_short.csv"
DEMOGRAPHICS_PATH = f"{BASE_DIR}/Data/ALSDemographics.csv"

# Output paths
OUTPUT_BASE_DIR = f"{BASE_DIR}/NMFClustering/outputs"
OUTPUT_DIR = OUTPUT_BASE_DIR  # Will be updated to versioned directory if USE_VERSIONED_OUTPUTS is True
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# Analysis parameters
ALSIBD_THRESHOLD = 25
NMF_RANK_RANGE = (2, 8)
NMF_MAX_ITER = 10000
NMF_SEED = "nndsvd"
RANDOM_SEED = 1

# Feature selection method - can be an integer (fixed number) or "kneedle" (dynamic)
TOP_N_FEATURES = "kneedle"  # Options: integer (e.g., 5) or "kneedle" for dynamic selection

# Output versioning - automatically increments run folders
USE_VERSIONED_OUTPUTS = True

# Plot display settings
SHOW_PLOTS = False  # Set to True to display plots interactively, False to only save

# Plotting parameters
PCA_FIGURE_SIZE = (16, 8)
BAR_FIGURE_SIZE = (16, 8)
FEATURE_PLOT_SIZE = (12, 6)
CLUSTER_FEATURE_SIZE = (12, 10)
SIMILARITY_PLOT_SIZE = (12, 5)

# Color mapping for clusters
CLUSTER_COLORS = {
    0: "firebrick", 
    1: "dodgerblue", 
    2: "purple", 
    3: "salmon", 
    4: "goldenrod", 
    5: "forestgreen"
}

# Features to exclude
FORMANT_FEATURES_TO_DROP = ['f3', 'f4', 'f5']

# Demographic columns to analyze
DEMOGRAPHIC_COLUMNS = [
    'age_years', 
    'sex_arm3', 
    'alsfrs_total_score_r_arm3', 
    'bulbar_score_r_arm3', 
    'alsibd_sit_intell', 
    'alsibd_spr', 
    'alsibd_total_score_v3_v3'
]

# Cross-modal alignment method configuration
# Options: "jsd_hungarian" (default), "column_similarity"
ALIGNMENT_METHOD = "column_similarity"  # Use "jsd_hungarian" for JSD + Hungarian alignments

# Column similarity alignment parameters (only used if ALIGNMENT_METHOD = "column_similarity")
COLUMN_SIMILARITY_METRIC = "cosine"  # Options: "cosine", "correlation"

# Handle all-zero perceptual data (healthy participants)
# When True, adds small epsilon to all-zero rows to prevent NMF mathematical issues
HANDLE_ZERO_PERCEPTUAL = False

# Epsilon value to add to all-zero perceptual rows (only if HANDLE_ZERO_PERCEPTUAL = True)
# Should be very small to represent "virtually no symptoms" rather than "no data"
PERCEPTUAL_EPSILON = 1e-10

# Visualization options for all-zero perceptual data
# When True, highlights participants with all-zero perceptual data in plots
HIGHLIGHT_ZERO_PERCEPTUAL = False
