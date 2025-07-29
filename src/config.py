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
OUTPUT_DIR = f"{BASE_DIR}/backup/outputs"
PLOTS_DIR = f"{OUTPUT_DIR}/plots"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# Analysis parameters
ALSIBD_THRESHOLD = 25
NMF_RANK_RANGE = (2, 8)
NMF_MAX_ITER = 10000
NMF_SEED = "nndsvd"
RANDOM_SEED = 1
TOP_N_FEATURES = 5

# Plotting parameters
PCA_FIGURE_SIZE = (16, 8)
BAR_FIGURE_SIZE = (7, 4)
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
