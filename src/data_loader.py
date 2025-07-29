"""
Data loading and preprocessing module for ALS clustering analysis.
"""

import pandas as pd
import numpy as np
import glob
from typing import Tuple, Dict
from .config import *


def load_acoustic_features() -> pd.DataFrame:
    """
    Load and preprocess acoustic features from CSV files.
    
    Returns:
        pd.DataFrame: Processed acoustic features with cleaned participant IDs
    """
    print("Loading acoustic features...")
    all_csvs = glob.glob(ACOUSTIC_FEATURES_PATH)
    all_results = []
    
    for csv_file in all_csvs:
        csv_file = csv_file.replace("\\", "/")
        fname = csv_file.split("/")[-1].split(".")[0]
        temp_data = pd.read_csv(csv_file, index_col="Unnamed: 0")
        temp_data.index = [fname]
        all_results.append(temp_data)
    
    # Concatenate all results
    all_results = pd.concat(all_results, axis=0)
    print(f"Loaded {len(all_results)} acoustic feature files")
    
    # Remove specified formant features
    drop_cols = [any(feat in c for feat in FORMANT_FEATURES_TO_DROP) for c in all_results.columns]
    all_results.drop(all_results.columns[drop_cols], axis=1, inplace=True)
    
    # Clean participant IDs
    all_results.index = [i.replace("-V", "").replace("ALS_", "").split("_")[0] for i in all_results.index]
    
    return all_results


def load_perceptual_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess perceptual ratings data.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (perceptual_data, clinical_summaries)
    """
    print("Loading perceptual data...")
    perceptual = pd.read_csv(PERCEPTUAL_DATA_PATH)
    perceptual.index = perceptual['Participant ID'].values
    perceptual.index = [i.replace("-V", "") for i in perceptual.index]
    perceptual = perceptual.iloc[:, 1:]
    
    # Extract clinical summaries (calculated scores)
    summary_cols = [("calculated" in c) for c in perceptual.columns]
    clinical_summaries = perceptual.loc[:, summary_cols]
    
    # Remove columns with NaN values and calculated scores
    drop_cols = [any(np.isnan(perceptual[c].values)) or ("calculated" in c) for c in perceptual.columns]
    perceptual.drop(perceptual.columns[drop_cols], inplace=True, axis=1)
    
    print(f"Loaded perceptual data: {perceptual.shape[0]} participants, {perceptual.shape[1]} features")
    
    return perceptual, clinical_summaries


def load_demographics() -> pd.DataFrame:
    """
    Load and preprocess demographics data.
    
    Returns:
        pd.DataFrame: Processed demographics data
    """
    print("Loading demographics data...")
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, index_col='participant_id')
    demographics = demographics.loc[~demographics.index.duplicated(keep='first')]
    demographics.index = [i.replace("-V", "") for i in demographics.index]
    
    print(f"Loaded demographics: {demographics.shape}")
    
    return demographics


def merge_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """
    Load and merge all data sources.
    
    Returns:
        Tuple containing:
        - all_data: merged perceptual and acoustic data
        - clinical_summaries: clinical summary scores
        - demographics: merged demographics with all data
        - first_acoustic: index of first acoustic feature column
    """
    # Load individual datasets
    acoustic_data = load_acoustic_features()
    perceptual_data, clinical_summaries = load_perceptual_data()
    demographics_data = load_demographics()
    
    # Merge perceptual and acoustic data
    all_data = pd.merge(left=perceptual_data, right=acoustic_data, left_index=True, right_index=True)
    first_acoustic = np.where(all_data.columns=="f1")[0][0]
    
    # Merge with demographics
    demographics_merged = pd.merge(left=all_data, right=demographics_data, left_index=True, right_index=True)
    
    print(f"Final merged dataset: {all_data.shape[0]} participants, {all_data.shape[1]} features")
    print(f"First acoustic feature at column {first_acoustic}")
    
    return all_data, clinical_summaries, demographics_merged, first_acoustic


def filter_demographics(demographics: pd.DataFrame, clinical_summaries: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter demographics based on ALSIBD scores.
    
    Args:
        demographics: Demographics dataframe
        clinical_summaries: Clinical summaries dataframe
    
    Returns:
        Tuple of filtered demographics and clinical demographics
    """
    # Create clinical demographics for analysis
    clinical_dem = pd.merge(left=clinical_summaries, right=demographics, left_index=True, right_index=True)
    
    # Filter based on ALSIBD threshold
    clinical_dem_filtered = clinical_dem[clinical_dem['ALSIBD Total Score (calculated)'] <= ALSIBD_THRESHOLD]
    demographics_filtered = demographics[demographics['alsibd_total_score_v3_v3'] > ALSIBD_THRESHOLD]
    
    print(f"Demographics filtered: {demographics_filtered.shape[0]} participants with ALSIBD > {ALSIBD_THRESHOLD}")
    
    return demographics_filtered, clinical_dem_filtered


def get_demographic_stats(demographics: pd.DataFrame) -> Dict:
    """
    Calculate demographic statistics for specified columns.
    
    Args:
        demographics: Demographics dataframe
        
    Returns:
        Dict: Statistics for each demographic column
    """
    stats = {}
    for col in DEMOGRAPHIC_COLUMNS:
        if col in demographics.columns:
            stats[col] = {
                'median': demographics[col].median(),
                'iqr': np.subtract(*np.nanpercentile(demographics[col], [75, 25]))
            }
    
    return stats
