# ALS Clustering Analysis - Modular Pipeline

This repository contains a modular Python pipeline for analyzing ALS (Amyotrophic Lateral Sclerosis) patient data using Non-negative Matrix Factorization (NMF) clustering on acoustic and perceptual speech features.

## Overview

The analysis pipeline performs comprehensive clustering of ALS patients based on:
- **Acoustic features**: Extracted from speech recordings (formants, spectral features, etc.)
- **Perceptual features**: Clinical ratings of speech characteristics
- **Cross-modal analysis**: Comparing clusters across both data types

## Project Structure

```
backup/
├── src/                          # Source code modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration and parameters
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── nmf_clustering.py        # NMF clustering implementation
│   ├── visualization.py         # Plotting and visualization functions
│   ├── cross_modal_analysis.py  # Cross-modal comparison analysis
│   └── results_export.py        # Results export and reporting
├── outputs/                     # Generated outputs (created by script)
│   ├── plots/                   # All visualization outputs
│   ├── tables/                  # CSV tables and statistics
│   └── results/                 # Analysis results and reports
├── run_analysis.py              # Main execution script
└── README.md                    # This file
```

## Features

### Data Processing
- Automatic loading of acoustic features from CSV files
- Perceptual ratings preprocessing and cleaning
- Demographics data integration
- Missing data handling and feature filtering

### Clustering Analysis
- Non-negative Matrix Factorization (NMF) clustering
- Automatic optimal cluster number detection using Kneedle algorithm
- Separate analysis for acoustic and perceptual data
- Feature importance ranking for each cluster

### Visualizations
- PCA-based cluster visualizations with KDE contours
- Cluster size distribution plots
- Severity analysis by cluster
- Feature importance bar plots
- Confusion matrix for cross-modal comparison
- Detailed statistical summaries for top features

### Cross-Modal Analysis
- Optimal cluster alignment between acoustic and perceptual data
- Jensen-Shannon divergence computation
- Cosine similarity analysis
- Bhattacharyya coefficient calculation
- Statistical association testing (Cramer's V)

### Results Export
- Organized CSV files for all statistics
- Cluster assignments and mappings
- Top features per cluster
- Cross-modal similarity metrics
- Comprehensive summary report

## Installation

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy kneed nimfa
```

### Required Data Files
Ensure the following data files are available in the specified paths (update `config.py` if needed):

1. **Acoustic Features**: CSV files in `WavFilesConvertedALL441k/Features/`
2. **Perceptual Data**: `Data/perceptual_ratings_final_short.csv`
3. **Demographics**: `Data/ALSDemographics.csv`

## Usage

### Quick Start
```bash
python run_analysis.py
```

This will run the complete analysis pipeline and generate all outputs in the `outputs/` directory.

### Customization

Edit `src/config.py` to modify:
- File paths
- Analysis parameters (ALSIBD threshold, NMF settings, etc.)
- Plotting parameters
- Output directories

### Module Usage

You can also use individual modules for specific analyses:

```python
from src.data_loader import merge_all_data
from src.nmf_clustering import run_nmf_clustering
from src.visualization import plot_pca_clusters

# Load data
all_data, clinical_summaries, demographics, first_acoustic = merge_all_data()

# Run clustering
all_clusters, all_feature_maps, coef_dists, basis_dists = run_nmf_clustering(
    all_data, clinical_summaries, first_acoustic
)

# Generate specific plots
# ... (see run_analysis.py for examples)
```

## Output Description

### Plots Directory (`outputs/plots/`)
- `pca_clusters_acoustic.png` - PCA visualization of acoustic clusters
- `pca_clusters_perceptual.png` - PCA visualization of perceptual clusters
- `cluster_distribution_*.png` - Cluster size distributions
- `severe_global_*.png` - Severe participants by cluster
- `severity_within_cluster_*.png` - Severity ratios within clusters
- `feature_importance_*_cluster_*.png` - Feature importance for each cluster
- `confusion_matrix.png` - Acoustic vs perceptual cluster confusion matrix
- `similarity_distributions.png` - Cross-modal similarity metrics
- `cluster_features_*_cluster_*.png` - Detailed feature analysis per cluster

### Tables Directory (`outputs/tables/`)
- `demographic_statistics.csv` - Summary statistics for demographic variables
- `cluster_assignments_*.csv` - Participant cluster assignments
- `cluster_statistics_*.csv` - Cluster size and percentage summaries
- `top_features_*.csv` - Top-ranked features per cluster with importance scores

### Results Directory (`outputs/results/`)
- `cluster_feature_mapping_*.csv` - Features associated with each cluster
- `cluster_severity_statistics.csv` - ALSIBD severity stats by cluster
- `acoustic_perceptual_mapping.csv` - Mapping between acoustic and perceptual clusters
- `cross_modal_similarities.csv` - Similarity metrics for each participant
- `combined_cluster_assignments.csv` - Both acoustic and perceptual assignments
- `analysis_summary_report.txt` - Comprehensive text summary of results

## Key Parameters

### Analysis Settings (config.py)
- `ALSIBD_THRESHOLD = 25` - Threshold for severe vs mild classification
- `NMF_RANK_RANGE = (2, 8)` - Range of cluster numbers to test
- `TOP_N_FEATURES = "kneedle"` - Feature selection method: integer (e.g., 5) or "kneedle" for dynamic selection
- `USE_VERSIONED_OUTPUTS = True` - Enable automatic output versioning (run1, run2, etc.)
- `RANDOM_SEED = 1` - For reproducible results

### Data Filtering
- Removes f3, f4, f5 formant features (configurable)
- Excludes features with NaN values
- Filters calculated/summary scores from perceptual data

## Methodology

1. **Data Preprocessing**: Load and clean acoustic features, perceptual ratings, and demographics
2. **NMF Clustering**: Apply NMF to both data types separately, using Kneedle algorithm for optimal cluster numbers
3. **Feature Analysis**: Extract top features and compute importance scores for each cluster
4. **Cross-Modal Alignment**: Use optimal assignment based on Jensen-Shannon divergence to align clusters
5. **Statistical Analysis**: Compute similarity metrics and association strengths
6. **Visualization**: Generate comprehensive plots for interpretation
7. **Results Export**: Save all results in organized, analysis-ready formats

## Citation

If you use this code in your research, please cite:

```
[Add appropriate citation when published]
```

## Contact

For questions or issues, please contact Vishnu Akundi.

## License

[Add appropriate license information]
