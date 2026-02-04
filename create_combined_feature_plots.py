


# """
# Create Combined Feature Importance Plots

# This script creates two separate 2x2 grids showing feature importance plots:
# 1. One plot for all 4 acoustic clusters
# 2. Another plot for all 4 perceptual clusters

# Usage:
#     python create_combined_feature_plots.py --run_number 32
#     python create_combined_feature_plots.py --run_dir "outputs/run32"

# Output:
#     - combined_feature_importance_acoustic.png (2x2 grid of acoustic clusters)
#     - combined_feature_importance_perceptual.png (2x2 grid of perceptual clusters)

# Author: GitHub Copilot
# Date: 2025
# """

# import argparse
# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# from pathlib import Path


# # ---------------------------
# # Subsystem dictionaries
# # ---------------------------

# perceptual_features = {
#     "Phonatory": ["Strained Voice", "Overall Dysphonia"],
#     "Respiratory": ["Loudness Decay", "Short Phrases"],
#     "Respiratory/ Phonatory": ["Reduced Loudness"],
#     "Prosody": ["Monopitch and/or Monoloudness", "Excess and Equal Stress", "Reduced Stress", "Overall Dysprosody"],
#     "Resonatory": ["Hypernasality"],
#     "Articulatory": ["Imprecise Articulation", "Slow Articulation Rate"],
#     "Articulatory/ Respiratory/ Prosody": ["Reduced Overall Speaking Rate"],
# }

# perceptual_colors = {
#     "Phonatory": "#E41A1C",
#     "Respiratory": "#4C97D8",
#     "Respiratory/ Phonatory": "#8EE68E",
#     "Prosody": "#8EE68E",
#     "Resonatory": "#FFD92F",
#     "Articulatory": "#FDD0A2",
#     "Articulatory/ Respiratory/ Prosody": "#7F7F7F",
# }

# acoustic_features = {
#     "Articulatory": [
#         "f1","f2","f1_bw","f2_bw",
#         "f1_d_dx_median", "f1_d_dx_prc_5","f1_d_dx_prc_95","f1_d_dx_prc_5_95",
#         "f2_d_dx_median","f2_d_dx_prc_5","f2_d_dx_prc_95","f2_d_dx_prc_5_95",
#         "F1_F2_comp",
#         "speech_dur","mean_phrase_dur","cv_phrase_dur"
#     ],
#     "Phonatory": ["f0_mean","f0_std","CPP_mean","CPP_F1_comp","CPP_F2_comp"],
#     "Phonatory/ Respiratory": ["intensity_CV"],
#     "Articulatory/ Respiratory/ Prosody": ["speech_rate","total_dur"],
#     "Respiratory": ["percent_pause","num_pause","mean_pause_dur","cv_pause_dur","total_pause_dur"],
#     "Spectral": [
#         "fft_peaks1","fft_peaks2","fft_ampli1","fft_ampli2",
#         "nrj_below_boundary","nrj_above_boundary","nrj_3_6","ratio_below_above"
#     ],
# }

# acoustic_colors = {
#     "Articulatory": "#FDD0A2",
#     "Phonatory": "#E41A1C",
#     "Phonatory/ Respiratory": "#984EA3",
#     "Articulatory/ Respiratory/ Prosody": "#A65628",
#     "Respiratory": "#4C97D8",
#     "Spectral": "#33A02C",
# }

# # ---------------------------
# # Helpers: robust matching
# # ---------------------------

# def _norm(s: str) -> str:
#     """Normalize strings for matching (case/space/punct tolerant)."""
#     if s is None:
#         return ""
#     s = str(s).strip().lower()
#     # keep letters/numbers, turn everything else into spaces
#     out = []
#     for ch in s:
#         out.append(ch if ch.isalnum() else " ")
#     return " ".join("".join(out).split())

# def _perceptual_short_name(raw: str) -> str:
#     """
#     Your perceptual feature_name looks like:
#     '13. Strained Voice (excess vocal effort/hyperfunction)'
#     We map it to 'Strained Voice' for subsystem lookup.
#     """
#     if raw is None:
#         return ""
#     raw = str(raw)
#     if ". " in raw and " (" in raw:
#         start_idx = raw.find(". ") + 2
#         end_idx = raw.find(" (")
#         return raw[start_idx:end_idx].strip()
#     return raw.strip()

# def _build_feature_to_subsystem(features_by_subsystem: dict) -> dict:
#     """Inverse map: normalized_feature -> subsystem."""
#     inv = {}
#     for subsystem, feats in features_by_subsystem.items():
#         for f in feats:
#             inv[_norm(f)] = subsystem
#     return inv

# PERCEPTUAL_F2S = _build_feature_to_subsystem(perceptual_features)
# ACOUSTIC_F2S = _build_feature_to_subsystem(acoustic_features)





# def load_feature_importance_data(run_dir: str) -> dict:
#     """
#     Load feature importance data from the specified run directory.
    
#     Args:
#         run_dir: Path to the run directory containing results
        
#     Returns:
#         Dictionary containing feature importance data for both modalities
#     """
#     results_dir = os.path.join(run_dir, "results")
#     feature_file = os.path.join(results_dir, "feature_importances_and_values.csv")
    
#     if not os.path.exists(feature_file):
#         raise FileNotFoundError(f"Feature importance file not found: {feature_file}")
    
#     df = pd.read_csv(feature_file)
    
#     # Group by modality and cluster
#     data = {}
#     for modality in ['acoustic', 'perceptual']:
#         data[modality] = {}
#         modality_data = df[df['modality'] == modality]
        
#         for cluster_id in sorted(modality_data['cluster_id'].unique()):
#             cluster_data = modality_data[modality_data['cluster_id'] == cluster_id]
#             # Sort by feature rank to maintain order
#             cluster_data = cluster_data.sort_values('feature_rank')
#             data[modality][cluster_id] = cluster_data
    
#     return data

# def create_feature_importance_subplot(ax, cluster_data, cluster_id, modality, max_features=10, x_max=None):
#     """
#     Create a feature importance subplot for a single cluster.
    
#     Args:
#         ax: Matplotlib axis object
#         cluster_data: DataFrame with feature importance data for the cluster
#         cluster_id: Cluster ID number
#         modality: 'acoustic' or 'perceptual'
#         max_features: Maximum number of features to display
#         x_max: Maximum x-axis value to use for consistent scaling across subplots
#     """
#     # Get top features (limit to max_features)
#     plot_data = cluster_data.head(max_features)
    
#     # Create horizontal bar plot with increased spacing between bars
#     bar_spacing = 1.5 if modality == 'perceptual' else 1.0  # More spacing for perceptual due to multi-line labels
#     y_pos = np.arange(len(plot_data)) * bar_spacing
#     feature_names = plot_data['feature_name'].values
#     importance_scores = plot_data['nmf_importance_score'].values
    
#     # Process feature names based on modality
#     display_names = []
#     for name in feature_names:
#         if modality == 'perceptual':
#             # Extract the short name between ". " and " ("
#             # Example: "13. Strained Voice (excess vocal effort/hyperfunction)" -> "Strained Voice"
#             if '. ' in name and ' (' in name:
#                 start_idx = name.find('. ') + 2
#                 end_idx = name.find(' (')
#                 short_name = name[start_idx:end_idx]
#             else:
#                 short_name = name
            
#             # Split long names into two lines if needed (more than 15 characters)
#             # Use a smaller line spacing to keep the two lines of the same label close together
#             if len(short_name) > 15:
#                 words = short_name.split()
#                 if len(words) > 1:
#                     mid = len(words) // 2
#                     # Use a smaller line spacing so the two lines stay close together
#                     short_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
            
#             display_names.append(short_name)
#         else:
#             # For acoustic features, truncate if too long
#             if len(name) > 25:
#                 display_names.append(name[:22] + "...")
#             else:
#                 display_names.append(name)
    
#     # Create bars with color coding - use consistent bar height for all bars
#     bar_height = 0.8  # Consistent bar height for all modalities
#     # ----- Color bars by subsystem -----
#     bar_colors = []
#     for raw_name in feature_names:
#         if modality == "perceptual":
#             key = _norm(_perceptual_short_name(raw_name))
#             subsystem = PERCEPTUAL_F2S.get(key, None)
#             color = perceptual_colors.get(subsystem, "#666666") if subsystem else "#666666"
#         else:
#             # acoustic feature names are usually already short (e.g., f0_mean)
#             key = _norm(raw_name)
#             subsystem = ACOUSTIC_F2S.get(key, None)
#             color = acoustic_colors.get(subsystem, "#666666") if subsystem else "#666666"

#         bar_colors.append(color)

#     bars = ax.barh(y_pos, importance_scores, height=bar_height, color=bar_colors, alpha=0.9)

#     import matplotlib.patches as mpatches

#     # Legend: only show subsystems that appear in this subplot
#     subsystems_in_plot = []
#     for raw_name in feature_names:
#         if modality == "perceptual":
#             ss = PERCEPTUAL_F2S.get(_norm(_perceptual_short_name(raw_name)))
#         else:
#             ss = ACOUSTIC_F2S.get(_norm(raw_name))
#         if ss and ss not in subsystems_in_plot:
#             subsystems_in_plot.append(ss)

#     if subsystems_in_plot:
#         cmap = perceptual_colors if modality == "perceptual" else acoustic_colors
#         handles = [mpatches.Patch(color=cmap[s], label=s) for s in subsystems_in_plot if s in cmap]
#         ax.legend(handles=handles, fontsize=12, loc="lower right", frameon=True)

#     # Customize the plot with larger font sizes
#     ax.set_yticks(y_pos)
#     # Set line spacing for multi-line labels to be tighter than bar spacing
#     ax.set_yticklabels(display_names, fontsize=25, linespacing=0.8)  # Tighter line spacing within labels
#     ax.set_xlabel('NMF Importance Score', fontsize=25)  # Increased from 11
#     ax.set_title(f'{modality.capitalize()} Cluster {cluster_id}\n({len(plot_data)} top features)', 
#                 fontsize=30, fontweight='bold', pad=15)  # Reduced padding from 35 to 15
    
#     # Add value labels on bars with larger font
#     for i, (bar, score) in enumerate(zip(bars, importance_scores)):
#         width = bar.get_width()
#         ax.text(width + 0.01 * max(importance_scores), bar.get_y() + bar.get_height()/2, 
#                 f'{score:.3f}', ha='left', va='center', fontsize=15)  # Increased from 8
    
#     # Invert y-axis so most important feature is at top and adjust limits for spacing
#     ax.invert_yaxis()
    
#     # Set y-axis limits with padding to accommodate increased spacing
#     if len(y_pos) > 0:
#         y_margin = bar_spacing * 0.5
#         ax.set_ylim(max(y_pos) + y_margin, min(y_pos) - y_margin)
    
#     # Set x-axis limits with some padding
#     if x_max is not None:
#         # Use consistent x-axis scale across all subplots
#         ax.set_xlim(0, x_max)
#     else:
#         # Use individual scaling for each subplot
#         ax.set_xlim(0, max(importance_scores) * 1.15)
    
#     # Add grid for better readability
#     ax.grid(axis='x', alpha=0.3, linestyle='--')
#     ax.set_axisbelow(True)
    
#     # Ensure labels don't get cut off by adjusting margins
#     ax.margins(y=0.02)  # Small margin for y-axis
    
#     # For better label visibility, especially on left subplots
#     if hasattr(ax, 'get_position'):
#         pos = ax.get_position()
#         # Add extra left margin for left column subplots (col 0)
#         if pos.x0 < 0.5:  # Left column
#             ax.tick_params(axis='y', pad=8)  # More padding for left column labels

# def create_modality_plot(data: dict, modality: str, save_path: str, run_name: str, consistent_scale: bool = False):
#     """
#     Create a 2x2 combined plot showing feature importance for a single modality (all 4 clusters).
    
#     Args:
#         data: Dictionary containing feature importance data for the modality
#         modality: 'acoustic' or 'perceptual'
#         save_path: Path to save the combined plot
#         run_name: Name of the run for the title
#         consistent_scale: Whether to use consistent x-axis scale across all subplots
#     """
#     # Create figure with subplots - larger height for perceptual plots due to multi-line labels
#     fig_height = 20 if modality == 'perceptual' else 16
#     fig, axes = plt.subplots(2, 2, figsize=(20, fig_height))
#     fig.suptitle(f'{modality.capitalize()} Feature Importance Analysis', 
#                 fontsize=25, fontweight='bold', y=0.95)
    
#     # Get all clusters for this modality (up to 4)
#     clusters = sorted(list(data.keys()))[:4]
    
#     # Calculate consistent x-axis scale if requested
#     x_max = None
#     if consistent_scale:
#         max_scores = []
#         for cluster_id in clusters:
#             cluster_data = data[cluster_id].head(10)  # Top 10 features
#             if not cluster_data.empty:
#                 max_scores.append(cluster_data['nmf_importance_score'].max())
#         if max_scores:
#             x_max = max(max_scores) * 1.15  # Add 15% padding
    
#     # Plot clusters in 2x2 grid
#     positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
#     for i, cluster_id in enumerate(clusters):
#         if i < 4:  # Only plot up to 4 clusters
#             row, col = positions[i]
#             create_feature_importance_subplot(
#                 axes[row, col], data[cluster_id], cluster_id, modality, x_max=x_max
#             )
            
#             # Additional spacing adjustments based on position
#             if col == 0:  # Left column - ensure labels don't get cut off
#                 axes[row, col].yaxis.set_label_coords(-0.15, 0.5)  # Move y-label further left
#             if col == 1:  # Right column - ensure labels don't overlap with left plots
#                 axes[row, col].tick_params(axis='y', pad=5)  # Less padding to keep labels closer
            
#             # Adjust title positioning for bottom row to prevent overlap with top row x-axes
#             if row == 1:  # Bottom row - move titles down more
#                 axes[row, col].title.set_position([0.5, -0.15])  # Move title further down
    
#     # Hide unused subplots if we have fewer than 4 clusters
#     for i in range(len(clusters), 4):
#         row, col = positions[i]
#         axes[row, col].set_visible(False)
    
#     # Adjust layout to prevent overlap with more spacing for larger fonts and labels
#     # Increase left margin to prevent left-side labels from being cut off
#     # Increase spacing between subplots to prevent right-side labels from overlapping left-side plots
#     # Increase vertical spacing to prevent bottom subplot titles from overlapping top subplot x-axes
#     plt.tight_layout(rect=[0.05, 0, 1, 0.90], h_pad=5.0, w_pad=8.0)

    
#     # Save the plot
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"{modality.capitalize()} feature importance plot saved to: {save_path}")
    
#     # Show plot if running interactively
#     try:
#         plt.show()
#     except:
#         pass  # Skip if not in interactive mode

# def main():
#     """Main execution function."""
#     parser = argparse.ArgumentParser(description='Create combined feature importance plots')
#     parser.add_argument('--run_number', type=int, help='Run number (e.g., 32 for run32)')
#     parser.add_argument('--run_dir', type=str, help='Full path to run directory')
#     parser.add_argument('--output_prefix', type=str, default='combined_feature_importance',
#                        help='Output filename prefix (default: combined_feature_importance)')
#     parser.add_argument('--consistent_scale', action='store_true',
#                        help='Use consistent x-axis scale across all subplots for each modality')
    
#     args = parser.parse_args()
    
#     # Determine run directory
#     if args.run_number:
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         run_dir = os.path.join(base_dir, "outputs", f"run{args.run_number}")
#         run_name = f"Run {args.run_number}"
#     elif args.run_dir:
#         run_dir = args.run_dir
#         run_name = os.path.basename(run_dir)
#     else:
#         parser.error("Must specify either --run_number or --run_dir")
    
#     # Validate run directory exists
#     if not os.path.exists(run_dir):
#         print(f"Error: Run directory does not exist: {run_dir}")
#         return 1
    
#     print(f"Processing {run_name}...")
#     print(f"Run directory: {run_dir}")
    
#     try:
#         # Load feature importance data
#         print("Loading feature importance data...")
#         data = load_feature_importance_data(run_dir)
        
#         # Display summary
#         print(f"Found data for:")
#         for modality in ['acoustic', 'perceptual']:
#             clusters = list(data[modality].keys())
#             print(f"  - {modality}: {len(clusters)} clusters ({clusters})")
        
#         # Create output directory
#         plots_dir = os.path.join(run_dir, "plots")
#         os.makedirs(plots_dir, exist_ok=True)
        
#         # Create separate plots for each modality
#         for modality in ['acoustic', 'perceptual']:
#             save_path = os.path.join(plots_dir, f"{args.output_prefix}_{modality}.png")
#             print(f"Creating {modality} plot...")
#             scale_info = " (consistent scale)" if args.consistent_scale else " (individual scales)"
#             print(f"  Using{scale_info}")
#             create_modality_plot(data[modality], modality, save_path, run_name, args.consistent_scale)
        
#         print("✓ Combined feature importance plots created successfully!")
#         print(f"  - Acoustic plot: {args.output_prefix}_acoustic.png")
#         print(f"  - Perceptual plot: {args.output_prefix}_perceptual.png")
        
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")
#         return 1
    
#     return 0

# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)




"""
Create Combined Feature Importance Plots

This script creates two separate 2x2 grids showing feature importance plots:
1. One plot for all 4 acoustic clusters
2. Another plot for all 4 perceptual clusters

Usage:
    python create_combined_feature_plots.py --run_number 32
    python create_combined_feature_plots.py --run_dir "outputs/run32"

Output:
    - combined_feature_importance_acoustic.png (2x2 grid of acoustic clusters)
    - combined_feature_importance_perceptual.png (2x2 grid of perceptual clusters)

Author: GitHub Copilot
Date: 2025
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches


# ---------------------------
# Subsystem dictionaries
# ---------------------------

perceptual_features = {
    "Phonatory": ["Strained Voice", "Overall Dysphonia"],
    "Respiratory": ["Loudness Decay", "Short Phrases", "Reduced Loudness"],
    "Prosody": ["Monopitch and/or Monoloudness", "Excess and Equal Stress", "Reduced Stress", "Overall Dysprosody"],
    "Resonatory": ["Hypernasality"],
    "Articulatory": ["Imprecise Articulation", "Slow Articulation Rate", "Reduced Overall Speaking Rate"],
}

perceptual_colors = {
    "Phonatory": "#8F0E10",
    "Respiratory": "#4C97D8",
    "Prosody": "#33A02C",
    "Resonatory": "#902FFF",
    "Articulatory": "#E7790B",
}

acoustic_features = {
    "Articulatory": [
        'f1', 'f2', "f1_d_dx_median", "f1_d_dx_prc_5","f1_d_dx_prc_95","f1_d_dx_prc_5_95",
        "f2_d_dx_median","f2_d_dx_prc_5","f2_d_dx_prc_95","f2_d_dx_prc_5_95",
        "F1_F2_comp",
        "speech_dur","mean_phrase_dur","cv_phrase_dur", "speech_rate","total_dur"
    ],
    "Phonatory": ["f0_mean","f0_std","CPP_mean","CPP_F1_comp","CPP_F2_comp"],
    "Respiratory": ["intensity_CV","percent_pause","num_pause","mean_pause_dur","cv_pause_dur","total_pause_dur"],
    "Prosody": [
        "fft_peaks1","fft_peaks2","fft_ampli1","fft_ampli2",
        "nrj_below_boundary","nrj_above_boundary","nrj_3_6","ratio_below_above"
    ],
    'Resonatory': ['F1width', 'f2_bw', 'A1P1', 'A1P1comp', 'P1amp']  # Added resonatory features for completeness
}

acoustic_colors = {
    "Articulatory": "#E7790B",
    "Phonatory": "#8F0E10",
    "Respiratory": "#4C97D8",
    "Prosody": "#33A02C",
    "Resonatory": "#902FFF",
}

# ---------------------------
# Helpers: robust matching
# ---------------------------

def _norm(s: str) -> str:
    """Normalize strings for matching (case/space/punct tolerant)."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else " ")
    return " ".join("".join(out).split())

def _perceptual_short_name(raw: str) -> str:
    """Extract short name from perceptual feature strings like '13. Strained Voice (....)'."""
    if raw is None:
        return ""
    raw = str(raw)
    if ". " in raw and " (" in raw:
        start_idx = raw.find(". ") + 2
        end_idx = raw.find(" (")
        return raw[start_idx:end_idx].strip()
    return raw.strip()

def _build_feature_to_subsystem(features_by_subsystem: dict) -> dict:
    """Inverse map: normalized_feature -> subsystem."""
    inv = {}
    for subsystem, feats in features_by_subsystem.items():
        for f in feats:
            inv[_norm(f)] = subsystem
    return inv

PERCEPTUAL_F2S = _build_feature_to_subsystem(perceptual_features)
ACOUSTIC_F2S = _build_feature_to_subsystem(acoustic_features)


def load_feature_importance_data(run_dir: str) -> dict:
    results_dir = os.path.join(run_dir, "results")
    feature_file = os.path.join(results_dir, "feature_importances_and_values.csv")

    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature importance file not found: {feature_file}")

    df = pd.read_csv(feature_file)

    data = {}
    for modality in ['acoustic', 'perceptual']:
        data[modality] = {}
        modality_data = df[df['modality'] == modality]

        for cluster_id in sorted(modality_data['cluster_id'].unique()):
            cluster_data = modality_data[modality_data['cluster_id'] == cluster_id]
            cluster_data = cluster_data.sort_values('feature_rank')
            data[modality][cluster_id] = cluster_data

    return data


def create_feature_importance_subplot(ax, cluster_data, cluster_id, modality, max_features=10, x_max=None):
    plot_data = cluster_data.head(max_features)

    bar_spacing = 1.5 if modality == 'perceptual' else 1.0
    y_pos = np.arange(len(plot_data)) * bar_spacing
    feature_names = plot_data['feature_name'].values
    importance_scores = plot_data['nmf_importance_score'].values

    # Display names
    display_names = []
    for name in feature_names:
        if modality == 'perceptual':
            short_name = _perceptual_short_name(name)
            if len(short_name) > 15:
                words = short_name.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    short_name = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
            display_names.append(short_name)
        else:
            display_names.append(name[:22] + "..." if len(name) > 25 else name)

    # Bar colors by subsystem
    bar_colors = []
    for raw_name in feature_names:
        if modality == "perceptual":
            key = _norm(_perceptual_short_name(raw_name))
            subsystem = PERCEPTUAL_F2S.get(key, None)
            color = perceptual_colors.get(subsystem, "#666666") if subsystem else "#666666"
        else:
            key = _norm(raw_name)
            subsystem = ACOUSTIC_F2S.get(key, None)
            color = acoustic_colors.get(subsystem, "#666666") if subsystem else "#666666"
        bar_colors.append(color)

    bars = ax.barh(y_pos, importance_scores, height=0.8, color=bar_colors, alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=25, linespacing=0.8)
    ax.set_xlabel('NMF Importance Score', fontsize=25)
    # Short modality label for paper-friendly titles
    mod_label = "A" if modality == "acoustic" else "P"

    ax.set_title(
        f'{mod_label}{cluster_id}\n({len(plot_data)} top features)',
        fontsize=30, fontweight='bold', pad=15
    )


    for bar, score in zip(bars, importance_scores):
        width = bar.get_width()
        ax.text(width + 0.01 * max(importance_scores), bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', ha='left', va='center', fontsize=15)

    ax.invert_yaxis()

    if len(y_pos) > 0:
        y_margin = bar_spacing * 0.5
        ax.set_ylim(max(y_pos) + y_margin, min(y_pos) - y_margin)

    ax.set_xlim(0, x_max if x_max is not None else max(importance_scores) * 1.15)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.margins(y=0.02)


def add_shared_legend(fig, modality: str):
    color_map = perceptual_colors if modality == "perceptual" else acoustic_colors

    # Alphabetically sort subsystem names
    sorted_subsystems = sorted(color_map.keys())

    handles = [
        mpatches.Patch(color=color_map[subsystem], label=subsystem)
        for subsystem in sorted_subsystems
    ]

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.78, 0.5),
        bbox_transform=fig.transFigure,
        fontsize=25,          # <-- increase item font
        frameon=True,
        title="Subsystem",
        title_fontsize=24     # <-- increase title font
    )





def create_modality_plot(data: dict, modality: str, save_path: str, run_name: str, consistent_scale: bool = False):
    fig_height = 20 if modality == 'perceptual' else 16
    fig, axes = plt.subplots(2, 2, figsize=(20, fig_height))
    fig.suptitle(f'{modality.capitalize()} Feature Importance Analysis',
                 fontsize=25, fontweight='bold', y=0.95)

    clusters = sorted(list(data.keys()))[:4]

    x_max = None
    if consistent_scale:
        max_scores = []
        for cluster_id in clusters:
            cluster_data = data[cluster_id].head(10)
            if not cluster_data.empty:
                max_scores.append(cluster_data['nmf_importance_score'].max())
        if max_scores:
            x_max = max(max_scores) * 1.15

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for i, cluster_id in enumerate(clusters):
        row, col = positions[i]
        create_feature_importance_subplot(axes[row, col], data[cluster_id], cluster_id, modality, x_max=x_max)

    # Add shared legend OUTSIDE
    add_shared_legend(fig, modality)

    # Leave room on the right for legend
    plt.tight_layout(rect=[0.05, 0, 0.75, 0.90], h_pad=5.0, w_pad=8.0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"{modality.capitalize()} feature importance plot saved to: {save_path}")

    try:
        plt.show()
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Create combined feature importance plots')
    parser.add_argument('--run_number', type=int, help='Run number (e.g., 32 for run32)')
    parser.add_argument('--run_dir', type=str, help='Full path to run directory')
    parser.add_argument('--output_prefix', type=str, default='combined_feature_importance')
    parser.add_argument('--consistent_scale', action='store_true')

    args = parser.parse_args()

    if args.run_number:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        run_dir = os.path.join(base_dir, "outputs", f"run{args.run_number}")
        run_name = f"Run {args.run_number}"
    elif args.run_dir:
        run_dir = args.run_dir
        run_name = os.path.basename(run_dir)
    else:
        parser.error("Must specify either --run_number or --run_dir")

    if not os.path.exists(run_dir):
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1

    print(f"Processing {run_name}...")
    print(f"Run directory: {run_dir}")

    try:
        data = load_feature_importance_data(run_dir)

        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for modality in ['acoustic', 'perceptual']:
            save_path = os.path.join(plots_dir, f"{args.output_prefix}_{modality}.png")
            create_modality_plot(data[modality], modality, save_path, run_name, args.consistent_scale)

        print("✓ Combined feature importance plots created successfully!")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
