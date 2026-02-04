# %% Merge Features and NasalityFeatures into ALLFeatures
# This script merges CSVs from Features and NasalityFeatures folders
# by matching filenames and concatenating columns horizontally.

import os
import pandas as pd
import glob

# Base directory
base_dir = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesConvertedALL441k/"

# Input folders
features_dir = base_dir + "Features/"
nasality_dir = base_dir + "NasalityFeatures/"

# Output folder
output_dir = base_dir + "ALLFeatures/"

# Option to select only specific nasality features (active by default)
select_nasality_features = True
selected_nasality_cols = ["F1width", "A1P1", "A1P1comp", "P1amp"]

# Columns to drop from the main features CSV
drop_features_cols = ["f1_bw"]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all CSV files from Features folder
features_files = glob.glob(features_dir + "*.csv")
features_files = [f.replace("\\", "/") for f in features_files]

print(f"Found {len(features_files)} files in Features folder")

# Track merge statistics
merged_count = 0
skipped_count = 0
skipped_files = []

for features_path in features_files:
    # Extract filename
    filename = os.path.basename(features_path)
    
    # Construct path to matching nasality file
    nasality_path = nasality_dir + filename
    
    # Check if matching nasality file exists
    if not os.path.isfile(nasality_path):
        print(f"WARNING: No matching nasality file for {filename} - skipping")
        skipped_count += 1
        skipped_files.append(filename)
        continue
    
    # Read both CSVs
    features_df = pd.read_csv(features_path, index_col=0)
    nasality_df = pd.read_csv(nasality_path, index_col=0)
    
    # Drop specified columns from features CSV
    cols_to_drop = [col for col in drop_features_cols if col in features_df.columns]
    if cols_to_drop:
        features_df = features_df.drop(columns=cols_to_drop)
    
    # Filter nasality features if option is enabled
    if select_nasality_features:
        available_cols = [col for col in selected_nasality_cols if col in nasality_df.columns]
        missing_cols = [col for col in selected_nasality_cols if col not in nasality_df.columns]
        if missing_cols:
            print(f"  Note: Missing columns in {filename}: {missing_cols}")
        nasality_df = nasality_df[available_cols]
    
    # Reset indices to ensure proper concatenation
    features_df = features_df.reset_index(drop=True)
    nasality_df = nasality_df.reset_index(drop=True)
    
    # Merge by concatenating columns (axis=1)
    merged_df = pd.concat([features_df, nasality_df], axis=1)
    
    # Save to output folder
    output_path = output_dir + filename
    merged_df.to_csv(output_path)
    
    merged_count += 1
    print(f"Merged: {filename} ({len(features_df.columns)} + {len(nasality_df.columns)} = {len(merged_df.columns)} columns)")

print("\n" + "="*50)
print(f"Merge complete!")
print(f"Successfully merged: {merged_count} files")
print(f"Skipped (no matching nasality file): {skipped_count} files")

if skipped_files:
    print("\nSkipped files:")
    for f in skipped_files:
        print(f"  - {f}")

print(f"\nOutput saved to: {output_dir}")
