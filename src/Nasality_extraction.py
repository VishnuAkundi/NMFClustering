# %% Import libraries

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import pipeline_analytics as pa
from pipeline_analytics.nasality import get_nasality
from pipeline_analytics.utils import get_segments
import librosa
import glob


print('Welcome to Bamboo Pipeline for Nasality Feature Extraction!')
print('This script will extract nasality features from the audio files in the specified directory.')

# Set input directory - same as Feature_extraction.py
directory = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesConvertedALL441k/"
# directory = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesLeftConverted441k/"
# directory = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesTest/"

# Get all WAV files
all_files = glob.glob(directory + "*.wav")
all_files = [f.replace("\\", "/") for f in all_files]

print(f"Found {len(all_files)} files")
print(all_files)

# %% Extract nasality features

for file in all_files:
    
    ext = os.path.splitext(file)[1]
    
    if ext == ".wav":
        type_ = "audio"
    elif (ext == ".webm") or (ext == ".mp4"):
        type_ = "video"
    
    print('\n\n\n')
    print(file)
    print(type_)
    print('\n\n\n')
    
    if ("PSG" in file) & (type_ == "audio"):
        
        new_fname = file.split("/")[-1].split(".")[0]
        # Output to NasalityFeatures subfolder
        new_fname_csv = directory + f"NasalityFeatures/{new_fname}.csv"

        # Skip if already processed
        if os.path.isfile(new_fname_csv):
            print(f"Skipping {new_fname} - already processed")
            continue
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(new_fname_csv), exist_ok=True)
        
        # Load audio - same approach as Feature_extraction.py
        sr0, y0 = wavfile.read(file)
        y, sr = librosa.load(file, sr=sr0)
        print("Sampling rate:", sr)
        
        # Adaptive energy threshold for segmentation - same logic as Feature_extraction.py
        energy_threshold = 0.01
        segments = get_segments(file, energy_threshold=energy_threshold)
        
        if len(segments) < 5:
            # Case: threshold was too high (too many islands covered up)
            segments = get_segments(file, energy_threshold=energy_threshold / 10)
        elif len(segments) > 25:
            # Case: threshold was too low (too many islands visible)
            segments = get_segments(file, energy_threshold=energy_threshold * 10)
        
        if (len(segments) >= 5) & (len(segments) <= 25):
            pass
        else:
            print(f"Skipping {new_fname} - segment count out of range: {len(segments)}")
            continue
        
        print(f"Found {len(segments)} segments")
        
        seg_dfs = []
        print("SEGMENTS:", segments)
        for seg in segments:
            start = seg[1]
            finish = seg[2]
            
            # Read audio and slice - same as analyze_sentence_acoustic
            sr_wav, y_full = wavfile.read(file)
            y_segment = y_full[int(sr_wav * start) : int(sr_wav * finish)]
            
            try:
                # Call get_nasality with sliced segment - same pattern as analyze_sentence_acoustic
                # Use skip_textgrid=True since we already have pre-segmented audio from get_segments()
                nasality_df = get_nasality(
                    file=y_segment,
                    sr=sr_wav,
                    sent_start=start,
                    sent_finish=finish,
                    min_pitch=60,
                    max_pitch=400,
                    skip_textgrid=True,  # Skip TextGrid sub-segmentation for pre-segmented audio
                )

                print(nasality_df)
                seg_dfs.append(nasality_df)
            except Exception as e:
                print(f"Error processing segment {start:.2f}-{finish:.2f}: {e}")
                continue
        
        if len(seg_dfs) == 0:
            print(f"No valid segments for {new_fname}")
            continue
        
        # Concatenate all segment dataframes
        nasality_df_all = pd.concat(seg_dfs)
        
        # Average across all segments to get one row - same as Feature_extraction.py
        nasality_df_final = pd.DataFrame(nasality_df_all.mean(axis=0)).T
        
        # Save to CSV
        nasality_df_final.to_csv(new_fname_csv)
        print(f"Saved nasality features to {new_fname_csv}")
    
    else:
        print("File type not known or not a PSG file!")

print("\n\nNasality extraction complete!")
