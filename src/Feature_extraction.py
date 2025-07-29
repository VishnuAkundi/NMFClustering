# %% Import libraries

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import pipeline_analytics as pa
from pipeline_analytics.pipeline_constructor import (
    analyze_sentence_acoustic,
    analyze_phonation_acoustic,
    analyze_ddk_acoustic,
    analyze_passage_acoustic,
    analyze_sentence_kinematics,
    analyze_passage_kinematics,
)
from pipeline_analytics.utils import get_segments
from pipeline_analytics.speech_pause_analysis import spa
from pipeline_analytics.prosody import get_EMS_parameters
import argparse
import glob





print('Welcome to Bamboo Pipeline for Feature Extraction!')
print('This script will extract acoustic features from the audio files in the specified directory.')
directory = "C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesLeftConverted441k/"

all_files = glob.glob(directory + "*.wav")
# all_files = glob.glob("C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesTest/*.wav")

all_files = [f.replace("\\", "/") for f in all_files]

print(len(all_files))
print(all_files)

# %% Extract acoustic features
# file='C:/Users/leifs/Documents/Taati & Yunusova labs/Projects/Clustering paper/Redux/Audio/ALS_MIBD09_PSG_BAMBOO_20220909_130614_r.wav'





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
    
    # if (("SENT" in file)|("BBP" in file)) & (type_ == "audio"):
    #     segments = get_segments(file)
    #     seg_dfs = []
    
    #     for seg in segments:
    #         start = seg[1]
    #         finish = seg[2]
    #         sentence_acoustic_df = analyze_sentence_acoustic(
    #             file, start, finish, min_pitch=60, max_pitch=400, pitch_hopsize=882
    #         )
    #         seg_dfs.append(sentence_acoustic_df)
    
    #     sentence_acoustic_df_final = pd.concat(seg_dfs)
    #     print(sentence_acoustic_df_final)
    
    #     # sentence_acoustic_df_final.to_csv("C:/Users/leifs/df.csv")
    
    if ("PSG" in file) & (type_ == "audio"):
        
        new_fname = file.split("/")[-1].split(".")[0]
        # new_fname_csv = f"Features/{new_fname}.csv"
        new_fname_csv = directory + f"Features/{new_fname}.csv"
        # new_fname_csv = f"C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesTest/Features/{new_fname}.csv"

        if os.path.isfile(new_fname_csv):
            continue
        # TODO compute within segments - artic/formant features etc. and pitch
        # TODO compute globally - SPA and prosody
        # sr, y = wavfile.read(file)
        
        # import matplotlib.pyplot as plt
        import librosa
        sr0, y0 = wavfile.read(file)
        y, sr = librosa.load(file, sr=sr0)
        # y, sr = librosa.load(file)
        print("Sampling in Feature_extraction.py",sr)
        # plt.plot(y*32768)
    
        # Add conditional control to automagically adjust the energy threshold for segmentation - this is a critical step in the process
        # and needs to be correct! Or else everything else is meaningless and will probably break
        # To understand the logic here, think of the energy_threshold as the sea level and the identified
        # segments as islands. Too high of an energy_threshold means most islands will be covered up. Too low of an energy_threshold,
        # and the seabed will be visible. This also operates on the assumption that there is a conventional number of phrases in a given passage
        # like Bamboo. For that one, it would be implausible to have fewer than 5 segments (there are 8 sentences) and also implausible
        # to have more than say 16 (assuming on average one mid-sentence break per sentence for very impaired speakers)
        # EDIT: one of our impaired speakers had about 19 well-characterized segments in an 82-sec recording. Upping high threshold to 25
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
            continue
    
        seg_dfs = []

        # print("FINAL SEGMENTS FOUND:", segments)
    
        for seg in segments:
            start = seg[1]
            finish = seg[2]
            psg_acoustic_df = analyze_passage_acoustic(
                file, start, finish, min_pitch=60, max_pitch=400, pitch_hopsize=882
            )
            seg_dfs.append(psg_acoustic_df)
    
        psg_acoustic_df_final = pd.concat(seg_dfs)
        # print(psg_acoustic_df_final)
    
        # # TODO perhaps necessary to optimize the choice of silence threshold? Empirically, -50 seems decent for SPA...
        # srs = []
        # for i in np.arange(-100, -20)[::-1]:
        #     spa_df = spa(file=y, sr=sr, silence_threshold=i, min_dip=1.5, min_pause_duration=0.3, input_source="scipy")
        #     srs.append(spa_df['speech_rate'])
        # plt.figure(); plt.plot(srs)
        spa_df = spa(
            file=y,
            sr=sr,
            silence_threshold=-50,
            min_dip=1.5,
            min_pause_duration=0.3,
            # input_source="scipy",
            input_source="librosa",
        )
        prosody_df = get_EMS_parameters(y, sr)
    
        psg_acoustic_df_final = pd.concat([pd.DataFrame(psg_acoustic_df_final.mean(axis=0)).T, spa_df, prosody_df], axis=1)
        psg_acoustic_df_final.to_csv(new_fname_csv)
    
    # elif ("VWL" in file) & (type_ == "audio"):
    #     segments = get_segments(file)
    #     seg_dfs = []
    
    #     # TODO do we want to aggregate over segments, or simply pick the first/second one?
    #     for seg in segments:
    #         start = seg[1]
    #         finish = seg[2]
    #         phonation_acoustic_df = analyze_phonation_acoustic(
    #             file, start, finish, min_pitch=60, max_pitch=400, pitch_hopsize=882
    #         )
    #         seg_dfs.append(phonation_acoustic_df)
    
    #     phon_df = pd.concat(seg_dfs)
    #     print(phon_df)
    
    # elif ("DDK" in file) & (type_ == "audio"):
    #     ddk_df = analyze_ddk_acoustic(file)
    
    # elif ("SENT" in file) & (type_ == "video"):
    #     sentence_kinematic_df_final = analyze_sentence_kinematics(file)
    
    # elif ("PSG" in file) & (type_ == "video"):
    #     passage_kinematic_df_final = analyze_passage_kinematics(file)
    
    else:
        print("File type not known!")
    

