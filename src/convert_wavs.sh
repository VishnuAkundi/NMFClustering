#!/bin/bash

SRC="C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesLeft"
DST="C:/Users/akundivi/Desktop/Acoustic_Pipeline_New/VirtualSLP-analytics/Clustering/WavFilesLeftConverted441k"

mkdir -p "$DST"

for f in "$SRC"/*.wav; do
    filename=$(basename "${f%.wav}")
    target="$DST/${filename}_converted.wav"

    # Use ffprobe to check audio properties
    channels=$(ffprobe -v error -show_entries stream=channels -of default=noprint_wrappers=1:nokey=1 "$f")
    sample_rate=$(ffprobe -v error -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$f")
    bit_depth=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_fmt -of default=noprint_wrappers=1:nokey=1 "$f")

    if [ "$channels" -eq 1 ] && [ "$sample_rate" -eq 44100 ] && [ "$bit_depth" = "s16" ]; then
        echo "Copying $filename.wav to destination (Already in correct format)..."
        cp "$f" "$target"
    else
        echo "Converting $filename.wav to target format..."
        ffmpeg -y -i "$f" -ac 1 -ar 44100 -sample_fmt s16 "$target"
    fi
done
