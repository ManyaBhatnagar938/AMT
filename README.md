# Automatic Music Transcription: Benchmarking and Evaluation

A comparative evaluation of automatic music transcription (AMT) systems using controlled musical simulations, with a focus on note detection, pitch recognition, and temporal accuracy.

## Overview

Automatic music transcription aims to convert audio recordings into symbolic musical representations, including note pitches, onset times, and offset times. Although existing systems can perform well on conventional recordings, their accuracy can vary depending on the audio characteristics and the timing of consecutive notes.

This project benchmarks three approaches to pitch recognition and automatic music transcription:

- **Librosa**
- **CREPE**
- **Basic Pitch**

The systems are evaluated on a controlled dataset of 50 simulated monophonic note sequences with known ground-truth annotations. By varying musical patterns, note durations, and inter-note intervals, the project investigates how different systems perform under comparable conditions.

The benchmarking results reveal that high note-detection accuracy does not necessarily translate into accurate temporal boundaries, and that different systems exhibit distinct strengths and limitations.

## Objectives

- Compare the note-detection and temporal accuracy of Librosa, CREPE, and Basic Pitch.
- Evaluate the effect of different hop sizes on transcription performance.
- Examine errors in predicted note onsets, offsets, durations, and inter-note intervals.
- Investigate how the spacing between consecutive notes relates to transcription errors.
- Establish a quantitative basis for further research into the behaviour of AMT systems.

## Dataset and Simulation Design

A dataset of **50 simulated monophonic audio sequences** was created using 10 musical patterns, each with five variations.

The patterns include diatonic and chromatic scales, alternating scale patterns, structured note sequences, and randomly selected notes.

The five variations cover different combinations of note order, note duration, and inter-note spacing:

1. Ascending notes with equal durations and no gaps.
2. Ascending notes with random durations and no gaps.
3. Ascending notes with equal durations and random gaps.
4. Ascending notes with random durations and random gaps.
5. Descending notes with equal durations and no gaps.

Each simulation has corresponding audio, MIDI, and ground-truth annotation files. The annotations provide a common reference for comparing the predicted note events against the intended musical sequence.

## Benchmarking Methodology

### Systems Evaluated

| System | Configurations |
|---|---|
| Librosa | Hop sizes of 5 ms, 10 ms, and 15 ms |
| CREPE | Hop sizes of 5 ms, 10 ms, and 15 ms, with and without post-processing |
| Basic Pitch | Standard transcription output |

For Librosa and CREPE, frame-wise pitch predictions were converted into note events using note-segmentation procedures. The resulting predictions were formatted consistently with the ground-truth annotations to support direct comparison.

### Evaluation Metrics

The benchmark evaluates performance across the following dimensions:

- **Note-detection precision:** The proportion of predicted note events that correspond to valid note events.
- **Onset MAE:** Mean absolute error in predicted note start times.
- **Offset MAE:** Mean absolute error in predicted note end times.
- **Duration MAE:** Mean absolute error in estimated note durations.
- **Inter-note interval MAE:** Mean absolute error in the gaps between consecutive notes.
- **Signed offset error:** Used to identify systematic early or late note-ending predictions.

## Results

### 1. Note Detection

Basic Pitch and post-processed CREPE achieved a note-detection precision of **1.00** on the simulated benchmark.

Other configurations achieved the following approximate precision values:

| System configuration | Precision |
|---|---:|
| Basic Pitch | 1.00 |
| CREPE with post-processing | 1.00 |
| Librosa, 10 ms | 0.99 |
| Librosa, 15 ms | 0.99 |
| Librosa, 5 ms | 0.98 |
| CREPE, 5 ms | 0.89 |
| CREPE, 10 ms | 0.85 |
| CREPE, 15 ms | 0.77 |

These results show that performance varied across systems and configurations, with the tested CREPE configurations particularly sensitive to hop size.

### 2. Temporal Accuracy

Basic Pitch achieved the lowest onset mean absolute error, at approximately **0.0064 seconds**. However, its note-offset error was substantially larger than that of the other systems.

| Metric | Basic Pitch |
|---|---:|
| Onset MAE | 0.0064 s |
| Offset MAE | 0.037 s |
| Signed offset bias | −0.0366 s |
| Duration MAE | 0.034 s |
| Inter-note interval MAE | 0.035 s |

The negative signed offset bias indicates that Basic Pitch tended to predict note endings earlier than their ground-truth endpoints.

This distinction is important: a system can identify the correct pitches and note beginnings while still producing inaccurate note durations and gaps between notes.

### 3. Differences Across Systems

The benchmark did not identify one system as uniformly superior across all evaluation metrics.

- **Basic Pitch** demonstrated high note-detection precision and accurate onset estimates, but larger errors in note offsets, durations, and inter-note intervals.
- **CREPE** exhibited different detection and timing behaviour across hop sizes, while post-processing substantially changed its note-event estimates.
- **Librosa** showed comparatively small offset errors across its tested configurations, although its onset errors were larger than those of Basic Pitch.

These findings highlight the importance of evaluating temporal accuracy alongside note-detection performance when comparing AMT systems.

### 4. Inter-Note Interval and Offset Error

The initial simulations revealed a relationship between the interval following a note and the error in its predicted endpoint. Shorter gaps were generally associated with larger early-offset errors in Basic Pitch.

A separate controlled experiment was designed using **26 simulations**, with inter-note intervals ranging from 0.001 to 0.2 seconds.

The analysis showed a strong negative correlation between inter-note interval and end-time RMSE (**Pearson's r = −0.85**). A simple linear model explained approximately **73% of the variation** in offset RMSE.

These results establish a strong relationship between note spacing and the observed offset errors in the tested simulations. They motivate further investigation into the acoustic information that influences note-boundary predictions.

## Ongoing Work

Further research is being conducted on explainable AI (XAI) to investigate limitations identified during benchmarking and to better understand the behaviour of AMT systems under different acoustic conditions.

This work explores how model predictions relate to the input audio, with the broader aim of understanding the factors contributing to transcription errors. The XAI investigations are ongoing and are not included in the current public benchmarking release.

## Limitations

The benchmarking results are based on a controlled dataset of 50 simulated monophonic sequences. They should not be interpreted as a comprehensive ranking of AMT systems across all musical genres, instruments, or recording conditions.

The controlled interval analysis identifies a strong relationship between inter-note spacing and offset error, but the correlation alone does not establish the underlying mechanism responsible for the observed behaviour.

Further evaluation on diverse recordings and acoustic conditions is necessary to assess how these findings generalise beyond the simulated dataset.

## Repository Contents

The repository contains the benchmarking implementation, simulation-related files, evaluation outputs, and supporting figures where available.

The exact file structure and execution instructions will be documented alongside the corresponding code and dependencies.

## Technologies

- Python
- Librosa
- CREPE
- Basic Pitch
- NumPy and Pandas
- Matplotlib

## Future Directions

Future work will investigate the factors contributing to the transcription limitations observed during benchmarking, including the relationship between acoustic input characteristics and model predictions. The broader goal is to develop a clearer understanding of AMT system behaviour and identify potential avenues for improving transcription reliability.

## Acknowledgements

This project was conducted as part of ongoing research into automatic music transcription, model evaluation, and explainability.
