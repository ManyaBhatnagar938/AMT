'''Code Structure : This is the entire code for the analysis of the different audio simulations generated through the different toolboxes. 
It follows the given structure : 
1. Functions to locate the ground truth .wav files in their given folder and save any given dataframe into a .csv file at the desired location.
2. The functions for conversions from frequency to midi number, midi to Note Code (C, C#...), Note mapping dictionary (Note code to Note) and midi to Note (Sa, Re...)
3. The Note Segmentation function - Crepe, Librosa, MIR Toolbox and Essentia all output a frame wise frequency prediction, though the methods they use for both pitch prediction
and frame calculations are different. The raw unformatted CSV is saved but the resulting data is also converted into a csv that follows our format for ease of comparison using
this function.
4.Seperate Analysis functions for all the 5 toolboxes. Apart from Basic Pitch, the other toolboxes use frame wise frequency prediction which is why the structure and the 
functions used in these are similar to each other. 
5. Running the pipeline on the original 50 monophonic simulations that were generated and saving the results in the same folders as the ground truths.'''

#Import Block
import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import crepe
from basic_pitch.inference import predict
from basic_pitch.inference import predict_and_save
import pretty_midi 
#import essentia.standard as es
#import matlab.engine

#Functions for Accessing and Saving Files

def get_wav_file(simulation_folder):
    '''input : path to simulation folder where the wanted file is stored
       output : the wanted .wav file'''
    simulation_folder = Path(simulation_folder)
    wav_files = list(simulation_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV file found in {simulation_folder}")
    return wav_files[0]

def save_csv(dataframe, output_folder, filename):
    '''input : the dataframe needed to be converted into an csv
       output_folder: the folder where the csv needs to be saved
       filename: the file name the csv needs to be given. '''
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / filename
    dataframe.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


#Note mapping dictionary, frequency to midi, midi to note code
NOTE_MAPPING = {
    "C4": ("Sa"),
    "C#4": ("Komal Re"),
    "D4": ("Re"),
    "D#4": ("Komal Ga"),
    "E4": ("Ga"),
    "F4": ("Ma"),
    "F#4": ("Tivra Ma"),
    "G4": ("Pa"),
    "G#4": ("Komal Dha"),
    "A4": ("Dha"),
    "A#4": ("Komal Ni"),
    "B4": ("Ni"),
    "C5": ("Sa")
}

def freq_to_midi(freq):
    if freq == 0:
        return 0
    return 69 + 12 * np.log2(freq / 440)

def midi_to_note_code(midi):
    if midi == 0:
        return None
    return pretty_midi.note_number_to_name(midi)

def midi_to_note(midi):
    if midi == 0:
        return None
    return NOTE_MAPPING[midi_to_note_code(midi)]

def note_segmentation(times, frequencies, confidences, confidence_threshold=0.6):

    """Converts frame-wise pitch estimates into note events.
    Parameters :
    times : array-like
        Timestamp of each frame (seconds).
    frequencies : array-like
        Frequency estimate (Hz) for each frame.
    confidences : array-like
        Confidence for each frame.
    confidence_threshold : float
        Frames below this threshold are treated as silence/intervals.
    Returns:
    pandas.DataFrame
        Formatted note-level dataframe."""

    rows = []
    in_note = False
    in_interval = False

    note_start = None
    interval_start = None

    current_frequencies = []
    current_confidences = []

    previous_midi = None
    previous_note = None
    previous_note_code = None

    previous_end_time = 0.0

    for t, freq, conf in zip(times, frequencies, confidences):

        # CASE 1 : Confidence below threshold -> Interval
        if conf < confidence_threshold:

            if in_note:

                note_end = t
                avg_frequency = np.mean(current_frequencies)
                avg_confidence = np.mean(current_confidences)

                rows.append({
                    "Note": previous_note,
                    "Note Code": previous_note_code,
                    "Frequency (Hz)": avg_frequency,
                    "MIDI": previous_midi,
                    "Start Time (s)": note_start,
                    "End Time (s)": note_end,
                    "Velocity": avg_confidence,
                    "interval_after (s)": None
                })

                previous_end_time = note_end
                current_frequencies = []
                current_confidences = []

                in_note = False

            if not in_interval:
                interval_start = t
                in_interval = True

            continue

        # Confidence is back above threshold
        current_midi = int(round(freq_to_midi(freq)))
        current_note = midi_to_note(current_midi)
        current_note_code = midi_to_note_code(current_midi)

        # CASE 2 : Starting first note
        if not in_note:
            if in_interval and len(rows) > 0:
                rows[-1]["interval_after (s)"] = t - previous_end_time

            in_interval = False
            note_start = t

            current_frequencies = [freq]
            current_confidences = [conf]

            previous_midi = current_midi
            previous_note = current_note
            previous_note_code = current_note_code

            in_note = True
            continue

        # CASE 3 : Same note
        if current_midi == previous_midi:
            current_frequencies.append(freq)
            current_confidences.append(conf)

            continue

        # CASE 4 : MIDI changed -> New note
        note_end = t
        avg_frequency = np.mean(current_frequencies)
        avg_confidence = np.mean(current_confidences)

        rows.append({
            "Note": previous_note,
            "Note Code": previous_note_code,
            "Frequency (Hz)": avg_frequency,
            "MIDI": previous_midi,
            "Start Time (s)": note_start,
            "End Time (s)": note_end,
            "Velocity": avg_confidence,
            "interval_after (s)": 0.0
        })

        previous_end_time = note_end

        note_start = t
        current_frequencies = [freq]
        current_confidences = [conf]

        previous_midi = current_midi
        previous_note = current_note
        previous_note_code = current_note_code

    # Save final note
    if in_note:
        avg_frequency = np.mean(current_frequencies)
        avg_confidence = np.mean(current_confidences)

        rows.append({
            "Note": previous_note,
            "Note Code": previous_note_code,
            "Frequency (Hz)": avg_frequency,
            "MIDI": previous_midi,
            "Start Time (s)": note_start,
            "End Time (s)": times[-1],
            "Velocity": avg_confidence,
            "interval_after (s)": None
        })

    return pd.DataFrame(rows)


#Basic Pitch Analysis Function
def basic_pitch_analysis(simulation_folder):

    """Runs Basic Pitch on the WAV file inside a simulation folder.
    Saves:
        SimulationName_BasicPitch.mid
        SimulationName_BasicPitch.csv"""

    # Locate WAV
    wav_path = get_wav_file(simulation_folder)
    simulation_name = Path(simulation_folder).name
    print(f"\nRunning Basic Pitch on {simulation_name}")

    model_output, midi_data, note_events = predict(str(wav_path))
    note_events = sorted(note_events, key=lambda x: x[0])

    midi_filename = f"{simulation_name}_BasicPitch.mid"
    midi_path = Path(simulation_folder) / midi_filename
    midi_data.write(str(midi_path))

    # Read generated MIDI to obtain velocities
    midi_file = pretty_midi.PrettyMIDI(str(midi_path))
    midi_notes = midi_file.instruments[0].notes

    # Convert Note Events
    rows = []

    for i, (note, midi_note) in enumerate(zip(note_events, midi_notes)):
        start_time = round(note[0], 4)
        end_time = round(note[1], 4)
        duration = round(end_time - start_time, 4)

        midi = int(round(note[2]))
        velocity = midi_note.velocity

        frequency = round(pretty_midi.note_number_to_hz(midi), 2)

        note_name = midi_to_note(midi)
        note_code = midi_to_note_code(midi)

        if i < len(note_events) - 1:
            interval_after = round(
                note_events[i + 1][0] - end_time,
                4
            )
        else:
            interval_after = None

        rows.append({

            "Note": note_name,
            "Note Code": note_code,
            "MIDI": midi,
            "Frequency (Hz)": frequency,
            "Start Time (s)": start_time,
            "End Time (s)": end_time,
            "Duration (s)": duration,
            "interval_after (s)": interval_after,
            "Velocity": velocity

        })

    df = pd.DataFrame(rows)

    csv_filename = f"{simulation_name}_BasicPitch.csv"
    save_csv(df,simulation_folder,csv_filename)
    print("Basic Pitch analysis complete.")


    #Crepe Analysis Block
def crepe_analysis(simulation_folder):

    """Runs CREPE on the WAV file inside a simulation folder.
    Performs analysis using step sizes:
        - 5 ms
        - 10 ms
        - 15 ms
    Saves for each step size:
        SimulationName_CREPE_5ms_Raw.csv
        SimulationName_CREPE_5ms_Formatted.csv
    """

    wav_path = get_wav_file(simulation_folder)
    simulation_name = Path(simulation_folder).name
    print(f"\nRunning CREPE on {simulation_name}")

    audio, sr = librosa.load(wav_path,sr=16000,mono=True)

    # Running for each step size
    for step_size in [5, 10, 15]:
        print(f"  Step Size : {step_size} ms")

        
        # Running CREPE
        times, frequencies, confidences, activations = crepe.predict(audio,sr,step_size=step_size,viterbi=True)

        # Saving Raw Output
        raw_df = pd.DataFrame({"Time": times,"Frequency": frequencies,"Confidence": confidences})

        raw_filename = (f"{simulation_name}"f"_CREPE_{step_size}ms_Raw.csv")
        save_csv(raw_df,simulation_folder,raw_filename)

        # Note Segmentation
        formatted_df = note_segmentation(times,frequencies,confidences)
        formatted_df["Duration (s)"] = (formatted_df["End Time (s)"]- formatted_df["Start Time (s)"])
        formatted_df = formatted_df[["Note","Note Code","MIDI","Frequency (Hz)","Start Time (s)","End Time (s)","Duration (s)","interval_after (s)","Velocity"]]

        formatted_filename = (f"{simulation_name}"f"_CREPE_{step_size}ms_Formatted.csv")
        save_csv(formatted_df,simulation_folder,formatted_filename)
    print("CREPE analysis complete.")


    #Librosa Analysis Block
def librosa_analysis(simulation_folder):

    """Runs Librosa pYIN on the WAV file inside a simulation folder.
    Performs analysis using hop sizes:
        - 5 ms
        - 10 ms
        - 15 ms
    Saves:
        SimulationName_Librosa_5ms_Raw.csv
        SimulationName_Librosa_5ms_Formatted.csv
    """
    wav_path = get_wav_file(simulation_folder)
    simulation_name = Path(simulation_folder).name
    print(f"\nRunning Librosa on {simulation_name}")

    audio, sr = librosa.load(wav_path,sr=None,mono=True)

    for hop_ms in [5, 10, 15]:
        print(f"  Hop Size : {hop_ms} ms")
        hop_length = int(sr * hop_ms / 1000)

    
        # Running pYIN
        f0, voiced_flag, voiced_probability = librosa.pyin(audio,fmin=librosa.note_to_hz('C3'),fmax=librosa.note_to_hz('C6'),sr=sr,hop_length=hop_length)
        times = librosa.times_like(f0,sr=sr,hop_length=hop_length)
        frequencies = np.nan_to_num(f0, nan=0.0)
        confidences = np.nan_to_num(voiced_probability,nan=0.0)

        raw_df = pd.DataFrame({"Time": times,"Frequency": frequencies,"Confidence": confidences})
        raw_filename = (f"{simulation_name}"f"_Librosa_{hop_ms}ms_Raw.csv")
        save_csv(raw_df,simulation_folder,raw_filename)

        formatted_df = note_segmentation(times,frequencies,confidences)
        formatted_df["Duration (s)"] = (formatted_df["End Time (s)"]- formatted_df["Start Time (s)"])
        formatted_df = formatted_df[["Note","Note Code","MIDI","Frequency (Hz)","Start Time (s)","End Time (s)","Duration (s)","interval_after (s)","Velocity"]]


        formatted_filename = (f"{simulation_name}"f"_Librosa_{hop_ms}ms_Formatted.csv")
        save_csv(formatted_df,simulation_folder,formatted_filename)
    print("Librosa analysis complete.")

#Complete analysis function for the three toolboxes
def run_analysis(simulation_folder):

   
    # Basic Pitch
    try:
        basic_pitch_analysis(simulation_folder)
        print("Basic Pitch completed.\n")

    except Exception as e:
        print(f"Basic Pitch failed:\n{e}\n")

    '''#CREPE
    try:
        crepe_analysis(simulation_folder)
        print("CREPE completed.\n")

    except Exception as e:
        print(f"CREPE failed:\n{e}\n")

    #Librosa
    try:
        librosa_analysis(simulation_folder)
        print("Librosa completed.\n")

    except Exception as e:
        print(f"Librosa failed:\n{e}\n")'''

    print("Finished.")

'''tracker = 0
Intervals = [0.0045, 0.004, 0.0037, 0.0035, 0.003, 0.0027, 0.0025, 0.002, 0.0017, 0.0015, 0.001]
#Durations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
for i in Intervals:
        wav_folder = f"C:\\Users\\PSYLAB03\\Desktop\\Manya\\TestSimulations_Interval\\I{i}"
        run_analysis(wav_folder)
        tracker = tracker + 1
        print("Completed analysis for", tracker, "simulation/s")'''


'''for j in Durations:
    wav_folder = f"C:\\Users\\PSYLAB03\\Desktop\\Manya\\TestSimulations_Duration\\{j}"
    run_analysis(wav_folder)
    tracker = tracker + 1
    print("Completed analysis for", tracker, "simulation/s")'''



'''def read_basicpitch_midi(simulation_folder):

    simulation_folder = Path(simulation_folder)

    sim_name = simulation_folder.name

    midi_file = simulation_folder / f"{sim_name}_BasicPitch.mid"

    if not midi_file.exists():
        raise FileNotFoundError(
            f"Basic Pitch MIDI not found:\n{midi_file}"
        )

    pm = pretty_midi.PrettyMIDI(str(midi_file))

    print(f"\nReading: {midi_file.name}\n")

    for instrument_num, instrument in enumerate(pm.instruments):

        print(f"Instrument {instrument_num}:")
        print(f"Name    : {instrument.name}")
        print(f"Program : {instrument.program}")
        print(f"Drum    : {instrument.is_drum}\n")

        for i, note in enumerate(instrument.notes, start=1):

            print(
                f"Note {i:02d} | "
                f"Pitch={note.pitch} | "
                f"Velocity={note.velocity} | "
                f"Start={note.start:.6f} | "
                f"End={note.end:.6f} | "
                f"Duration={note.end-note.start:.6f}")

import matplotlib.pyplot as plt
def inspect_note_probabilities(
    wav_path,
    gt_csv,
    note_index=0
):

    model_output, midi_data, note_events = predict(wav_path)

    note_probs = model_output["note"]
    onset_probs = model_output["onset"]
    contour_probs = model_output["contour"]

    print("Shapes")
    print("Note    :", note_probs.shape)
    print("Onset   :", onset_probs.shape)
    print("Contour :", contour_probs.shape)

    # Read GT
    gt = pd.read_csv(gt_csv)

    gt_row = gt.iloc[note_index]

    # Read BP MIDI
    bp_note = midi_data.instruments[0].notes[note_index]

    midi_pitch = bp_note.pitch

    # Convert MIDI pitch to note-bin index
    pitch_bin = midi_pitch - 21

    note_curve = note_probs[:, pitch_bin]
    onset_curve = onset_probs[:, pitch_bin]

    fps = 86  # Basic Pitch annotation FPS

    time = np.arange(len(note_curve)) / fps

    plt.figure(figsize=(15,5))

    plt.plot(
        time,
        note_curve,
        label="Frame probability",
        linewidth=2
    )

    plt.plot(
        time,
        onset_curve,
        label="Onset probability",
        linewidth=2
    )

    plt.axvline(
        gt_row["Start Time (s)"],
        color="green",
        linestyle="--",
        label="GT Start"
    )

    plt.axvline(
        gt_row["End Time (s)"],
        color="red",
        linestyle="--",
        label="GT End"
    )

    plt.axvline(
        bp_note.start,
        color="green",
        linewidth=2,
        alpha=0.5,
        label="BP Start"
    )

    plt.axvline(
        bp_note.end,
        color="red",
        linewidth=2,
        alpha=0.5,
        label="BP End"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Probability")
    plt.title(f"Note {note_index+1} (MIDI {midi_pitch})")

    plt.ylim(-0.05,1.05)

    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def inspect_adjacent_note_interaction(
    wav_path,
    gt_csv,
    note_index=0
):

    model_output, midi_data, _ = predict(wav_path)

    next_gt = gt.iloc[note_index+1]

    current_bin = current_bp.pitch - 21
    next_bin = next_bp.pitch - 21

    current_frame = frames[
    frames = model_output["note"]
    onsets = model_output["onset"]

    gt = pd.read_csv(gt_csv)

    bp_notes = midi_data.instruments[0].notes

    if note_index >= len(bp_notes)-1:
        raise ValueError("Need a following note.")

    current_bp = bp_notes[note_index]
    next_bp = bp_notes[note_index+1]

    current_gt = gt.iloc[note_index]:, current_bin]
    next_frame = frames[:, next_bin]
    next_onset = onsets[:, next_bin]

    fps = 22050 / 256          # Basic Pitch FPS
    time = np.arange(len(current_frame)) / fps

    plt.figure(figsize=(16,6))

    # ---------------- Curves ----------------

    plt.plot(
        time,
        current_frame,
        linewidth=3,
        label=f"Current Frame (MIDI {current_bp.pitch})"
    )

    plt.plot(
        time,
        next_frame,
        linewidth=3,
        label=f"Next Frame (MIDI {next_bp.pitch})"
    )

    plt.plot(
        time,
        next_onset,
        linewidth=2,
        linestyle="--",
        label=f"Next Onset (MIDI {next_bp.pitch})"
    )

    # ---------------- Current Note ----------------

    plt.axvline(
        current_gt["End Time (s)"],
        color="red",
        linestyle="--",
        linewidth=2,
        label="GT Current Offset"
    )

    plt.axvline(
        current_bp.end,
        color="red",
        linewidth=2,
        alpha=0.5,
        label="BP Current Offset"
    )

    # ---------------- Next Note ----------------

    plt.axvline(
        next_gt["Start Time (s)"],
        color="green",
        linestyle="--",
        linewidth=2,
        label="GT Next Onset"
    )

    plt.axvline(
        next_bp.start,
        color="green",
        linewidth=2,
        alpha=0.5,
        label="BP Next Onset"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Probability")

    plt.title(
        f"Interaction: Note {note_index+1} → Note {note_index+2}"
    )

    plt.ylim(-0.05,1.05)
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    print(f"\nCurrent Note : MIDI {current_bp.pitch}")
    print(f"Next Note    : MIDI {next_bp.pitch}")'''


'''def print_frame_probabilities(wav_path, midi_pitch=60):

    model_output, _, _ = predict(wav_path)

    frames = model_output["note"]

    pitch_bin = midi_pitch - 21

    frame_probs = frames[:, pitch_bin]

    fps = 22050 / 256

    print(f"\nMIDI Pitch: {midi_pitch}")
    print(f"Pitch Bin : {pitch_bin}\n")

    print(f"{'Frame':>6} {'Time(s)':>10} {'Probability':>12}")

    for i, prob in enumerate(frame_probs):

        time = i / fps

        print(f"{i:6d} {time:10.4f} {prob:12.6f}")



def frame_probability_summary(wav_path, gt_csv):

    model_output, _, _ = predict(wav_path)

    frames = model_output["note"]

    fps = 22050 / 256

    gt = pd.read_csv(gt_csv)

    summary = []

    for i, row in gt.iterrows():

        midi = int(row["MIDI"])
        pitch_bin = midi - 21

        curve = frames[:, pitch_bin]

        start_frame = int(round(row["Start Time (s)"] * fps))
        end_frame = int(round(row["End Time (s)"] * fps))

        start_frame = max(0, start_frame)
        end_frame = min(len(curve)-1, end_frame)

        note_curve = curve[start_frame:end_frame+1]

        # Helper
        def value_at(time_sec):
            frame = int(round(time_sec * fps))
            frame = max(0, min(frame, len(curve)-1))
            return curve[frame]

        summary.append({

            "Note": i + 1,

            "MIDI": midi,

            "GT Start": row["Start Time (s)"],

            "GT End": row["End Time (s)"],

            "Peak Prob": np.max(note_curve),

            "Mean Prob": np.mean(note_curve),

            "100 ms Before End":
                value_at(row["End Time (s)"] - 0.10),

            "50 ms Before End":
                value_at(row["End Time (s)"] - 0.05),

            "At GT End":
                value_at(row["End Time (s)"])

        })

    df = pd.DataFrame(summary)

    print(df.round(3))

    return df'''

#predict_and_save([audio_file], output_directory, model_or_model_path= ICASSP_2022_MODEL_PATH, save_midi = True, sonify_midi=True, save_model_outputs= True, save_notes = True)

from basic_pitch import ICASSP_2022_MODEL_PATH
audio_file = "C:\\Users\\PSYLAB03\\Documents\\Audacity\\Tanpura\\TanpuraE.wav"
output_wav = "C:\\Users\\PSYLAB03\\Documents\\Audacity\\Tanpura\\TanpuraE_crepe.wav"
audio, sr = librosa.load(audio_file,sr=16000,mono=True)
# CREPE analysis
time, frequency, confidence, activation = crepe.predict(
    audio, sr,
    model_capacity="full",
    step_size=10
)

# Convert CREPE pitch track to MIDI notes
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)

valid = confidence > 0.5

for i in range(len(time) - 1):

    if not valid[i] or frequency[i] <= 0:
        continue

    midi_note = int(round(pretty_midi.hz_to_midi(frequency[i])))

    start = time[i]
    end = time[i + 1]

    note = pretty_midi.Note(
        velocity=100,
        pitch=midi_note,
        start=start,
        end=end
    )

    instrument.notes.append(note)

midi.instruments.append(instrument)

# Synthesize CREPE prediction to WAV
audio = midi.fluidsynth()

sf.write(
    output_wav,
    audio,
    44100
)

print("CREPE analysis complete.")
print("WAV saved to:", output_wav)
'''from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_FOLDER = Path(
    r"C:\Users\PSYLAB03\Desktop\Manya\1-10_Simulations"
)

OUTPUT_FOLDER = ROOT_FOLDER / "benchmarking_graphs"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# VARIATION LABELS
# ============================================================

variation_names = {
    "a": "Ascending - Equal Duration - No Interval",
    "b": "Ascending - Random Duration - No Interval",
    "c": "Ascending - Equal Duration - Random Interval",
    "d": "Ascending - Random Duration - Random Intervals",
    "e": "Descending - Equal Duration - No Interval"
}


# ============================================================
# FIND SIMULATION FOLDERS
# ============================================================

simulation_folders = []

for folder in ROOT_FOLDER.iterdir():

    if not folder.is_dir():
        continue

    match = re.fullmatch(
        r"Simulation\s*(\d+)\s*([a-e])",
        folder.name.strip(),
        re.IGNORECASE
    )

    if match:

        number = int(match.group(1))
        variation = match.group(2).lower()

        simulation_folders.append(
            (number, variation, folder)
        )


simulation_folders.sort(
    key=lambda x: (x[0], x[1])
)


print("=" * 70)
print(
    f"Simulation folders found: "
    f"{len(simulation_folders)}"
)
print("=" * 70)


for number, variation, folder in simulation_folders:
    print(
        f"Simulation {number}{variation}: "
        f"{folder}"
    )


print("=" * 70)


# ============================================================
# CSV READER
# ============================================================

def read_notes_csv(path):

    df = pd.read_csv(path)

    required_columns = [
        "Note",
        "Note Code",
        "Frequency (Hz)",
        "MIDI",
        "Start Time (s)",
        "End Time (s)",
        "Duration (s)",
        "interval_after (s)"
    ]

    missing = [
        c
        for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {missing}"
        )

    numeric_columns = [
        "Frequency (Hz)",
        "MIDI",
        "Start Time (s)",
        "End Time (s)",
        "Duration (s)",
        "interval_after (s)"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Start Time (s)",
            "End Time (s)"
        ]
    ).copy()

    df = df[
        df["End Time (s)"]
        >=
        df["Start Time (s)"]
    ].copy()

    df.reset_index(
        drop=True,
        inplace=True
    )

    return df


# ============================================================
# PITCH ERROR
# ============================================================

def pitch_error_cents(
    predicted_frequency,
    true_frequency
):

    if (
        pd.isna(predicted_frequency)
        or pd.isna(true_frequency)
        or predicted_frequency <= 0
        or true_frequency <= 0
    ):
        return np.nan

    return (
        1200
        *
        np.log2(
            predicted_frequency
            /
            true_frequency
        )
    )


# ============================================================
# TEMPORAL OVERLAP
# ============================================================

def temporal_overlap(
    pred_start,
    pred_end,
    gt_start,
    gt_end
):

    overlap_start = max(
        pred_start,
        gt_start
    )

    overlap_end = min(
        pred_end,
        gt_end
    )

    return max(
        0.0,
        overlap_end - overlap_start
    )


# ============================================================
# NOTE MATCHING
# ============================================================

def match_notes(
    ground_truth,
    prediction,
    pitch_tolerance_cents=50
):

    gt = ground_truth.reset_index(drop=True)
    pred = prediction.reset_index(drop=True)

    candidates = []

    for gt_idx, gt_row in gt.iterrows():

        for pred_idx, pred_row in pred.iterrows():

            cents = pitch_error_cents(
                pred_row["Frequency (Hz)"],
                gt_row["Frequency (Hz)"]
            )

            if pd.isna(cents):
                continue

            if abs(cents) > pitch_tolerance_cents:
                continue

            overlap = temporal_overlap(
                pred_row["Start Time (s)"],
                pred_row["End Time (s)"],
                gt_row["Start Time (s)"],
                gt_row["End Time (s)"]
            )

            if overlap <= 0:
                continue

            candidates.append(
                (
                    gt_idx,
                    pred_idx,
                    overlap,
                    abs(cents)
                )
            )

    # Largest temporal overlap first,
    # then smallest pitch error.

    candidates.sort(
        key=lambda x: (
            -x[2],
            x[3]
        )
    )

    matched_gt = set()
    matched_pred = set()

    matches = []

    for (
        gt_idx,
        pred_idx,
        overlap,
        abs_cents
    ) in candidates:

        if gt_idx in matched_gt:
            continue

        if pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

        matches.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": pred_idx
            }
        )

    return matches


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    ground_truth,
    prediction
):

    gt = ground_truth.reset_index(drop=True)
    pred = prediction.reset_index(drop=True)

    matches = match_notes(
        gt,
        pred
    )

    tp = len(matches)

    fp = len(pred) - tp

    fn = len(gt) - tp


    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )


    pitch_errors = []
    start_errors = []
    end_errors = []
    duration_errors = []
    interval_errors = []


    for match in matches:

        gt_row = gt.iloc[
            match["gt_idx"]
        ]

        pred_row = pred.iloc[
            match["pred_idx"]
        ]


        # ----------------------------------------------------
        # Pitch
        # ----------------------------------------------------

        cents = pitch_error_cents(
            pred_row["Frequency (Hz)"],
            gt_row["Frequency (Hz)"]
        )

        if not pd.isna(cents):
            pitch_errors.append(cents)


        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        start_errors.append(
            pred_row["Start Time (s)"]
            -
            gt_row["Start Time (s)"]
        )


        # ----------------------------------------------------
        # End
        # ----------------------------------------------------

        end_errors.append(
            pred_row["End Time (s)"]
            -
            gt_row["End Time (s)"]
        )


        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_errors.append(
            pred_row["Duration (s)"]
            -
            gt_row["Duration (s)"]
        )


        # ----------------------------------------------------
        # Interval
        # ----------------------------------------------------

        gt_interval = gt_row["interval_after (s)"]
        pred_interval = pred_row["interval_after (s)"]

        if (
            not pd.isna(gt_interval)
            and
            not pd.isna(pred_interval)
        ):

            interval_errors.append(
                pred_interval
                -
                gt_interval
            )


    # ========================================================
    # PITCH
    # ========================================================

    if pitch_errors:

        pitch_errors = np.array(
            pitch_errors
        )

        pitch_accuracy = np.mean(
            np.abs(pitch_errors) <= 50
        )

        mean_abs_pitch_error = np.mean(
            np.abs(pitch_errors)
        )

    else:

        pitch_accuracy = np.nan
        mean_abs_pitch_error = np.nan


    # ========================================================
    # START TIME
    # ========================================================

    if start_errors:

        start_time_mae = np.mean(
            np.abs(start_errors)
        )

    else:

        start_time_mae = np.nan


    # ========================================================
    # END TIME
    # ========================================================

    if end_errors:

        end_time_mae = np.mean(
            np.abs(end_errors)
        )

        signed_end_time_bias = np.mean(
            end_errors
        )

    else:

        end_time_mae = np.nan
        signed_end_time_bias = np.nan


    # ========================================================
    # DURATION
    # ========================================================

    if duration_errors:

        duration_mae = np.mean(
            np.abs(duration_errors)
        )

    else:

        duration_mae = np.nan


    # ========================================================
    # INTERVAL
    # ========================================================

    if interval_errors:

        interval_mae = np.mean(
            np.abs(interval_errors)
        )

    else:

        interval_mae = np.nan


    return {

        "GT Notes": len(gt),

        "Predicted Notes": len(pred),

        "TP": tp,

        "FP": fp,

        "FN": fn,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Pitch Accuracy ±50 cents":
            pitch_accuracy,

        "Mean Absolute Pitch Error (cents)":
            mean_abs_pitch_error,

        "Start-Time MAE (s)":
            start_time_mae,

        "End-Time MAE (s)":
            end_time_mae,

        "Signed End-Time Bias (s)":
            signed_end_time_bias,

        "Duration MAE (s)":
            duration_mae,

        "Interval MAE (s)":
            interval_mae,

        "Matched Notes": tp
    }


# ============================================================
# EXACT FILE DEFINITIONS
# ============================================================

def get_files(
    folder,
    number,
    variation
):

    # Example:
    #
    # Simulation 1a_CREPE_5ms_Formatted.csv
    #
    prefix = f"Simulation {number}{variation}"

    files = {

        "Ground Truth":
            folder
            /
            f"simulation{number}{variation}.csv",

        "CREPE 5 ms":
            folder
            /
            f"{prefix}_CREPE_5ms_Formatted.csv",

        "CREPE 10 ms":
            folder
            /
            f"{prefix}_CREPE_10ms_Formatted.csv",

        "CREPE 15 ms":
            folder
            /
            f"{prefix}_CREPE_15ms_Formatted.csv",

        "CREPE Post-Processing":
            folder
            /
            f"simulation{number}{variation}CrepeNotes.csv",

        "Librosa 5 ms":
            folder
            /
            f"{prefix}_Librosa_5ms_Formatted.csv",

        "Librosa 10 ms":
            folder
            /
            f"{prefix}_Librosa_10ms_Formatted.csv",

        "Librosa 15 ms":
            folder
            /
            f"{prefix}_Librosa_15ms_Formatted.csv",

        "Basic Pitch":
            folder
            /
            f"{prefix}_BasicPitch.csv"
    }

    return files


# ============================================================
# PROCESS ALL SIMULATIONS
# ============================================================

all_results = []


for number, variation, folder in simulation_folders:

    print()
    print("-" * 70)
    print(
        f"Processing: Simulation {number}{variation}"
    )
    print("-" * 70)


    files = get_files(
        folder,
        number,
        variation
    )


    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    print("Files:")

    for method, path in files.items():

        if path.exists():

            print(
                f"  ✓ {method}: "
                f"{path.name}"
            )

        else:

            print(
                f"  ✗ MISSING {method}: "
                f"{path.name}"
            )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_path = files["Ground Truth"]


    if not gt_path.exists():

        print(
            "Skipping simulation because "
            "ground truth is missing."
        )

        continue


    try:

        ground_truth = read_notes_csv(
            gt_path
        )

    except Exception as e:

        print(
            f"ERROR reading ground truth: {e}"
        )

        continue


    # --------------------------------------------------------
    # Prediction files
    # --------------------------------------------------------

    prediction_methods = [

        "CREPE 5 ms",

        "CREPE 10 ms",

        "CREPE 15 ms",

        "CREPE Post-Processing",

        "Librosa 5 ms",

        "Librosa 10 ms",

        "Librosa 15 ms",

        "Basic Pitch"
    ]


    # --------------------------------------------------------
    # Process each prediction
    # --------------------------------------------------------

    for method in prediction_methods:

        prediction_path = files[method]


        if not prediction_path.exists():

            print(
                f"  SKIPPING {method}: "
                f"file not found"
            )

            continue


        try:

            prediction = read_notes_csv(
                prediction_path
            )


            metrics = calculate_metrics(
                ground_truth,
                prediction
            )


            result = {

                "Simulation":
                    number,

                "Simulation Folder":
                    folder.name,

                "Variation":
                    variation,

                "Variation Description":
                    variation_names[
                        variation
                    ],

                "Method":
                    method,

                "File":
                    prediction_path.name
            }


            result.update(
                metrics
            )


            all_results.append(
                result
            )


        except Exception as e:

            print(
                f"  ERROR processing "
                f"{method}: {e}"
            )


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print(
    f"Total benchmark results: "
    f"{len(all_results)}"
)
print("=" * 70)


if not all_results:

    raise SystemExit(
        "NO RESULTS GENERATED."
    )


results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE ALL RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FOLDER
    /
    "benchmarking_all_simulations.csv",
    index=False
)


# ============================================================
# METRICS
# ============================================================

metric_columns = [

    "Precision",

    "Recall",

    "F1",

    "Pitch Accuracy ±50 cents",

    "Mean Absolute Pitch Error (cents)",

    "Start-Time MAE (s)",

    "End-Time MAE (s)",

    "Signed End-Time Bias (s)",

    "Duration MAE (s)",

    "Interval MAE (s)"
]


# ============================================================
# SUMMARY BY METHOD
# ============================================================

summary_df = (
    results_df
    .groupby("Method")[metric_columns]
    .mean()
    .reset_index()
)


summary_df.to_csv(
    OUTPUT_FOLDER
    /
    "benchmarking_summary.csv",
    index=False
)


# ============================================================
# SUMMARY BY VARIATION
# ============================================================

variation_df = (
    results_df
    .groupby(
        [
            "Method",
            "Variation",
            "Variation Description"
        ]
    )[metric_columns]
    .mean()
    .reset_index()
)


variation_df.to_csv(
    OUTPUT_FOLDER
    /
    "benchmarking_by_variation.csv",
    index=False
)


# ============================================================
# METRIC DEFINITIONS
# ============================================================

definitions = pd.DataFrame({

    "Metric": [

        "Precision",

        "Recall",

        "F1",

        "Pitch Accuracy ±50 cents",

        "Mean Absolute Pitch Error (cents)",

        "Start-Time MAE (s)",

        "End-Time MAE (s)",

        "Signed End-Time Bias (s)",

        "Duration MAE (s)",

        "Interval MAE (s)"
    ],

    "Definition": [

        "TP / (TP + FP)",

        "TP / (TP + FN)",

        "Harmonic mean of precision and recall",

        "Percentage of matched notes within ±50 cents",

        "Mean absolute pitch error of matched notes in cents",

        "Mean absolute onset error",

        "Mean absolute offset error",

        "Mean signed offset error; negative = earlier prediction, positive = later prediction",

        "Mean absolute duration error",

        "Mean absolute inter-note interval error"
    ]
})


definitions.to_csv(
    OUTPUT_FOLDER
    /
    "metric_definitions.csv",
    index=False
)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def save_graph(filename):

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved graph: {filename}"
    )


# ============================================================
# GRAPH 1 — PRECISION
# ============================================================

data = (
    results_df
    .groupby("Method")["Precision"]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("Precision")
plt.title("Note Detection Precision")

plt.ylim(0, 1)

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "01_note_precision.png"
)


# ============================================================
# GRAPH 2 — RECALL
# ============================================================

data = (
    results_df
    .groupby("Method")["Recall"]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("Recall")
plt.title("Note Detection Recall")

plt.ylim(0, 1)

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "02_note_recall.png"
)


# ============================================================
# GRAPH 3 — F1
# ============================================================

data = (
    results_df
    .groupby("Method")["F1"]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("F1 Score")
plt.title("Note Detection F1 Score")

plt.ylim(0, 1)

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "03_note_f1.png"
)


# ============================================================
# GRAPH 4 — PITCH ACCURACY
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Pitch Accuracy ±50 cents"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("Accuracy")
plt.title(
    "Pitch Accuracy Within ±50 Cents"
)

plt.ylim(0, 1)

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "04_pitch_accuracy_50cents.png"
)


# ============================================================
# GRAPH 5 — PITCH ERROR
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Mean Absolute Pitch Error (cents)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel(
    "Mean Absolute Error (cents)"
)

plt.title(
    "Mean Absolute Pitch Error"
)

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "05_mean_absolute_pitch_error.png"
)


# ============================================================
# GRAPH 6 — START TIME
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Start-Time MAE (s)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("MAE (s)")
plt.title("Start-Time Error")

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "06_start_time_mae.png"
)


# ============================================================
# GRAPH 7 — END TIME
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "End-Time MAE (s)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("MAE (s)")
plt.title("End-Time Error")

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "07_end_time_mae.png"
)


# ============================================================
# GRAPH 8 — SIGNED END-TIME BIAS
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Signed End-Time Bias (s)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.axhline(
    0,
    linewidth=1
)

plt.ylabel("Signed Error (s)")
plt.title("Signed End-Time Bias")

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "08_signed_end_time_bias.png"
)


# ============================================================
# GRAPH 9 — DURATION
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Duration MAE (s)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("MAE (s)")
plt.title("Note Duration Error")

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "09_duration_mae.png"
)


# ============================================================
# GRAPH 10 — INTERVAL
# ============================================================

data = (
    results_df
    .groupby("Method")[
        "Interval MAE (s)"
    ]
    .mean()
)


plt.figure(figsize=(11, 6))

plt.bar(
    data.index,
    data.values
)

plt.ylabel("MAE (s)")
plt.title("Inter-Note Interval Error")

plt.xticks(
    rotation=35,
    ha="right"
)

save_graph(
    "10_interval_mae.png"
)


# ============================================================
# GRAPH 11 — F1 BY VARIATION
# ============================================================

pivot = (
    results_df
    .groupby(
        [
            "Variation",
            "Method"
        ]
    )["F1"]
    .mean()
    .unstack()
)


pivot = pivot.reindex(
    ["a", "b", "c", "d", "e"]
)


plt.figure(figsize=(12, 7))

pivot.plot(
    kind="line",
    marker="o",
    ax=plt.gca()
)

plt.xlabel("Variation")
plt.ylabel("F1 Score")

plt.title(
    "F1 Score Across Simulation Variations"
)

plt.xticks(
    range(5),
    [
        "a: Equal duration, no interval",
        "b: Random duration, no interval",
        "c: Equal duration, random interval",
        "d: Random duration, random intervals",
        "e: Descending, equal duration"
    ],
    rotation=25,
    ha="right"
)

plt.ylim(0, 1)

plt.legend(
    title="Method"
)

save_graph(
    "11_f1_by_variation.png"
)


# ============================================================
# GRAPH 12 — END TIME BY VARIATION
# ============================================================

pivot = (
    results_df
    .groupby(
        [
            "Variation",
            "Method"
        ]
    )["End-Time MAE (s)"]
    .mean()
    .unstack()
)


pivot = pivot.reindex(
    ["a", "b", "c", "d", "e"]
)


plt.figure(figsize=(12, 7))

pivot.plot(
    kind="line",
    marker="o",
    ax=plt.gca()
)

plt.xlabel("Variation")
plt.ylabel("End-Time MAE (s)")

plt.title(
    "End-Time Error Across Simulation Variations"
)

plt.xticks(
    range(5),
    [
        "a: Equal duration, no interval",
        "b: Random duration, no interval",
        "c: Equal duration, random interval",
        "d: Random duration, random intervals",
        "e: Descending, equal duration"
    ],
    rotation=25,
    ha="right"
)

plt.legend(
    title="Method"
)

save_graph(
    "12_end_time_by_variation.png"
)


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("BENCHMARKING COMPLETE")
print("=" * 70)

print(
    f"Simulation folders: "
    f"{len(simulation_folders)}"
)

print(
    f"Benchmark rows: "
    f"{len(results_df)}"
)

print(
    "Expected: 50 × 8 = 400 prediction rows"
)

print()
print(
    f"All outputs are in:\n{OUTPUT_FOLDER}"
)

print("=" * 70)
'''