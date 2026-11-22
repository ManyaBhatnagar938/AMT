import crepe_notes
import subprocess
import pretty_midi
import pandas as pd
from pathlib import Path

def get_wav_file(simulation_folder):
    '''input : path to simulation folder where the wanted file is stored
       output : the wanted .wav file'''
    simulation_folder = Path(simulation_folder)
    wav_files = list(simulation_folder.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(
            f"No WAV file found in {simulation_folder}")
    print(wav_files[0])
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

def midi_to_note_code(midi):
    if midi == 0:
        return None
    return pretty_midi.note_number_to_name(midi)

def midi_to_note(midi):
    if midi == 0:
        return None
    return NOTE_MAPPING[midi_to_note_code(midi)]

def midi_to_csv(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    rows = []

    for instrument in midi.instruments:
        notes = sorted(instrument.notes, key = lambda n: n.start)

        for i,note in enumerate(notes):
            start_time = round(note.start, 4)
            end_time = round(note.end, 4)
            duration = round(end_time - start_time, 4)
            midi_num = note.pitch
            frequency = round(pretty_midi.note_number_to_hz(midi_num), 2)
            note_name = midi_to_note(midi_num)
            note_code = midi_to_note_code(midi_num)
            velocity = note.velocity

            if i < len(notes)-1 :
                interval_after = round(notes[i+1].start - end_time, 4)
            else:
                interval_after = None

            rows.append({
                "Note":note_name,
                "Note Code":note_code,
                "MIDI":midi_num,
                "Frequency (Hz)": frequency,
                "Start Time (s)": start_time,
                "End Time (s)": end_time,
                "Duration (s)": duration,
                "interval_after (s)": interval_after,
                "Velocity": velocity
            })

    return pd.DataFrame(rows)

def FullFunction(simulation_folder):
    wav_path = get_wav_file(simulation_folder)
    subprocess.run(["crepe_notes", str(wav_path)], check = True)
    midi_path = wav_path.parent/f"{wav_path.stem}.crepe.transcription.mid"
    df = midi_to_csv(midi_path)
    filename = wav_path.parent/f"{wav_path.stem}CrepeNotes.csv"
    save_csv(df,simulation_folder,filename)

'''tracker = 0
for i in range(1,11):
    for j in ["a","b","c","d","e"]:
        wav_folder = f"\\Simulation {i}{j}"
        FullFunction(wav_folder)
        tracker = tracker + 1
        print("Completed analysis for", tracker, "simulation/s")'''

