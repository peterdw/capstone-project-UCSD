# load_music_data.py

import os
import glob
import pickle
import music21
from music21 import converter
from constants import CELLO_DATASET_FOLDER, PARSE_MIDI_FILES, NOTES_FILE, DURATIONS_FILE, SEQ_LEN
import pretty_midi
import numpy as np

def midi_to_piano_roll(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # Assuming you want to work with one instrument
    piano_roll = midi_data.instruments[0].get_piano_roll(fs=100)
    return np.array(piano_roll, dtype=np.float32)

def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data(data, file_path):
    ensure_directory_exists(file_path)
    with open(file_path, 'wb') as filepath:
        pickle.dump(data, filepath)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'rb') as filepath:
        return pickle.load(filepath)

def parse_midi_files(file_list):
    notes = ["START"]
    durations = ["0.0"]

    for file in file_list:
        print(f"Parsing {file}")
        score = converter.parse(file).chordify()

        for element in score.flatten():
            if isinstance(element, music21.key.Key):
                notes.append(str(element.tonic.name) + ":" + str(element.mode))
                durations.append("0.0")
            elif isinstance(element, music21.meter.TimeSignature):
                notes.append(str(element.ratioString) + "TS")
                durations.append("0.0")
            elif isinstance(element, music21.chord.Chord):
                notes.append(element.pitches[-1].nameWithOctave)
                durations.append(str(element.duration.quarterLength))
            elif isinstance(element, music21.note.Rest):
                notes.append(str(element.name))
                durations.append(str(element.duration.quarterLength))
            elif isinstance(element, music21.note.Note):
                notes.append(str(element.nameWithOctave))
                durations.append(str(element.duration.quarterLength))

    notes_sequences = []
    durations_sequences = []

    for i in range(len(notes) - SEQ_LEN):
        notes_sequences.append(" ".join(notes[i: i + SEQ_LEN]))
        durations_sequences.append(" ".join(durations[i: i + SEQ_LEN]))

    return notes_sequences, durations_sequences

def load_music_data():
    if PARSE_MIDI_FILES:
        file_list = glob.glob(os.path.join(CELLO_DATASET_FOLDER, '**/*.mid*'))
        print(f"Found {len(file_list)} MIDI files")

        notes, durations = parse_midi_files(file_list)

        # Save the data to disk
        save_data(notes, NOTES_FILE)
        save_data(durations, DURATIONS_FILE)

        return notes, durations
    else:
        # Load the data from disk
        try:
            notes = load_data(NOTES_FILE)
            durations = load_data(DURATIONS_FILE)
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please set PARSE_MIDI_FILES to True to parse and save the data first.")
            raise

        return notes, durations