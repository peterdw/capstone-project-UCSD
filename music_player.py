import collections
import subprocess

import numpy as np
import pandas as pd
import pygame
import pathlib
import glob
import pretty_midi
from matplotlib import pyplot as plt
import os


def play_music(file, duration):
    pygame.init()
    pygame.mixer.music.load(file)
    pygame.mixer.music.play()
    pygame.time.wait(int(duration * 1000))
    pygame.mixer.music.stop()


def print_instruments_in_files(data_dir):
    """
    Iterates over MIDI files in a specified directory and its subdirectories,
    printing the instruments present in each file. This function is particularly
    useful for understanding the composition of MIDI files in a dataset, including
    identifying files with no instruments or those with a diverse set of instruments.

    Parameters:
    - data_dir (str): Path to the directory containing MIDI files to be analyzed.

    The function lists out each MIDI file found, specifying the number of instruments
    it contains and the names of these instruments. It provides an easy way to
    review the musical instrument digital interface (MIDI) files' instrumentations,
    aiding in tasks such as dataset curation or analysis where knowing the instrument
    makeup of files is beneficial. Additionally, it counts and prints the total number
    of MIDI files processed, offering insight into the dataset's size and composition.
    """
    data_dir_path = pathlib.Path(data_dir)
    filenames = glob.glob(str(data_dir_path / '**/*.mid*'))
    num_files_parsed = 0

    for file in filenames:
        pm = pretty_midi.PrettyMIDI(file)
        num_instruments = len(pm.instruments)
        if num_instruments > 0:
            print(f"Instruments used in {file} ({num_instruments} instruments):")
            for instrument in pm.instruments:
                print("- " + pretty_midi.program_to_instrument_name(instrument.program))
            print()
            num_files_parsed += 1
        else:
            print(f"No instruments found in {file}")
            print()

    print(f"Total number of files parsed: {num_files_parsed}")


def find_single_piano_files(data_dir: str):
    """
    Searches through a specified directory (and its subdirectories) for MIDI files
    that contain exactly one instrument, specifically an 'Acoustic Grand Piano'.
    This function is useful for filtering MIDI files for specific analyses,
    such as piano music generation or studying piano compositions.

    Parameters:
    - data_dir (str): The path to the directory where MIDI files are stored.

    Returns:
    - list: A list of paths to MIDI files that meet the criteria of containing
      exactly one 'Acoustic Grand Piano' instrument.

    The function iterates over all MIDI files found in the given directory,
    examining the number and type of instruments in each. It collects and returns
    the paths of files that contain only a single 'Acoustic Grand Piano' instrument,
    making it easier to identify and work with MIDI files of interest for piano-related projects.
    """
    data_dir_path = pathlib.Path(data_dir)
    filenames = glob.glob(str(data_dir_path / '**/*.mid*'))

    single_piano_files = []

    for file in filenames:
        pm = pretty_midi.PrettyMIDI(file)
        instruments = pm.instruments
        if len(instruments) == 1 and pretty_midi.program_to_instrument_name(
                instruments[0].program) == 'Acoustic Grand Piano':
            single_piano_files.append(file)

    return single_piano_files


def print_multi_instrument_files(data_dir: str):
    """
    Scans a directory (and its subdirectories) for MIDI files and prints out the
    filenames that contain more than one instrument. For each such file, it also
    prints a list of the instruments found within. This can be useful for identifying
    complex MIDI files in a dataset that feature multiple instruments, aiding in
    tasks such as data filtering or analysis where files with multiple instruments
    are of interest.

    Parameters:
    - data_dir (str): The path to the directory containing MIDI files to be analyzed.

    This function goes through each MIDI file in the specified directory, checks the
    number of instruments in each file, and for those with more than one instrument,
    prints the file's path and a list of instruments contained within. It also keeps
    track of and prints the total number of such multi-instrument files found.
    """
    data_dir_path = pathlib.Path(data_dir)
    filenames = glob.glob(str(data_dir_path / '**/*.mid*'))
    num_files_parsed = 0

    for file in filenames:
        pm = pretty_midi.PrettyMIDI(file)
        num_instruments = len(pm.instruments)
        if num_instruments > 1:
            print(f"MIDI file: {file}")
            print("Instruments:")
            for instrument in pm.instruments:
                instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
                print(f"- {instrument_name}")
            print()
            num_files_parsed += 1

    print(f"Total number of files parsed: {num_files_parsed}")


def midi_to_notes(midi_file: str) -> pd.DataFrame:
    """
    Converts a MIDI file to a structured DataFrame containing the notes played
    by the first instrument track found in the file. Each note is represented by
    its pitch, start time, end time, time step from the previous note, and its
    duration. The notes are sorted by their start time to maintain temporal order.

    Parameters:
    - midi_file (str): The path to the MIDI file to be converted.

    Returns:
    - pd.DataFrame: A DataFrame with columns for 'pitch', 'start', 'end', 'step',
      and 'duration' for each note played by the first instrument in the MIDI file.
    """
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]

    # Assert that the instrument is an Acoustic Grand Piano
    assert instrument.program == 0, "The instrument is not a Grand Acoustic Piano."

    notes = collections.defaultdict(list)

    # Sort the notes by start time
    sorted_notes = sorted(instrument.notes, key=lambda n: n.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})


def piano_roll_to_png(notes: pd.DataFrame, count: int = None):
    """
    Generates a visual representation (piano roll) of MIDI note events as a PNG image.
    The piano roll displays the pitch of notes against time, with an option to limit
    the number of notes displayed. This function handles both the creation of the
    plot and its storage as a PNG file, and optionally attempts to open the image
    using the system's default image viewer.

    Parameters:
    - notes (pd.DataFrame): DataFrame containing MIDI note data, expected to have
      'pitch', 'start', and 'end' columns.
    - count (int, optional): The number of notes to include in the visualization.
      If None, the whole track is visualized.

    The function creates a plot with the specified title reflecting either the whole
    track or a subset of notes, saves the plot as 'piano_roll_plot.png' in the current
    working directory, and attempts to open the saved image file with the system's
    default viewer, handling any exceptions that occur during the attempt to open.
    """
    if count is None:
        title = 'Whole track'
        count = len(notes['pitch'])
    else:
        title = f'First {count} notes'

    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time [s]')
    plt.ylabel('Pitch')
    plt.title(title)

    # Save the plot to a PNG file
    filename = "piano_roll_plot.png"
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory

    # Use subprocess to open the file with a specific application
    # Replace 'open' with the command for your preferred image viewer
    # For example, on Linux you might use 'xdg-open', on Windows 'start', on macOS 'open'
    filepath = os.path.abspath(filename)
    try:
        subprocess.run(['xdg-open', filepath], check=True)  # Example for Linux
    except Exception as e:
        print(f"Failed to open image: {e}")


def notes_to_midi(
        notes: pd.DataFrame,
        out_file: str,
        velocity: int = 100,  # note loudness
) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm
