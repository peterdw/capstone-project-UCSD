import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

import music_player
from constants import BEST_LSTM_MODEL, KEY_ORDER, PITCH_VOCAB_SIZE, SEQ_LENGTH, MAESTRO_DATASET_FOLDER
from filenames_singleton import FilenamesSingleton
from tensorflow.keras.models import load_model

from predict import predict_next_note, create_midi
from model_builder import mean_squared_error_with_penalty_for_negatives


def start():
    ##################
    # GENERATE NOTES #
    ##################

    filenames_singleton = FilenamesSingleton(data_dir=pathlib.Path(MAESTRO_DATASET_FOLDER))
    filenames = filenames_singleton.filenames

    with tf.keras.utils.custom_object_scope(
            {'mean_squared_error_with_penalty_for_negatives': mean_squared_error_with_penalty_for_negatives}):
        model = load_model(BEST_LSTM_MODEL)
    model.summary()

    temperature = 2.0
    num_predictions = 50

    sample_file = filenames[1]
    raw_notes = music_player.midi_to_notes(sample_file)
    print(raw_notes.head())
    sample_notes = np.stack([raw_notes[key] for key in KEY_ORDER], axis=1)
    print(sample_notes)
    # The initial sequence of notes; pitch is normalized similar to training sequences
    input_notes = (sample_notes[:SEQ_LENGTH] / np.array([PITCH_VOCAB_SIZE, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        new_start = prev_start + step
        end = new_start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, new_start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = new_start

    columns = list(KEY_ORDER) + ['start', 'end']
    # generated_notes = pd.DataFrame(generated_notes, columns=columns)
    generated_notes = pd.DataFrame(
        generated_notes, columns=columns)
    print('generated_notes:')
    print(generated_notes.head(10))
    create_midi(generated_notes)


if __name__ == '__main__':
    start()
