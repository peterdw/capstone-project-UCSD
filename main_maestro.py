import pathlib
import time

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pretty_midi

from constants import BEST_LSTM_MODEL, MAESTRO_DATASET_FOLDER, SEQ_LENGTH, PITCH_VOCAB_SIZE
from filenames_singleton import FilenamesSingleton
from model_builder import create_music_sequence_dataset, create_midi_training_dataset, get_lstm_model
from music_player import midi_to_notes


def start():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_dir = pathlib.Path('data/maestro-v3.0.0')
    filenames_singleton = FilenamesSingleton(data_dir=pathlib.Path(MAESTRO_DATASET_FOLDER))
    filenames = filenames_singleton.filenames

    ##############
    # TEST BLOCK #
    ##############

    sample_file = filenames[1]
    pm = pretty_midi.PrettyMIDI(sample_file)
    print('Number of instruments:', len(pm.instruments))
    instrument = pm.instruments[0]
    instrument_name = pretty_midi.program_to_instrument_name(instrument.program)
    print('Instrument name:', instrument_name)
    for i, note in enumerate(instrument.notes[:10]):
        note_name = pretty_midi.note_number_to_name(note.pitch)
        duration = note.end - note.start
        print(f'{i}: pitch={note.pitch}, note_name={note_name},'
              f' duration={duration:.4f}')
    raw_notes = midi_to_notes(sample_file)
    print(raw_notes.head())
    get_note_names = np.vectorize(pretty_midi.note_number_to_name)
    sample_note_names = get_note_names(raw_notes['pitch'])
    print(sample_note_names[:10])

    ##################################
    # CREATE DATASET FROM MIDI FILES #
    ##################################

    # create the tensorflow dataset
    notes_ds = create_midi_training_dataset(filenames, 5)
    # print(notes_ds.element_spec)
    dataset_size = tf.data.experimental.cardinality(notes_ds).numpy()
    print('Number of items in notes_ds:', dataset_size)

    seq_ds = create_music_sequence_dataset(notes_ds, SEQ_LENGTH, PITCH_VOCAB_SIZE)
    print(seq_ds.element_spec)
    for seq, target in seq_ds.take(1):
        print('sequence shape:', seq.shape)
        print('sequence elements (first 10):', seq[0:10])
        print()
        print('target:', target)

    train_size = int(0.8 * dataset_size)

    train_ds = seq_ds.take(train_size)
    val_ds = seq_ds.skip(train_size)

    batch_size = 64

    # train_ds = (train_ds
    #             .shuffle(buffer_size=train_size)
    #             .batch(batch_size, drop_remainder=True)
    #             .cache()
    #             .prefetch(tf.data.experimental.AUTOTUNE))

    buffer_size = dataset_size - SEQ_LENGTH  # the number of items in the dataset
    train_ds = (seq_ds
                .shuffle(buffer_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))
    print(train_ds.element_spec)

    # val_ds = (val_ds
    #           .batch(batch_size, drop_remainder=True)  # No need to shuffle validation data
    #           .cache()
    #           .prefetch(tf.data.experimental.AUTOTUNE))
    # print(val_ds.element_spec)

    # tuner = kt.RandomSearch(
    #     build_hypermodel,
    #     objective='val_loss',
    #     max_trials=10,
    #     executions_per_trial=1,
    #     directory='tuning',
    #     project_name='lstm_tuning_random_search'
    # )

    # tuner = kt.Hyperband(
    #     build_lstm_hypermodel,
    #     objective='val_loss',
    #     max_epochs=50,
    #     directory='tuning',
    #     project_name='lstm_tuning'
    # )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True)
    ]
    # Start hyperparameter search
    # tuner.search(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)

    # Get the optimal hyperparameters
    # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # print(f"Best learning rate: {best_hps.get('learning_rate')}")
    # print(f"Best LSTM units: {best_hps.get('units')}")

    ###################
    # train THE MODEL #
    ###################

    start_time = time.time()
    best_epochs = 50

    # Build the model with the best hyperparameters and train it on the data
    # best_model = tuner.hypermodel.build(best_hps)

    best_model = get_lstm_model()

    history = best_model.fit(train_ds, epochs=best_epochs, callbacks=callbacks)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    ##################
    # SAVE THE MODEL #
    ##################

    model_save_path = BEST_LSTM_MODEL
    best_model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    #############################
    # PLOT THE TRAINING HISTORY #
    #############################

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    loss_plot = f'loss_plot_{best_epochs}_epochs.png'
    plt.savefig(loss_plot)
    plt.clf()
    print(f"Loss plot saved as: {loss_plot}")


if __name__ == '__main__':
    start()
