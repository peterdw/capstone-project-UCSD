import glob
import pathlib
import time

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import music_player
from model_builder import create_music_sequence_dataset, create_midi_training_dataset, build_lstm_hypermodel


def start():
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_dir = pathlib.Path('data/maestro-v3.0.0')
    filenames = glob.glob(str(data_dir / '**/*.mid*'))
    notes_ds = create_midi_training_dataset(filenames, 10)
    n_notes = tf.data.experimental.cardinality(notes_ds).numpy()
    print('Number of items in notes_ds:', n_notes)

    seq_length = 25
    vocab_size = 128
    seq_ds = create_music_sequence_dataset(notes_ds, seq_length, vocab_size)

    dataset_size = n_notes
    train_size = int(0.8 * dataset_size)

    train_ds = seq_ds.take(train_size)
    val_ds = seq_ds.skip(train_size)

    batch_size = 64

    train_ds = (train_ds
                .shuffle(buffer_size=train_size)
                .batch(batch_size, drop_remainder=True)
                .cache()
                .prefetch(tf.data.experimental.AUTOTUNE))

    val_ds = (val_ds
              .batch(batch_size, drop_remainder=True)  # No need to shuffle validation data
              .cache()
              .prefetch(tf.data.experimental.AUTOTUNE))

    tuner = kt.Hyperband(
        build_lstm_hypermodel,
        objective='val_loss',
        max_epochs=50,
        directory='tuning',
        project_name='lstm_tuning'
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True)
    ]
    # Start hyperparameter search
    tuner.search(train_ds, epochs=50, validation_data=val_ds, callbacks=callbacks)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"Best learning rate: {best_hps.get('learning_rate')}")
    print(f"Best LSTM units: {best_hps.get('units')}")

    # callbacks = [
    #     tf.keras.callbacks.ModelCheckpoint(
    #         filepath='./training_checkpoints/ckpt_{epoch}',
    #         save_weights_only=True),
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor='loss',
    #         patience=5,
    #         verbose=1,
    #         restore_best_weights=True),
    # ]
    start_time = time.time()
    best_epochs = 50
    #
    # history = model.fit(
    #     train_ds,
    #     epochs=epochs,
    #     callbacks=callbacks,
    # )

    # Build the model with the best hyperparameters and train it on the data
    best_model = tuner.hypermodel.build(best_hps)
    history = best_model.fit(train_ds, epochs=best_epochs, callbacks=callbacks)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    loss_plot = f'loss_plot_{best_epochs}_epochs.png'
    plt.savefig(loss_plot)
    plt.clf()
    print(f"Loss plot saved as: {loss_plot}")


def create_midi(raw_notes: pd.DataFrame, instrument_name: str):
    example_file = 'example.midi'
    _ = music_player.notes_to_midi(
        raw_notes, out_file=example_file, instrument_name=instrument_name)


if __name__ == '__main__':
    start()
