import numpy as np
import pandas as pd
import tensorflow as tf
import keras_nlp
import music_player
from constants import KEY_ORDER, SEQ_LENGTH
from tensorflow.keras.layers import Dense


def create_music_generation_lstm_model(hp):
    """
    Builds an LSTM model with tunable hyperparameters.

    Args:
    - hp: A HyperParameters instance from Keras Tuner for defining the search space.

    Returns:
    - A compiled Keras model.
    """

    input_shape = (None, 3)
    inputs = tf.keras.Input(shape=input_shape)

    # LSTM layer
    lstm_units = hp.Int('units', min_value=32, max_value=256, step=32)
    x = tf.keras.layers.LSTM(lstm_units)(inputs)

    # Output layers
    pitch_output = Dense(128, activation='softmax', name='pitch')(x)
    step_output = Dense(1, name='step')(x)
    duration_output = Dense(1, name='duration')(x)

    model = tf.keras.Model(inputs=inputs,
                           outputs={'pitch': pitch_output, 'step': step_output, 'duration': duration_output})

    # Define loss functions for each output
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(),
        'step': mean_squared_error_with_penalty_for_negatives,
        'duration': mean_squared_error_with_penalty_for_negatives,
    }

    # Learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_weights = {
        'pitch': 0.05,
        'step': 1.0,
        'duration': 1.0,
    }
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)
    return model


def create_music_generation_transformer_model(hp):
    input_shape = (None, 3)
    inputs = tf.keras.Input(shape=input_shape)

    # Transformer layer
    d_model = hp.Int('d_model', min_value=32, max_value=256, step=32)  # Model dimensionality
    num_heads = hp.Choice('num_heads', values=[2, 4, 8])  # Number of attention heads
    transformer_block = keras_nlp.layers.TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        dropout=0.1,
    )
    x = transformer_block(inputs)

    # Output layers
    pitch_output = tf.keras.layers.Dense(128, activation='softmax', name='pitch')(x)
    step_output = tf.keras.layers.Dense(1, name='step')(x)
    duration_output = tf.keras.layers.Dense(1, name='duration')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[pitch_output, step_output, duration_output])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(),
        'step': mean_squared_error_with_penalty_for_negatives,
        'duration': mean_squared_error_with_penalty_for_negatives,
    }
    loss_weights = {
        'pitch': 0.05,
        'step': 1.0,
        'duration': 1.0
    }
    model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights)

    return model


def mean_squared_error_with_penalty_for_negatives(y_true: tf.Tensor, y_pred: tf.Tensor):
    # Calculate the mean squared error between true and predicted values.
    mse = (y_true - y_pred) ** 2
    # Apply a penalty for negative predictions to encourage positive values.
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    # Return the mean of the MSE adjusted for the penalty on negative predictions.
    return tf.reduce_mean(mse + positive_pressure)


def create_music_sequence_dataset(dataset: tf.data.Dataset, seq_length: int, vocab_size=128) -> tf.data.Dataset:
    """Generate a TensorFlow dataset of sequences for music generation."""
    # Adjust sequence length for the label.
    seq_length = seq_length + 1

    # Create windows of the specified sequence length.
    windows = dataset.window(seq_length, shift=1, stride=1, drop_remainder=True)

    # Flatten the dataset of windows into a dataset of sequences.
    flatten = lambda x: x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x / [vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]  # All but the last for inputs.
        labels_dense = sequences[-1]  # The last for labels.
        labels = {key: labels_dense[i] for i, key in enumerate(KEY_ORDER)}

        return scale_pitch(inputs), labels

    # Apply splitting and scaling to the sequences.
    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


def create_midi_training_dataset(filenames: list[str], num_files: int = None) -> tf.data.Dataset:
    """
    Creates a TensorFlow dataset for training purposes from a list of MIDI files by parsing
    each file into note sequences. This function allows for the selection of a specific number
    of files from the provided list to be included in the dataset. If the number of files to
    include is not specified, all files in the list are used.

    Parameters:
    - filenames (list[str]): List of paths to the MIDI files to be processed.
    - num_files (int, optional): The number of files to process from the list. If not specified,
      all files in the list are processed.

    Returns:
    - tf.data.Dataset: A TensorFlow dataset containing the parsed note sequences from the
      selected MIDI files. The dataset features sequences of notes represented by their
      pitch, step (time since last note), and duration.

    The function first converts MIDI files to structured note sequences and then concatenates
    them into a single pandas DataFrame. It then extracts the relevant features ('pitch',
    'step', 'duration') and creates a TensorFlow dataset from these features. This dataset
    can be used for training machine learning models on tasks such as music generation or
    prediction.
    """
    # Adjust the number of files to process based on the input parameter or use all files
    num_files = num_files if num_files is not None else len(filenames)
    all_notes = []
    for f in filenames[:num_files]:
        df_notes = music_player.midi_to_notes(f)
        all_notes.append(df_notes)

    # make one big DataFrame
    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    # create a tf.data.Dataset from the parsed notes
    train_notes = np.stack([all_notes[key] for key in KEY_ORDER], axis=1)
    print(train_notes)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    return notes_ds


def build_hypermodel(hp):
    return create_music_generation_lstm_model(hp)


def get_lstm_model() -> tf.keras.Model:
    input_shape = (SEQ_LENGTH, 3)
    learning_rate = 0.005

    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(128)(inputs)

    outputs = {
        'pitch': tf.keras.layers.Dense(128, name='pitch')(x),
        'step': tf.keras.layers.Dense(1, name='step')(x),
        'duration': tf.keras.layers.Dense(1, name='duration')(x),
    }

    model = tf.keras.Model(inputs, outputs)

    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mean_squared_error_with_penalty_for_negatives,
        'duration': mean_squared_error_with_penalty_for_negatives,
    }
    loss_weights = {
        'pitch': 0.05,
        'step': 1.0,
        'duration': 1.0,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)
    return model
