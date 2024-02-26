from datetime import datetime

import pandas as pd
import tensorflow as tf
import numpy as np

import music_player


def predict_next_note(
        notes: np.ndarray,
        model: tf.keras.Model,
        temperature: float = 1.0) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

    assert temperature > 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    # pitch_logits, step, duration = predictions
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    # `step` and `duration` values should be non-negative
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def create_midi(notes: pd.DataFrame):
    # Format the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    example_file = f'output/generated_{timestamp}.midi'

    _ = music_player.notes_to_midi(
        notes, out_file=example_file)

    print(f"MIDI file saved as: {example_file}")
