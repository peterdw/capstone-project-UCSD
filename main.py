# main.py

import datetime
import os
import time

import tensorflow as tf
from tensorflow.keras import callbacks

from constants import DATASET_REPETITIONS, EPOCHS, LOAD_MODEL, SPLIT_RATIO
from load_music_data import load_music_data
from timestamp_callback import TimeStampCallback
from transformer_model import get_compiled_model, plot_training_history
from transformer_utils import MusicGenerator, create_dataset


def start():
    # load the midi files
    notes, durations = load_music_data()

    # Print a sample of the parsed data
    print("Sample of Notes:", notes[:10])  # Adjust the number to display more or fewer samples
    print("Sample of Durations:", durations[:10])

    notes_seq_ds, notes_vectorize_layer, notes_vocab = create_dataset(notes)
    durations_seq_ds, durations_vectorize_layer, durations_vocab = create_dataset(durations)
    seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

    notes_vocab_size = len(notes_vocab)
    durations_vocab_size = len(durations_vocab)


    # Create the training set of sequences and the same sequences shifted by one note
    def prepare_inputs(notes, durations):
        notes = tf.expand_dims(notes, -1)
        durations = tf.expand_dims(durations, -1)
        tokenized_notes = notes_vectorize_layer(notes)
        tokenized_durations = durations_vectorize_layer(durations)
        x = (tokenized_notes[:, :-1], tokenized_durations[:, :-1])
        y = (tokenized_notes[:, 1:], tokenized_durations[:, 1:])
        return x, y




    ds = seq_ds.map(prepare_inputs).repeat(DATASET_REPETITIONS)

    model = get_compiled_model(notes_vocab_size, durations_vocab_size)

    # att_model = models.Model(inputs=[note_inputs, durations_inputs], outputs=attention_scores)

    model.summary()
    # music_generator = MusicGenerator(notes_vocab, durations_vocab)
    music_generator = MusicGenerator(model, notes_vocab, durations_vocab)

    if LOAD_MODEL:
        model.load_weights("./checkpoint/checkpoint.ckpt")
    else:
        # Create a model save checkpoint
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath="./checkpoint/checkpoint.ckpt",
            save_weights_only=True,
            save_freq="epoch",
            verbose=0,
        )

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Tokenize starting prompt

        timestamp_callback = TimeStampCallback()

        total_size = len(list(ds))
        validation_size = int(total_size * SPLIT_RATIO)
        validation_dataset = ds.take(validation_size)
        training_dataset = ds.skip(validation_size)

        history = model.fit(
            training_dataset,
            epochs=EPOCHS,
            validation_data=validation_dataset,
            callbacks=[
                model_checkpoint_callback,
                tensorboard_callback,
                music_generator,
                timestamp_callback,
            ],
        )

        plot_training_history(history)

        ########################
        # Save the final model #
        ########################

        model.save("./models/model", save_format='tf')

    ########################################
    # Generate music using the Transformer #
    ########################################

    info = music_generator.generate(
        ["START"], ["0.0"], max_tokens=50, temperature=0.5
    )

    midi_stream = info[-1]["midi"].chordify()
    # midi_stream.show()

    ############################
    # Write music to MIDI file #
    ############################

    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_dir = "./output"
    output_file_path = os.path.join(output_dir, "output-" + timestr + ".mid")

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the file
    midi_stream.write("midi", fp=output_file_path)


if __name__ == "__main__":
    start()