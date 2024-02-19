# main.py
import datetime
import os
import time

import tensorflow as tf
from tensorflow.keras import callbacks

from constants import DATASET_REPETITIONS, EPOCHS, LOAD_MODEL, SPLIT_RATIO
from input_preparation import prepare_inputs
from load_music_data import load_music_data
from timestamp_callback import TimeStampCallback
from transformer_model import get_compiled_model, plot_training_history
from transformer_utils import MusicGenerator, create_dataset


def start():
    # load the midi files and retrieve the notes and durations
    # notes : [
    #           'START G:major 3/4TS rest G3 G3 D3 G2 B3 C4 D4 C4 B3 A3 B3 D3 G2 G3 A3 B3 G3 E3 C3 C2 A3 B3 C4 B3 A3 G3 F#3 D3 D2 D3 E3 F#3 G3 A3 B3 C4 B3 C4 A3 C4 B3 C4 A3 D3 C4 B3',
    #           'G:major 3/4TS rest G3 G3 D3 G2 B3 C4 D4 C4 B3 A3 B3 D3 G2 G3 A3 B3 G3 E3 C3 C2 A3 B3 C4 B3 A3 G3 F#3 D3 D2 D3 E3 F#3 G3 A3 B3 C4 B3 C4 A3 C4 B3 C4 A3 D3 C4 B3 A3',
    #           '3/4TS rest G3 G3 D3 G2 B3 C4 D4 C4 B3 A3 B3 D3 G2 G3 A3 B3 G3 E3 C3 C2 A3 B3 C4 B3 A3 G3 F#3 D3 D2 D3 E3 F#3 G3 A3 B3 C4 B3 C4 A3 C4 B3 C4 A3 D3 C4 B3 A3 B3',
    #           ...
    #         ]
    # durations: [
    #               '0.0 0.0 0.0 2.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25',
    #               '0.0 0.0 2.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25',
    #               '0.0 2.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.5 0.5 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.5 0.5 0.5 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25',
    #               ...
    #            ]
    notes, durations = load_music_data()

    # Print a sample of the parsed data
    print("Sample of Notes:", notes[:10])  # Adjust the number to display more or fewer samples
    print("Sample of Durations:", durations[:10])

    notes_seq_ds, notes_vectorize_layer, notes_vocab = create_dataset(notes)
    durations_seq_ds, durations_vectorize_layer, durations_vocab = create_dataset(durations)
    seq_ds = tf.data.Dataset.zip((notes_seq_ds, durations_seq_ds))

    notes_vocab_size = len(notes_vocab)
    durations_vocab_size = len(durations_vocab)

    ds = seq_ds.map(lambda notes, durations: prepare_inputs(notes, durations, notes_vectorize_layer,
                                                            durations_vectorize_layer)).repeat(DATASET_REPETITIONS)

    model = get_compiled_model(notes_vocab_size, durations_vocab_size)

    model.summary()
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
    # sys.exit()

    info = music_generator.generate(
        ["START"], ["0.0"], max_tokens=250, temperature=0.5
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