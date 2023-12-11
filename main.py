# main.py

from load_music_data import load_music_data
from timestamp_callback import TimeStampCallback
from transformer_utils import MusicGenerator, TokenAndPositionEmbedding, TransformerBlock, create_dataset
import tensorflow as tf
from constants import DATASET_REPETITIONS, EMBEDDING_DIM, EPOCHS, FEED_FORWARD_DIM, KEY_DIM, LOAD_MODEL, N_HEADS
from tensorflow.keras import layers, models, losses, callbacks
import time
import os

notes, durations = load_music_data()

# Print a sample of the parsed data
# print("Sample of Notes:", notes[:10])  # Adjust the number to display more or fewer samples
# print("Sample of Durations:", durations[:10])

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

tpe = TokenAndPositionEmbedding(notes_vocab_size, 32)

note_inputs = layers.Input(shape=(None,), dtype=tf.int32)
durations_inputs = layers.Input(shape=(None,), dtype=tf.int32)
note_embeddings = TokenAndPositionEmbedding(
    notes_vocab_size, EMBEDDING_DIM // 2
)(note_inputs)
duration_embeddings = TokenAndPositionEmbedding(
    durations_vocab_size, EMBEDDING_DIM // 2
)(durations_inputs)
embeddings = layers.Concatenate()([note_embeddings, duration_embeddings])
x, attention_scores = TransformerBlock(
    N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, name="attention"
)(embeddings)
note_outputs = layers.Dense(
    notes_vocab_size, activation="softmax", name="note_outputs"
)(x)
duration_outputs = layers.Dense(
    durations_vocab_size, activation="softmax", name="duration_outputs"
)(x)
model = models.Model(
    inputs=[note_inputs, durations_inputs],
    outputs=[note_outputs, duration_outputs],  # attention_scores
)
model.compile(
    "adam",
    loss=[
        losses.SparseCategoricalCrossentropy(),
        losses.SparseCategoricalCrossentropy(),
    ],
)
att_model = models.Model(
    inputs=[note_inputs, durations_inputs], outputs=attention_scores
)

model.summary()
#music_generator = MusicGenerator(notes_vocab, durations_vocab)
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

    tensorboard_callback = callbacks.TensorBoard(log_dir="./logs")

    # Tokenize starting prompt
    
    timestamp_callback = TimeStampCallback()

    model.fit(
        ds,
        epochs=EPOCHS,
        callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            music_generator,
            timestamp_callback,
        ],
    )

    ########################
    # Save the final model #
    ########################

    model.save("./models/model", save_format='tf')

########################################
# Generate music using the Transformer #
########################################

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