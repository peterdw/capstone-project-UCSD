import tensorflow as tf
from tensorflow.keras import layers, models, losses
from constants import EMBEDDING_DIM, N_HEADS, KEY_DIM, FEED_FORWARD_DIM
from transformer_utils import TokenAndPositionEmbedding, TransformerBlock
import matplotlib.pyplot as plt


def get_uncompiled_model(notes_vocab_size, durations_vocab_size):
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
        outputs=[note_outputs, duration_outputs],
    )
    return model


def get_compiled_model(notes_vocab_size, durations_vocab_size):
    model = get_uncompiled_model(notes_vocab_size, durations_vocab_size)
    model.compile(
        optimizer='adam',
        loss=[
            losses.SparseCategoricalCrossentropy(),
            losses.SparseCategoricalCrossentropy()
        ],
        metrics={
            'note_outputs': ['accuracy'],
            'duration_outputs': ['accuracy']
        }
    )
    return model


def plot_training_history(history):
    # Extracting values
    note_outputs_loss = history.history['note_outputs_loss']
    duration_outputs_loss = history.history['duration_outputs_loss']
    note_outputs_accuracy = history.history['note_outputs_accuracy']
    duration_outputs_accuracy = history.history['duration_outputs_accuracy']

    # Creating subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    # Plotting Loss
    axes[0].plot(note_outputs_loss, label='Note Outputs Loss')
    axes[0].plot(duration_outputs_loss, label='Duration Outputs Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plotting Accuracy
    axes[1].plot(note_outputs_accuracy, label='Note Outputs Accuracy')
    axes[1].plot(duration_outputs_accuracy, label='Duration Outputs Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    # Show the plots
    plt.tight_layout()
    # plt.show()
    plt.savefig('training_loss_accuracy.png')
