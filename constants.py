# constants.py

CELLO_DATASET_FOLDER = './data/bach_cello'  # Path to dataset folder

PARSE_MIDI_FILES = True  # Set to False to load pre-parsed data

NOTES_FILE = './parsed_data/notes.pkl'  # Path to save/load notes data
DURATIONS_FILE = './parsed_data/durations.pkl'  # Path to save/load durations data
SEQ_LEN = 50  # Length of each sequence
BATCH_SIZE = 256
DATASET_REPETITIONS = 1
DROPOUT_RATE = 0.3
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 5
FEED_FORWARD_DIM = 256
EPOCHS = 5000
GENERATE_LEN = 50
LOAD_MODEL = True
SPLIT_RATIO = 0.2

'''
MAESTRO CONSTANTS 
'''

MAESTRO_DATASET_FOLDER = './data/maestro-v3.0.0'  # Path to dataset folder
_SAMPLING_RATE = 16000
PLAYBACK_DURATION_SECONDS = 10  # Adjust this value to change the playback duration
KEY_ORDER = ['pitch', 'step', 'duration']
BEST_LSTM_MODEL = 'saved_models/best_lstm_model'
SEQ_LENGTH = 25
PITCH_VOCAB_SIZE = 128