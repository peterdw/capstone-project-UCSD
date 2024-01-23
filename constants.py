# constants.py

DATASET_FOLDER      = './data/bach_cello'               # Path to dataset folder
PARSE_MIDI_FILES    = False                              # Set to False to load pre-parsed data

NOTES_FILE          = './parsed_data/notes.pkl'         # Path to save/load notes data
DURATIONS_FILE      = './parsed_data/durations.pkl'     # Path to save/load durations data
SEQ_LEN             = 50                                # Length of each sequence
BATCH_SIZE          = 256
DATASET_REPETITIONS = 1
DROPOUT_RATE        = 0.3
EMBEDDING_DIM       = 256
KEY_DIM             = 256
N_HEADS             = 5
FEED_FORWARD_DIM    = 256
EPOCHS              = 5000
GENERATE_LEN        = 50
LOAD_MODEL          = True
SPLIT_RATIO         = 0.2