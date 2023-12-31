# capstone-project-UCSD

The `main.py` script in the "capstone-project-UCSD" GitHub repository is a Python program for generating music using a Transformer-based model. It performs the following functions:

1. **Data Loading and Preprocessing:** Loads music data (notes and durations) and preprocesses it into suitable formats for training.

2. **Model Creation:** Constructs a Transformer-based neural network model with embedding and attention mechanisms, tailored for music generation.

3. **Training and Saving:** If a pre-trained model is not loaded, it trains the model on the processed data, saving checkpoints and logging progress. The final model is saved for future use.

4. **Music Generation:** Uses the trained model to generate music sequences, converting them into a MIDI format.

5. **Output Handling:** Saves the generated music sequences as MIDI files, with filenames including timestamps to distinguish between different outputs.

This program is a comprehensive tool for automated music creation, leveraging advanced machine learning techniques.

# datasets:

- Bach Cello dataset : http://www.jsbach.net/midi/midi_solo_cello.html
- The MAESTRO Dataset : https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip