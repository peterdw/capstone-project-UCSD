# timestamp_callback.py

import datetime
from tensorflow.keras.callbacks import Callback

class TimeStampCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Timestamp at end of Epoch {epoch}: {current_time}")
