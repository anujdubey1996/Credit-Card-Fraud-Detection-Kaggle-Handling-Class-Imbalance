import configuration as config
import utilities as utils

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

class Trainer():
    def __init__(self, model, data):        
        self.history = model.fit(
            data.X_train,
            data.y_train,
            batch_size=config.BATCH_SIZE,
            epochs=20,
            validation_split=0.05, 
            shuffle=True,
            verbose=0
        )
        