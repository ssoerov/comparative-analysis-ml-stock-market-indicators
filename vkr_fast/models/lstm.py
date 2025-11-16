from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dropout,
    Dense,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def lstm_arch(input_shape: Tuple[int, int], with_attention: bool = False) -> tf.keras.Model:
    x = Input(shape=input_shape)
    h = LSTM(64, return_sequences=with_attention)(x)
    if with_attention:
        h = MultiHeadAttention(4, 16)(h, h)
        h = LayerNormalization()(h)
        h = GlobalAveragePooling1D()(h)
    else:
        h = Dropout(0.3)(h)
    y = Dense(1)(h)
    return Model(x, y)


def fit_lstm(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 12,
    batch_size: int = 128,
) -> tf.keras.Model:
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True, verbose=0)],
        verbose=0,
    )
    return model

