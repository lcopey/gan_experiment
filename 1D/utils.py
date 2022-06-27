from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras import Model
import tensorflow as tf


def build_dense(input_dim, out_dim, layers: list, activation, name, dropout=None, out_activation=None):
    input_layer = Input(shape=input_dim, dtype=tf.float32)
    inner_layer = input_layer

    for n, layer in enumerate(layers):
        inner_layer = Dense(layer, activation=activation, dtype=tf.float32)(inner_layer)
        if dropout:
            inner_layer = Dropout(dropout, dtype=tf.float32)(inner_layer)

    out_layer = Dense(out_dim, activation=out_activation, dtype=tf.float32)(inner_layer)

    return Model(input_layer, out_layer, name=name)
