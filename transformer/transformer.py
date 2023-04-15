# -*- coding: utf-8 -*-
# @Time    : 2023/3/25 21:40
# @Author  : Haonan Wang
# @File    : transformer.py
# @Software: PyCharm


import json
from keras import layers


class Transformer(object):
    """ Building the Recurrent Neural Network for Multivariate time series forecasting
    """
    def __init__(self):
        """ Initialization of the object
        """
        # Get directories name
        self.head_size=16
        self.num_heads=2
        self.ff_dim=64
        self.num_transformer_blocks=2
        self.mlp_units=[128]
        self.mlp_dropout=0.2
        self.dropout=0.2

    def transformer_encoder(self, inputs):

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu", padding="same")(x)

        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, padding="same")(x)
        return x + res



