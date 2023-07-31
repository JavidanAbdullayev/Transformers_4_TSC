import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import arange
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
  def __init__(self, num_hiddens=1, dropout=0.5, max_len=1882, **kwargs):
          super(PositionalEncoding, self).__init__()
          self.dropout = layers.Dropout(dropout)
          # Create a long enough `P`
          self.P = np.zeros((1, max_len, num_hiddens))
          X = np.arange(max_len).reshape(-1, 1) / np.power(
              10000, np.arange(0, num_hiddens, 2) / num_hiddens)
          self.P[:, :, 0::2] = np.sin(X)
          self.P[:, :, 1::2] = np.cos(X)

          self.P = tf.cast(tf.convert_to_tensor(self.P), tf.float32)
  def get_config(self):
        return super().get_config()

  def call(self, inputs, **kwargs):
      inputs = inputs + self.P[:, :inputs.shape[1], :]
      return self.dropout(inputs)
  

class PositionalEncoding1D(Layer):
    def __init__(self):
        super(PositionalEncoding1D, self).__init__()

    def build(self, input_shape):

        self.B, self.TS, self.C = input_shape

        inv_freq = 1. / (10_000 ** (arange(0, self.C, 2, dtype="float32") / self.C))      
        pos_x = arange(self.TS, dtype="float32")


        sin_inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
        self.emb_x = tf.add(tf.math.sin(sin_inp_x), tf.math.cos(sin_inp_x))
        print(">>> shape sin_inp_x", sin_inp_x.shape)
        print(">>> shape emb_x", self.emb_x.shape)

        #emb = tf.zeros((self.TS, self.C))
        #emb[:, :self.C] = emb_x
        #self.pos_enc = tf.repeat(emb[None,: , :self.C], (self.B, 1, 1))

    def call(self, inputs):
        print('Inputs shape: ', inputs.shape)
        print('Embedding shape: ', self.emb_x.shape)
        return inputs + self.emb_x
