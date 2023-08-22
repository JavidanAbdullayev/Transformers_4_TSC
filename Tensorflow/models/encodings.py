import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import arange
from tensorflow.keras import layers
import numpy as np

class PositionalEncoding(layers.Layer):
  def __init__(self, num_hiddens=1, dropout=0.5, max_len=256, **kwargs):
          super(PositionalEncoding, self).__init__()
          
          # Create a long enough `P`
          self.P = np.zeros((1, max_len, num_hiddens))
          X = np.arange(max_len).reshape(-1, 1) / np.power(max_len, np.arange(0, num_hiddens, 2) / num_hiddens)
          self.P[:, :, 0::2] = np.sin(X)
          self.P[:, :, 1::2] = np.cos(X)

          self.P = tf.cast(tf.convert_to_tensor(self.P), tf.float32)
  def get_config(self):
        return super().get_config()

  def call(self, inputs, **kwargs):
      inputs = inputs + self.P[:, :inputs.shape[1], :]
      return inputs
  

class PositionalEncoding1D(Layer):
    def __init__(self):
        super(PositionalEncoding1D, self).__init__()

    def build(self, input_shape):

        self.B, self.TS, self.C = input_shape
        print('self.B: ', self.B,  '  self.TS: ', self.TS, '  self.C: ', self.C)
        inv_freq = 1. / (10_000 ** (arange(0, self.C, 2, dtype="float32") / self.C))     
        print('inv_freq: ', inv_freq.shape)
        pos_x = arange(self.TS, dtype="float32")
        
        inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_x = tf.math.sin(inp_x[0::2])
        cos_inp_x = tf.math.cos(inp_x[1::2])

        emb = tf.zeros((self.TS, self.C))

        even_indices = tf.range(0, self.TS, 2)
        odd_indices = tf.range(1, self.TS, 2)

        emb = tf.tensor_scatter_nd_update(emb, tf.expand_dims(even_indices, axis=1), sin_inp_x)
        emb = tf.tensor_scatter_nd_update(emb, tf.expand_dims(odd_indices, axis=1), cos_inp_x)

        self.emb_x = emb
        # sin_inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
        # inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
        # sin_inp_x = tf.math.sin(inp_x)

        # self.emb_x = tf.add(tf.math.sin(sin_inp_x), tf.math.cos(sin_inp_x))
        # print(">>> shape sin_inp_x", sin_inp_x.shape)
        # print(">>> shape emb_x", self.emb_x.shape)

        #emb = tf.zeros((self.TS, self.C))
        #emb[:, :self.C] = emb_x
        #self.pos_enc = tf.repeat(emb[None,: , :self.C], (self.B, 1, 1))

    def call(self, inputs):
        print('Inputs shape: ', inputs.shape)
        print('Embedding shape: ', self.emb_x.shape)
        return inputs + self.emb_x
