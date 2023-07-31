import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import arange

class PositionalEncoding1D(Layer):
    def __init__(self):
        super(PositionalEncoding1D, self).__init__()

    def build(self, input_shape):

        self.B, self.TS, self.C = input_shape

        inv_freq = 1. / (10_000 ** (arange(0, self.C, 2, dtype="float32") / self.C))      
        pos_x = arange(self.TS, dtype="float32")


        sin_inp_x = tf.einsum("i,j->ij", pos_x, inv_freq)
        self.emb_x = tf.concat([tf.math.sin(sin_inp_x), tf.math.cos(sin_inp_x)], axis=-1)
        print(">>> shape sin_inp_x", sin_inp_x.shape)
        print(">>> shape emb_x", self.emb_x.shape)
        
    def call(self, inputs):
        return inputs + self.emb_x