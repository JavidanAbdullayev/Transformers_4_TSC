import os
import json
import time
import argparse

# Data manipulation
import numpy as np

# Constants
from utils.constants import dataset_names
from utils.constants import Iterations

# Functions
from utils.utils import load_dataset
from utils.utils import plot_metrics

# Tenslorflow models
from models.model_1 import *

# Encodings
from models.encodings import PositionalEncoding, PositionalEncoding1D

# Tensorflow imports
import tensorflow as tf
    

if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='Retrieve I/O directories')

    parser.add_argument('--input_dir', metavar='path', required=True, help='path to the UCR Archive datasets')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path to save results of training')

    args = parser.parse_args()

    dataset_name = dataset_names[0]
    path_out = args.output_dir + dataset_name + '/' + 'Iteration_' + str(iter) + '/'
    
    if not os.path.exists(path_out) :   
        try:
            os.mkdir(path_out)
        except:
            pass

    print("\n >> Training will be saved in '{}'".format(args.output_dir))

    # Load data
    x_train, y_train, x_test, y_test, num_classes, enc = load_dataset(args.input_dir, dataset_name, to_categorical=True)
    lay_enc = PositionalEncoding()
    x_train = lay_enc(x_train)

    # Load model
    input_shape = x_train.shape[1:]

    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], num_classes=num_classes, mlp_dropout=0.4, dropout=0.25)

    # Specify training parameters 
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['categorical_accuracy'])

    model.summary()

    # Train the model
    epochs = 1000

    # Start training
    start_time = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=64)
    end_time = time.time()

    






