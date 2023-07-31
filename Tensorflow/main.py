# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import argparse
from utils.constants import dataset_names
from utils.utils import load_dataset



def test(input_dir, output_dir):
    print('Program started')
    print('Input directory: ', input_dir)
    print('Schema: ', output_dir)
    print('Program stopped')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')

    parser.add_argument('--input_dir', metavar='path', required=True, help='path to the UCR Archive datasets')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path to save results of training')

    args = parser.parse_args()

    test(input_dir=args.input_dir, output_dir=args.output_dir)

    # Load data
    x_train, y_train, x_test, y_test, num_classes, enc = load_dataset(args.input_dir, 'ArrowHead', to_categorical=True)

    # Load model
    from models.model_1 import *
    input_shape = x_train.shape[1:]
    # from models.encodings import PositionalEncoding
    # x_train = PositionalEncoding()(x_train)
    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], num_classes=num_classes, mlp_dropout=0.4, dropout=0.25)

    # Specify training parameters 
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=['categorical_accuracy'])
    callbacks = [keras.callbacks.EarlyStopping(patience=200, restore_best_weights=True)]
    model.summary()

    # Train the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1000, batch_size=64, )

    # Evaluate the model
    model.evaluate(x_test, y_test)

    print('New changes')

