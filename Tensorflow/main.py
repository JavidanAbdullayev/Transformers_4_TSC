# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
import os
import argparse
from utils.constants import *
from utils.utils import load_dataset, save_loss_and_accuracy_fig, plot_metrics

import tensorflow as tf


def test(input_dir, output_dir):
    print('\n\nProgram started')
    print('Input directory: ', input_dir)
    print('Schema: ', output_dir)
    print('Program stopped\n\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')

    parser.add_argument('--input_dir', metavar='path', required=True, help='path to the UCR Archive datasets')
    parser.add_argument('--output_dir', metavar='path', required=True, help='path to save results of training')

    args = parser.parse_args()

    test(input_dir=args.input_dir, output_dir=args.output_dir)

    # Load dataset
    x_train, y_train, x_test, y_test, num_classes, enc = load_dataset(args.input_dir, 'ArrowHead', to_categorical=True)

    # Load model
    from models.model_1 import *
    input_shape = x_train.shape[1:]
    # from models.encodings import PositionalEncoding
    # x_train = PositionalEncoding()(x_train)
    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], num_classes=num_classes, mlp_dropout=0.4, dropout=0.25)
    model.summary()

    # Specify training parameters 
    model.compile(loss=TRAIN_PARAMS['loss'], optimizer=TRAIN_PARAMS['optimizer'], metrics=TRAIN_PARAMS['metrics'])

    # Initializing training callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-4),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(path_out, 'best_model.h5'),
            monitor='loss',
            save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(path_out, "history.csv"), separator=",", append=True),
    ]

    # Train the model
    start_time = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=TRAIN_PARAMS['epochs'], batch_size=TRAIN_PARAMS['batch_size'], callbacks=callbacks)
    end_time = time.time()

    # Evaluate the model
    results = model.evaluate(x_test, y_test)
    results = {
        "accuracy": results[1],
        "loss":     results[0],
        "training_time": end_time-start_time
    }

    # Save losses and accuracies plots for training and validation data
    if not os.path.exists(path_out):
        try:
            os.makedirs(path_out)
        except:
            pass
    plot_metrics(history.history, path_out, TRAIN_PARAMS['epochs'])


    with open(os.path.join(path_out, 'results.json'), 'w') as fp:
        json.dump(results, fp, indent=4)


    # Training has ended
    os.mkdir(os.path.join(path_out, 'Done'))

    print('New changes')

