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
from models.model_1 import *
from models.encodings import PositionalEncoding, PositionalEncoding1D

import tensorflow as tf

def positional_encoding(max_len, d_model):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
        if pos != 0 else np.zeros(d_model)
        for pos in range(max_len)
    ])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # Apply sine to even indices
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # Apply cosine to odd indices
    return tf.convert_to_tensor(position_enc, dtype=tf.float32)


def training(dataset_name, path_out, iter):
    path_out = path_out + dataset_name + '/' + 'iter_' + str(iter) + '/'

    # Load model
    global x_train
    input_shape = x_train.shape[1:]
    print('x_train shape 11: ', x_train.shape)
    # x_train = PositionalEncoding()(x_train)
    # print('x_train shape 22:  ', x_train.shape)
    # x_train = PositionalEncoding1D()(x_train)
    # print('x_train shape 33:  ', x_train.shape)

    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=1, mlp_units=[128], num_classes=num_classes, mlp_dropout=0.4, dropout=0.25)
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
    best_model = tf.keras.models.load_model(path_out + 'best_model.h5')
    results = best_model.evaluate(x_test, y_test)
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

    return results['accuracy']
    print('New changes')









if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Create a ArcHydro schema')

    # parser.add_argument('--input_dir', metavar='path', required=True, help='path to the UCR Archive datasets')
    # parser.add_argument('--output_dir', metavar='path', required=True, help='path to save results of training')

    # args = parser.parse_args()

    for dataset_name in  dataset_names:
        # Load dataset
        x_train, y_train, x_test, y_test, num_classes, enc = load_dataset(input_dir, dataset_name, to_categorical=True)
        accuracies = []
        for iter in range(1, num_terations+1):
            acc = training(dataset_name, path_out, iter)
            accuracies.append(acc)
        
            print('For the ' + dataset_name + ' and iteration number ' + str(iter) + ' accuracy performance is: ', acc)
        print('Overall accuracies are: ', accuracies)

        avg_acc = np.mean(np.array(accuracies))
        std = np.std(np.array(accuracies))
        res = {
            "avg_accuracy": np.round(avg_acc, 3),
            "std":     np.round(std, 3),
            "num_of_iterations": num_terations
        }
        with open(os.path.join(path_out + dataset_name + '/', 'overall_results.json'), 'w') as fp:
            json.dump(res, fp, indent=4)


    
