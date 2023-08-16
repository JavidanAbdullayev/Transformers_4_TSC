

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_labels(y_train, y_test, format=None):
    """Encoding for labels
    Format:
    - None: Default encoding -> 0, 1, 2, ..., use with sparse categorical cross entropy
    - OHE: One Hot Encoding, use with categorical cross entropy

    Args:
        y_train (np.array): Train labels
        y_test (np.array): Test labels
        format (string): The format for the encoding. Defaults to None.

    Returns:
        (np.array, np.array, int): The ecnode label y_train, encode label y_test and number of classes
    """

    # init encoder
    if format == None:
        encoder = LabelEncoder()
    elif format == 'OHE':
        encoder = OneHotEncoder(sparse=False)
        # Change data format by expanding dimension
        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
    else:
        print('Error wrong parameter, either None or OHE expected!')
        exit()

    # Concat train and test labels
    y_train_test = np.concatenate((y_train, y_test), axis=0)
    # Count num_classes
    num_classes = len(np.unique(y_train_test))
    # Fit the encoder and transform data
    new_y_train_test = encoder.fit_transform(y_train_test)
    # Resplit the train and test labels
    new_y_train = new_y_train_test[0:len(y_train)]
    new_y_test = new_y_train_test[len(y_train):]

    return new_y_train, new_y_test, num_classes, encoder



def z_norm(data):
    """Z-Normalization for the data

    Args:
        data (np.array): training data with input features

    Returns:
        (np.array): data in a normalized format
    """

    std_ = data.std(axis=1, keepdims=True)
    std_[std_ == 0] = 1.0
    data_prep = (data - data.mean(axis=1, keepdims=True)) / std_
    
    return data_prep


def read_ucr(filename, delimiter):
    """Read UCR tsv file and split data

    Args:
        file_name (string): Path to a specific UCR dataset to read the file
        delimiter_ (string): Charater or a set of character used to seperate individual values (fields) 

    Returns:
        (np.array, np.array): input features (X) and their corresponding labels (Y)
    """
    data = np.loadtxt(fname=filename, delimiter=delimiter)
    X = data[:, 1:]
    Y = data[:, 0]
    
    return X, Y.astype(int)

def load_dataset(input_dir, dataset_name, to_categorical=True):
    """Read, load and transform an UCR dataset

    Args:
        input_dir (string): path to a specific UCR Archive dataset
        dataset_names (string): name of dataset that is processing
        to_categorical (bool, optional): The format for the encoding. Defaults to True.

    Returns:
        (np.array, np.array, np.array, np.array, int, enc): x_train, y_train, x_test, y_test, number of classes and encoding
    """
    # Load raw data
    x_train, y_train = read_ucr(input_dir + '/' + dataset_name + '/' + dataset_name + '_TRAIN.tsv', delimiter='\t')
    x_test, y_test = read_ucr(input_dir + '/' + dataset_name + '/' + dataset_name + '_TEST.tsv', delimiter='\t')

    # Change shape to be appropriate with Tensorflow library
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Z normalize train and test data
    x_train = z_norm(x_train)
    x_test = z_norm(x_test)

    # Label encoding
    encoder = None
    if to_categorical:
        # Classes to OneHotEncoding
        y_train, y_test, num_classes, enc = encode_labels(y_train, y_test, format='OHE')
    else:
        # Sparse encoding, from 0 to num_classes
        y_train, y_test, num_classes, enc = encode_labels(y_train, y_test)

    # Return data with encoder
    return x_train, y_train, x_test, y_test, num_classes, enc



def save_loss_and_accuracy_fig(history, epochs, path):
    import matplotlib.pyplot as plt
    
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    plt.figure(1)
    plt.plot([i+1 for i in range(epochs)], train_losses, label='Training loss')
    plt.plot([i+1 for i in range(epochs)], val_losses, label='Validation loss')
    plt.xlabel('Epoches')
    plt.ylabel('Losses')
    plt.legend()
    plt.savefig(f'{path}train_val_losses_{epochs}.pdf')


    train_losses = history.history['categorical_accuracy']
    val_losses = history.history['val_categorical_accuracy']
    
    plt.figure(2)
    plt.plot([i+1 for i in range(epochs)], train_losses, label='Training Accuracy')
    plt.plot([i+1 for i in range(epochs)], val_losses, label='Validation Accuracy')
    plt.xlabel('Epoches')
    plt.ylabel('Accuracies')
    plt.legend()
    plt.savefig(f'{path}train_val_accuracies_{epochs}.pdf')


def plot_metrics(history, save_path, epochs, metrics=None, with_val=True):
    if metrics == None:
        # Plot all metrics
        metrics = history.keys()
        print('metrics: ', metrics)
    # Remove val metrics
    regex = re.compile(r'val_.*')
    metrics = [m for m in metrics if not regex.match(m)]
    print('metrics later: ', metrics)
    for m in metrics:
        if m in history.keys():# and 'val_'+m in history.keys():
            # Plot
            plt.figure(figsize=(6, 3))
            plt.plot(history[m], color='blue')
            
            if with_val and m not in ['lr']:
                plt.plot(history['val_' + m], color='darkorange')
                plt.legend(['train', 'test'], loc='best')
            else:
                plt.legend(['train'], loc='best')
                
            plt.xlabel('Epochs')
            plt.ylabel(m)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'plt_'+ m + '_' + str(epochs) + '.pdf'))
            plt.close()