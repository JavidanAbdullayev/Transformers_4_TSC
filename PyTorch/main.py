import os
import argparse
import logging
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from art import *

# Import Project Modules
from utils import Setup, Initialization, Data_Verifier, Data_Loader, dataset_class
from Models.model import model_factory

logger = logging.getLogger('__main__')
parser = argparse.ArgumentParser()

#-------------------------------------------- Input and Output --------------------------------------------------------
parser.add_argument('--data_path', default='Dataset/UEA/', choices={'Dataset/UEA/', 'Dataset/Segmentation/'}, help='Data path')
parser.add_argument('--output_dir', default='Results', help='Root output directory. Must exist. Time-stamped directories will be created inside.')
parser.add_argument('--Norm', type=bool, default=False, help='Data Normalization')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Proportion of the train-set to be used as validation')
parser.add_argument('--print interval', type=int, default=10, help='Print batch info every this many batches')
# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------- Model Parameter and Hyperparameter ---------------------------------------------
parser.add_argument('--Net_Type', default=['C-T'], choices={'T', 'C-T'}, help="Network Architecture. Convolution (C)"
                                                                              "Transformers (T)")

# Transformers Parameters ------------------------------
parser.add_argument('--emb_size', type=int, default=16, help='Internal dimension of transformer embeddings')
parser.add_argument('--dim_ff', type=int, default=256, help='Dimension of dense feedforward part of transformer layer')
parser.add_argument('--num_heads', type=int, default=8, help='Number of multi-headed attention heads')
parser.add_argument('--Fix_pos_encode', choices={'tAPE', 'Learn', 'None'}, default='tAPE', help='Fix Position Embedding')
parser.add_argument('--Rel_pos_encode', choices={'eRPE', 'Vector', 'None'}, default='eRPE', help='Relative Position Embedding')


# Training Parameters/ Hyper-Parameters ----------------
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.01, help='Droupout regularization ratio')
parser.add_argument('--val_interval', type=int, default=2, help='Evaluate on validation every XX epochs. Must be >= 1')
parser.add_argument('--key_metric', choices={'loss', 'accuracy', 'precision'}, default='accuracy', help='Metric used for defining best epoch')
# ----------------------------------------------------------------------------------------------------------------------



# ------------------------------------------------------ System --------------------------------------------------------
parser.add_argument('--gpu', type=int, default='0', help='GPU index, -1 for CPU')
parser.add_argument('--console', action='store_true', help="Optimize printout for console output; otherwise for file")
parser.add_argument('--seed', default=1234, type=int, help='Seed used for splitting sets')





args = parser.parse_args()

if __name__ == '__main__':
    config = Setup(args)  # configuration dictionary
    device = Initialization(config)
    Data_Verifier(config)  # Download the UEA and HAR datasets if they are not in the directory
    All_Results = ['Datasets', 'ConvTrain']  # Use to store the accuracy of ConvTrain in e.g "Result/Dataset/UEA"

    for problem in os.listdir(config['data_path']):  # for loop on the all datsets in "data_dir" directory
        config['data_dir'] = os.path.join(config['data_path'], problem)
        print(text2art(problem, font='small'))

        # ------------------------------------ Load Data ---------------------------------------------------------------
        logger.info('Loading Data ...')
        Data = Data_Loader(config)
        print("Data['train_data'].shape: ", Data['train_data'].shape)
        print("Data['train_label'].shape: ", Data['train_label'].shape)

        print("Data['test_data'].shape: ", Data['test_data'].shape)
        print("Data['test_label'].shape: ", Data['test_label'].shape)

        train_dataset = dataset_class(Data['train_data'], Data['train_label'])
        test_dataset = dataset_class(Data['test_data'], Data['test_label'])

        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        # --------------------------------------------------------------------------------------------------------------


        # -------------------------------------------- Build Model -----------------------------------------------------
        
        dic_position_results = [config['data_dir'].split('/')[-1]]
        print('dic_position_results ', dic_position_results)

        logger.info('Creating model ...')
        config['Data_shape'] = Data['train_data'].shape
        config['num_labels'] = int(max(Data['train_label'])) + 1
        model = model_factory(config)
        model.to(device)
        

        
        for i, batch in enumerate(train_loader):

            X, targets, IDs = batch
            targets = targets.to(device)
            predictions = model(X.to(device))


            break
        
        break