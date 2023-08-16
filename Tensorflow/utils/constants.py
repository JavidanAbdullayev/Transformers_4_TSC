# dataset_names = ['ArrowHead', 'TwoPatterns',  'Yoga', 'Car', 'Ham']
dataset_names = ['ArrowHead', ]

path_out = '/home/javidan/Codes/transformers/results/plots/'

# Training parameters
TRAIN_PARAMS = {
    'epochs'      :  3,
    'batch_size'  : 64,
    'optimizer'   : 'adam',
    'loss'        : 'categorical_crossentropy',
    'metrics'     : ['categorical_accuracy']
 }



Iterations = 5