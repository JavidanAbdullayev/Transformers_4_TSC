# dataset_names = ['ArrowHead', 'TwoPatterns',  'Yoga', 'Car', 'Ham']
dataset_names = ['ArrowHead', 'Yoga', 'GunPointOldVersusYoung', 'OliveOil', 'Wine', 'InsectWingbeatSound', 'FaceAll', 'Earthquakes']
input_dir = '/home/javidan/Codes/UCRArchive_2018'
path_out = '/home/javidan/Codes/transformers/results/'

# Training parameters
TRAIN_PARAMS = {
    'epochs'      :  1500,
    'batch_size'  : 64,
    'optimizer'   : 'adam',
    'loss'        : 'categorical_crossentropy',
    'metrics'     : ['categorical_accuracy']
 }

num_terations = 5