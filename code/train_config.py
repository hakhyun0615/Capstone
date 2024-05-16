import os

# data
DATA = 'original_data'

# model
MODEL_NAME = 'InceptionResNet'
IMAGE_SIZE = 299
'''
if TripletNet, use weight path
'''
WEIGHT_PATH = 'path/to/your/weights.h5'

# hyperparameter
EPOCHS = 1000
<<<<<<< HEAD
LEARNING_RATE = 0.01
BATCH_SIZE = 256
=======
LEARNING_RATE = 0.0003
BATCH_SIZE = 128 # number of images in a single batch
>>>>>>> e5cfacb17234af13b22c4f629db84911e3307d56

# root path
ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone'

# train/val data path
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, DATA, 'train_data')
VAL_DATA_PATH = os.path.join(ROOT_PATH, DATA, 'val_data')

# result path
RESULT_PATH = os.path.join(ROOT_PATH, 'result')
EXPERIMENT_PATH = f"{MODEL_NAME}_{DATA}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}"
RESULT_FILE_PATH = os.path.join(RESULT_PATH, EXPERIMENT_PATH)

# tensorboard path
TSBOARD_PATH = os.path.join(RESULT_FILE_PATH, 'tensorboard')
FINETUNE_TSBOARD_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_tensorboard')

# checkpoint path
CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'checkpoint')
CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')
FINETUNE_CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_checkpoint')
FINETUNE_CHECKPOINT_FILE_PATH = os.path.join(FINETUNE_CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')