import os

# data
DATA = 'cropped_data'

# model
MODEL_NAME = 'InceptionResNet'
IMAGE_SIZE = 299
'''
if TripletNet, use weight path
'''
WEIGHT_PATH = 'path/to/your/weights.h5'

# hyperparameter
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 256 # number of images in a single batch

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

# InceptionResNet checkpoint path
CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'checkpoint')
CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')
FINETUNE_CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_checkpoint')
FINETUNE_CHECKPOINT_FILE_PATH = os.path.join(FINETUNE_CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')

# # TripletNet checkpoint path
# CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'checkpoint')
# CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')
# FINETUNE_CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_checkpoint')
# FINETUNE_CHECKPOINT_FILE_PATH = os.path.join(FINETUNE_CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')