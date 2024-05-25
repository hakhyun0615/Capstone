import os

# data
DATA = 'original_data'

# model
MODEL_NAME = 'InceptionResNet' # InceptionResNet, TripletNet
IMAGE_SIZE = 224

# hyperparameter
EPOCHS = 1000
LEARNING_RATE = 0.1
BATCH_SIZE = 512 # number of images in a single batch

# root path
ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone'

# train/val data path
TRAIN_DATA_PATH = os.path.join(ROOT_PATH, DATA, 'train_data')
VAL_DATA_PATH = os.path.join(ROOT_PATH, DATA, 'val_data')

# result path
RESULT_PATH = os.path.join(ROOT_PATH, 'train_result')
EXPERIMENT_PATH = f"{MODEL_NAME}_{DATA}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}"
RESULT_FILE_PATH = os.path.join(RESULT_PATH, EXPERIMENT_PATH)

# tensorboard path
TSBOARD_PATH = os.path.join(RESULT_FILE_PATH, 'initial_tensorboard')
FINETUNE_TSBOARD_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_tensorboard')

# checkpoint path
if MODEL_NAME == 'InceptionResNet':
    CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'initial_checkpoint')
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')
    FINETUNE_CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_checkpoint')
    FINETUNE_CHECKPOINT_FILE_PATH = os.path.join(FINETUNE_CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')
elif MODEL_NAME == 'TripletNet':
    CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'initial_checkpoint')
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')
    FINETUNE_CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'finetune_checkpoint')
    FINETUNE_CHECKPOINT_FILE_PATH = os.path.join(FINETUNE_CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.weights.h5')
    
    PRETRAINED_WEIGHT_PATH_1 = os.path.join(RESULT_PATH, f'InceptionResNet_{DATA}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}', 'finetune_checkpoint')
    PRETRAINED_WEIGHT_PATH_2 = os.listdir(PRETRAINED_WEIGHT_PATH_1)[-1]
    PRETRAINED_WEIGHT_PATH = os.path.join(PRETRAINED_WEIGHT_PATH_1, PRETRAINED_WEIGHT_PATH_2)