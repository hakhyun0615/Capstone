import os
# data
DATA = 'original_data'

# model
MODEL_NAME = 'InceptionResNet'
IMAGE_SIZE = 299

# hyperparameter
EPOCHS = 500
LEARNING_RATE = 0.0001
BATCH_SIZE = 64

# root path
ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone' # '~/Desktop/Git/capstone/Capstone'

# train/val/test data path
TRAIN_DATA_PATH = f"{ROOT_PATH}/{DATA}/train_data/"
VAL_DATA_PATH = f"{ROOT_PATH}/{DATA}/val_data/"
TEST_DATA_PATH = f"{ROOT_PATH}/{DATA}/test_data/"

# result path
RESULT_PATH = f"{ROOT_PATH}/result"
EXPERIMENT_PATH = f"{MODEL_NAME}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}"
RESULT_FILE_PATH = os.path.join(os.path.expanduser(RESULT_PATH), EXPERIMENT_PATH)

# tensorboard path
TSBOARD_PATH = os.path.join(os.path.expanduser(RESULT_FILE_PATH), "tensorboard")

# checkpoint path
CHECKPOINT_PATH = os.path.join(os.path.expanduser(RESULT_FILE_PATH), "checkpoint")
CHECKPOINT_FILE = os.path.join(os.path.expanduser(CHECKPOINT_PATH), 'model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')