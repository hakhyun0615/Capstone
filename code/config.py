import os

# data
DATA = 'cropped_data' # original_data, cropped_data
NUM_CLASS = 7

# model
MODEL_NAME = 'EfficientNetB7' # EfficientNetB7, TripletNet
IMAGE_SIZE = 224

# hyperparameter
EPOCHS = 1000
LEARNING_RATE = 0.0003
BATCH_SIZE = 8 # number of images in a single batch

# root path
ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone'

# train/val/test data path
DATA_PATH = os.path.join(ROOT_PATH, DATA)
TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train_data')
VAL_DATA_PATH = os.path.join(DATA_PATH, 'val_data')
TEST_DATA_PATH = os.path.join(DATA_PATH, 'test_data')

# train result path
RESULT_PATH = os.path.join(ROOT_PATH, 'train_result')
EXPERIMENT_PATH = f"{MODEL_NAME}_{DATA}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}"
RESULT_FILE_PATH = os.path.join(RESULT_PATH, EXPERIMENT_PATH)

# test result path
TEST_RESULT_PATH = os.path.join(ROOT_PATH, 'test_result')
TEST_RESULT_FILE_PATH = os.path.join(TEST_RESULT_PATH, EXPERIMENT_PATH)

# checkpoint path
if MODEL_NAME == 'TripletNet':
    CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'checkpoint')
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}.h5')
    
    PRETRAINED_CHECKPOINT_PATH_1 = os.path.join(RESULT_PATH, f'EfficientNetB7_{DATA}_{IMAGE_SIZE}_{EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}', 'checkpoint')
    PRETRAINED_CHECKPOINT_PATH_2 = os.listdir(PRETRAINED_CHECKPOINT_PATH_1)[-1]
    PRETRAINED_CHECKPOINT_PATH = os.path.join(PRETRAINED_CHECKPOINT_PATH_1, PRETRAINED_CHECKPOINT_PATH_2)
elif MODEL_NAME == 'EfficientNetB7':
    CHECKPOINT_PATH = os.path.join(RESULT_FILE_PATH, 'checkpoint')
    CHECKPOINT_FILE_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint-{epoch:03d}-{loss:03f}-{val_loss:03f}-{accuracy:03f}-{val_accuracy:03f}.h5')