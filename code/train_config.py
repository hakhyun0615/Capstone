import os

# model
MODEL_NAME = 'resnet'
IMAGE_SIZE = 224

# hyperparameter
EPOCH = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 16

#shuffle True or Flase
SHUFFLE = True

# tfrecord 및 결과 저장 상위 경로
CONST_ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone' # '~/Desktop/Git/capstone/Capstone'

# tfrecord 경로 (하위폴더 구조 : (train / val)/*.tfrecords) 
CONST_DIR_PATH = f"{CONST_ROOT_PATH}/tfrecord/tfrecord_1_2_3_4_5_6_7/"

# train, test 경로
CONST_TRAIN_PATH = os.path.expanduser(CONST_DIR_PATH+'train')
CONST_VAL_PATH = os.path.expanduser(CONST_DIR_PATH+'val')

# tensorboard, model, console log 결과 저장 경로
CONST_SAVE_DIR = f"{CONST_ROOT_PATH}/result"

