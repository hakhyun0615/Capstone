import os

# tfrecord 및 결과 저장 상위 경로
CONST_ROOT_PATH = '~/Desktop/Git/capstone/Capstone'

# tfrecord 경로 (하위폴더 구조 : (train / val)/*.tfrecords) 
CONST_DIR_PATH = f"{CONST_ROOT_PATH}/tfrecord/tfrecord_1_2_3_4_5_6_7"

# tensorboard, model, console log 결과 저장 경로
CONST_SAVE_DIR = f"{CONST_ROOT_PATH}/result"

CONST_TRAIN_PATH = os.path.expanduser(CONST_DIR_PATH + 'train')
CONST_VAL_PATH = os.path.expanduser(CONST_DIR_PATH)

# model
MODEL_NAME = 'inception_v4'

# hyperparameter
EPOCH = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 16
IMAGE_SIZE = 224

#shuffle True or Flase
SHUFFLE = True

# fixed hyperparameter
CLASS_COUNT = 7
ACTIVATION = 'softmax'
LOSS = 'categorical_crossentropy'