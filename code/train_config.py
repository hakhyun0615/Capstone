import os
# model
MODEL_NAME = 'inception'
IMAGE_SIZE = 224

# hyperparameter
EPOCH = 100
LEARNING_RATE = 0.00001
BATCH_SIZE = 32

#shuffle True or Flase
SHUFFLE = True

# tfrecord 및 결과 저장 상위 경로
CONST_ROOT_PATH = 'C:/Users/USER/Desktop/Git/capstone/Capstone' # '~/Desktop/Git/capstone/Capstone'

# tfrecord 경로 (하위폴더 구조 : (train / val)/*.tfrecords) 
CONST_DIR_PATH = f"{CONST_ROOT_PATH}/tfrecord/tfrecord_1_2_3_4_5_6_7/"