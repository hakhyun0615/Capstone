import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from train_config import *
from tfrecord2image import tfrecord
from pytz import timezone
from datetime import datetime
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
# from tensorflow.keras.layers.experimental.preprocessing import Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LambdaCallback
from tensorflow.keras.optimizers import SGD  
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2

'''#GPU 선택 / GPU 하나일 경우 아래 두 코드 주석 처리
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Load_model:
    def __init__(self, MODEL_NAME, IMAGE_SIZE):
        self.model_name = MODEL_NAME
        self.image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # def DataAugmentation(self):
    #     return Sequential([
    #             Rescaling(1./255), 
    #             RandomFlip("horizontal"), 
    #             RandomRotation(0.2), 
    #             RandomZoom(height_factor=0.2, width_factor=0.2), 
    #             RandomTranslation(height_factor=0.2, width_factor=0.2), 
    #             RandomContrast(factor=0.2), 
    #     ])

    def build_network(self):
        if self.model_name == 'Inception':
            Inception = InceptionV3(input_shape=self.image_shape, include_top=False, weights='imagenet')
            Inception.trainable = False
            inputs = Input(self.image_shape)
            x = tf.keras.applications.inception_v3.preprocess_input(inputs)
            # x = self.DataAugmentation()(inputs)
            x = Inception(x, training=False)
            x = GlobalAveragePooling2D()(x)
            # x = Dense(1024, activation='relu')(x)
            outputs = Dense(7, activation='softmax')(x)
        elif self.model_name == 'InceptionResNet':
            InceptionResNet = InceptionResNetV2(input_shape=self.image_shape, include_top=False, weights='imagenet')
            InceptionResNet.trainable = False
            inputs = Input(self.image_shape)
            x = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
            # x = self.DataAugmentation()(inputs)
            x = InceptionResNet(x, training=False)
            x = GlobalAveragePooling2D()(x)
            # x = Dense(1024, activation='relu')(x)
            outputs = Dense(7, activation='softmax')(x)
        elif self.model_name == 'Facenet':
            pass
        else:
            raise ValueError(f"unsupported model name: {self.model_name}")

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        return model

class Train_model:
    def __init__(self, train_path, val_path, model_name, image_size): 
        with tf.device('/cpu:0'):
            if SHUFFLE == True:
                train_data = tfrecord(train_path, image_size)\
                    .shuffle(10000)\
                    .cache()\
                    .batch(BATCH_SIZE)\
                    .prefetch(tf.data.experimental.AUTOTUNE)
            else:
                train_data = tfrecord(train_path, image_size)\
                    .batch(BATCH_SIZE)\
                    .prefetch(tf.data.experimental.AUTOTUNE)

            val_data = tfrecord(val_path, image_size).batch(BATCH_SIZE)

        self.train_data = train_data
        self.val_data = val_data
        self.load_model = Load_model(model_name, image_size)

    def train(self):
        model = self.load_model.build_network()
        optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.999, nesterov=True)
        model.compile(loss= 'categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        check_point = ModelCheckpoint(CHECKPOINT_FILE, verbose=1, monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True)
        tbd_callback = TensorBoard(log_dir = SAVE_FOLDER_TSBOARD, histogram_freq = 1)

        history = model.fit(
            self.train_data,
            epochs = EPOCH,
            validation_data=self.val_data,
            callbacks = [check_point, tbd_callback, early_stopping]
        )

        print("Train_Accuracy : ", history.history['accuracy'][-1])
        print("Train_Precision : ", history.history['precision'][-1])
        print("Train_Recall : ", history.history['recall'][-1])

        print("Val_Accuracy : ", history.history['val_accuracy'][-1])
        print("Val_Precision : ", history.history['val_precision'][-1])
        print("Val_Recall : ", history.history['val_recall'][-1])

        end = datetime.now(timezone('Asia/Seoul'))
        print(f"train end : {end}")
        
        return history

    def save_accuracy(self, history):
        accuracy = history.history['accuracy']
        loss = history.history['loss']
        precision = history.history['precision']
        recall = history.history['recall']

        val_accuracy = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        val_precision = history.history['val_precision']
        val_recall = history.history['val_recall']

        epochs = range(len(accuracy))
        epoch_list = list(epochs)

        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': accuracy, 'train_loss': loss, 'train_precision': precision, 'train_recall': recall, 'validation_accuracy': val_accuracy, 'validation_loss': val_loss, 'validation_precision': val_precision, 'validation_recall': val_recall},
                          columns=['epoch', 'train_accuracy', 'train_loss', 'train_precision', 'train_recall', 'validation_accuracy', 'validation_loss', 'validation_precision', 'validation_recall'])
        df_save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'result.csv')
        df.to_csv(df_save_path, index=False, encoding='euc-kr')

        plt.plot(epochs, accuracy, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'accuracy.png')
        plt.savefig(save_path)
        plt.cla()

        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        save_path = os.path.join(os.path.expanduser(RESULT_FILE_DIR), 'loss.png')
        plt.savefig(save_path)
        plt.cla()

 
start = datetime.now(timezone('Asia/Seoul'))
print(f"train start : {start}")

if __name__ == '__main__':
    # tensorboard, model saved path
    CONST_SAVE_DIR = f"{CONST_ROOT_PATH}/result"
    EXPERIMENT_DIR = f"{MODEL_NAME}_{IMAGE_SIZE}_{EPOCH}_{LEARNING_RATE}_{BATCH_SIZE}_{SHUFFLE}"
    RESULT_FILE_DIR = os.path.join(os.path.expanduser(CONST_SAVE_DIR), EXPERIMENT_DIR)
    
    # tensorboard
    SAVE_FOLDER_TSBOARD = os.path.join(os.path.expanduser(RESULT_FILE_DIR), "tensorboard")
    if not os.path.exists(SAVE_FOLDER_TSBOARD):
        os.makedirs(SAVE_FOLDER_TSBOARD)
       
    # model checkpoint
    SAVE_FOLDER_MODEL = os.path.join(os.path.expanduser(RESULT_FILE_DIR), "model")
    if not os.path.exists(SAVE_FOLDER_MODEL):
        os.makedirs(SAVE_FOLDER_MODEL)
    CHECKPOINT_FILE = os.path.join(os.path.expanduser(SAVE_FOLDER_MODEL), 'model-{epoch:03d}-{accuracy:03f}-{val_accuracy:03f}.weights.h5')

    # train, val
    CONST_TRAIN_PATH = os.path.expanduser(CONST_DIR_PATH + 'train')
    CONST_VAL_PATH = os.path.expanduser(CONST_DIR_PATH + 'val')
    train_model = Train_model(train_path = CONST_TRAIN_PATH,
                              val_path = CONST_VAL_PATH,
                              model_name = MODEL_NAME,
                              image_size = IMAGE_SIZE)
     
    history = train_model.train()
    train_model.save_accuracy(history)
