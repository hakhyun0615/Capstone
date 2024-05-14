import os
import tensorflow as tf
from utils import *
from train_config import *
from pytz import timezone
from datetime import datetime
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import SGD  
from tensorflow.keras.metrics import Precision, Recall
from import_data import Import_data
from load_model import Load_model

'''
GPU 선택 / GPU 하나일 경우 아래 두 코드 주석 처리
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1
"'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Train_model:
    def __init__(self, train_data_path, val_data_path, model_name, image_size, batch_size, epochs): 
        self.model = Load_model(model_name, image_size)
        self.data = Import_data(image_size, batch_size, train_data_path=train_data_path, val_data_path=val_data_path)
        self.epochs = epochs

    def train(self):
        train_generator, val_generator = self.data.build_generators('train')
        optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.999, nesterov=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
        check_point = ModelCheckpoint(CHECKPOINT_FILE_PATH, verbose=1, monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True)
        tbd_callback = TensorBoard(log_dir=TSBOARD_PATH, histogram_freq=1)
        
        model = self.model.build_model()
        model.compile(loss= 'categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples//train_generator.batch_size,
            validation_data=val_generator,
            validation_steps=val_generator.samples//val_generator.batch_size,
            epochs = self.epochs,
            callbacks = [check_point, tbd_callback, early_stopping],
            verbose=1
        )

        print("Train_Accuracy : ", history.history['accuracy'][-1])
        print("Train_Precision : ", history.history['precision'][-1])
        print("Train_Recall : ", history.history['recall'][-1])

        print("Val_Accuracy : ", history.history['val_accuracy'][-1])
        print("Val_Precision : ", history.history['val_precision'][-1])
        print("Val_Recall : ", history.history['val_recall'][-1])
        
        return history

start = datetime.now(timezone('Asia/Seoul'))
print(f"Train start : {start}")

if __name__ == '__main__':
    if not os.path.exists(TSBOARD_PATH):
        os.makedirs(TSBOARD_PATH)
       
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    train_model = Train_model(train_data_path=TRAIN_DATA_PATH,
                              val_data_path=VAL_DATA_PATH,
                              model_name=MODEL_NAME,
                              image_size=IMAGE_SIZE,
                              batch_size=BATCH_SIZE,
                              epochs=EPOCHS)
     
    history = train_model.train()
    save_result(history)

end = datetime.now(timezone('Asia/Seoul'))
print(f"Train end : {end}")