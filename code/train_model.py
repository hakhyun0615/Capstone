import tensorflow as tf
from utils import *
from train_config import *
from pytz import timezone
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from import_data import Import_InceptionResNet_data, Import_TripletNet_data
from model.InceptionResNet import InceptionResNet_model
from model.TripletNet import TripletNet_model

'''
GPU 선택 / GPU 하나일 경우 아래 두 코드 주석 처리
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
'''

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Train_model:
    def __init__(self, train_data_path, val_data_path, model_name, image_size, batch_size, epochs, learning_rate, pretrained_checkpoint_path=None): 
        self.model_name = model_name   
        if self.model_name == 'TripletNet':
            self.train_triplet_generator = Import_TripletNet_data(train_data_path, batch_size, image_size)
            self.val_triplet_generator = Import_TripletNet_data(val_data_path, batch_size, image_size)
        elif self.model_name == 'InceptionResNet':
            self.train_generator, self.val_generator = Import_InceptionResNet_data(image_size, batch_size, train_data_path=train_data_path, val_data_path=val_data_path).build_generators('train')
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.image_size = image_size

    def train(self):
        callbacks = create_callbacks(CHECKPOINT_PATH, CHECKPOINT_FILE_PATH, TSBOARD_PATH)
        
        if self.model_name == 'TripletNet':
            if self.pretrained_checkpoint_path:
                print(f"Pretrained checkpoint found: {self.pretrained_checkpoint_path}")
            else:
                print("No pretrained checkpoint found")
            model = TripletNet_model(self.image_size, self.pretrained_checkpoint_path).configure_model()
            model.summary()
            model.add_loss(triplet_loss(model.outputs[0], model.outputs[1], model.outputs[2]))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate))
            history = model.fit(
                self.train_triplet_generator,
                steps_per_epoch=len(self.train_triplet_generator),
                validation_data=self.val_triplet_generator,
                validation_steps=len(self.val_triplet_generator),
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=0
            )
        elif self.model_name == 'InceptionResNet':
            fine_tune_callback = FineTune(self.learning_rate*0.1, self.train_generator, self.val_generator, self.epochs)
            
            model = InceptionResNet_model(self.image_size).configure_model()
            model.summary()
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=self.learning_rate),
                metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
            )
            history = model.fit(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples//self.train_generator.batch_size,
                validation_data=self.val_generator,
                validation_steps=self.val_generator.samples//self.val_generator.batch_size,
                epochs=self.epochs,
                callbacks=callbacks+[fine_tune_callback],
                verbose=0
            )
        
        return history

start = datetime.now(timezone('Asia/Seoul'))
print(f"Train start : {start}")

if __name__ == '__main__':
    if MODEL_NAME == 'TripletNet':
        train_model = Train_model(train_data_path=TRAIN_DATA_PATH,
                                  val_data_path=VAL_DATA_PATH,
                                  model_name=MODEL_NAME,
                                  image_size=IMAGE_SIZE,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  learning_rate=LEARNING_RATE,
                                  pretrained_checkpoint_path=PRETRAINED_CHECKPOINT_PATH)
    elif MODEL_NAME == 'InceptionResNet':
        train_model = Train_model(train_data_path=TRAIN_DATA_PATH,
                                  val_data_path=VAL_DATA_PATH,
                                  model_name=MODEL_NAME,
                                  image_size=IMAGE_SIZE,
                                  batch_size=BATCH_SIZE,
                                  epochs=EPOCHS,
                                  learning_rate=LEARNING_RATE)
     
    history = train_model.train()
    save_result(history)

end = datetime.now(timezone('Asia/Seoul'))
print(f"Train end : {end}")