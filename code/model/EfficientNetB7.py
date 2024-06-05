import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
from config import *

class EfficientNetB7_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def configure_model(self):
        model = tf.keras.applications.EfficientNetB7(
            input_shape=self.image_shape, 
            include_top=False, 
            weights='imagenet'
        )

        inputs = Input(shape=self.image_shape)
        x = model(inputs)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.5)(x)

        x = Dense(512, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)

        x = Dense(256, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Dropout(0.5)(x)

        outputs = Dense(NUM_CLASS, activation='softmax', kernel_initializer=GlorotUniform())(x)

        return Model(inputs=inputs, outputs=outputs)