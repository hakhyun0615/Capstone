import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class InceptionResNet_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def configure_model(self):
        model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.image_shape, 
            include_top=False, 
            weights='imagenet'
        )
        model.trainable = False # freeze

        inputs = Input(shape=self.image_shape)
        x = model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        outputs = Dense(7, activation='softmax')(x)

        return Model(inputs=inputs, outputs=outputs)