import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class Inception_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def configure_model(self):
        model = tf.keras.applications.InceptionV3(
            input_shape=self.image_shape, 
            include_top=False, 
            weights='imagenet'
        )
        model.trainable = False
        
        inputs = Input(shape=self.image_shape)
        x = tf.keras.applications.inception_v3.preprocess_input(inputs)
        x = model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(7, activation='softmax')(x)

        return Model(inputs=inputs, outputs=outputs)