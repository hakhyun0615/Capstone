import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class EfficientNet_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def configure_model(self):
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=self.image_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False  # freeze

        inputs = Input(shape=self.image_shape)
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(7, activation='softmax')(x)

        return Model(inputs=inputs, outputs=outputs)