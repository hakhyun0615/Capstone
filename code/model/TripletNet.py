import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model

class TripletNet_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def configure_model(self):
        model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.image_shape, 
            include_top=False, 
            weights='imagenet'
        )
        model.trainable = False
        
        inputs = Input(shape=self.image_shape)
        x = tf.keras.applications.inception_resnet_v2.preprocess_input(inputs)
        x = model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        outputs = BatchNormalization()(x)

        return Model(inputs=inputs, outputs=outputs)