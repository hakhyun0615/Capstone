import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class TripletNet_model:
    def __init__(self, image_size):
        self.image_shape = (image_size,image_size,3)

    def create_base_model(self):
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
        x = BatchNormalization()(x)
        outputs = tf.keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

        return Model(inputs=inputs, outputs=outputs)

    def configure_model(self):
        base_model = self.create_base_model()

        input_anchor = Input(shape=self.image_shape, name='anchor_input')
        input_positive = Input(shape=self.image_shape, name='positive_input')
        input_negative = Input(shape=self.image_shape, name='negative_input')
        
        encoded_anchor = base_model(input_anchor)
        encoded_positive = base_model(input_positive)
        encoded_negative = base_model(input_negative)

        return Model(inputs=[input_anchor, input_positive, input_negative], 
                     outputs=[encoded_anchor, encoded_positive, encoded_negative])