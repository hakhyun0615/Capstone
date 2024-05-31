import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, BatchNormalization, Lambda
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import GlorotUniform

class TripletNet_model:
    def __init__(self, image_size, pretrained_checkpoint_path):
        self.image_shape = (image_size,image_size,3)
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

    def create_base_model(self):
        model = tf.keras.applications.InceptionResNetV2(
            input_shape=self.image_shape, 
            include_top=False, 
            weights=None
        )
        model.load_weights(self.pretrained_checkpoint_path, by_name=True, skip_mismatch=True)
        model.trainable = False
        
        inputs = Input(shape=self.image_shape)
        x = model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu', kernel_initializer=GlorotUniform())(x)
        x = BatchNormalization()(x)
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

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