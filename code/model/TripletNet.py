import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

class TripletNet_model:
    def __init__(self, image_size, pretrained_checkpoint_path):
        self.image_shape = (image_size,image_size,3)
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

    def create_model(self):
        base_model = load_model(self.pretrained_checkpoint_path)
        base_model_output = base_model.layers[-3].output
        outputs = Lambda(lambda x: K.l2_normalize(x, axis=1))(base_model_output)

        return  Model(inputs=base_model.input, outputs=outputs)

    def configure_model(self):
        model = self.create_model()

        inputs = [
            Input(shape=self.image_shape, name='anchor_input'),
            Input(shape=self.image_shape, name='positive_input'),
            Input(shape=self.image_shape, name='negative_input')
        ]

        model_input = tf.concat(inputs, axis=0)
        moddel_output = model(model_input)
        
        outputs = tf.split(moddel_output, 3, axis=0)

        return Model(inputs=inputs, outputs=outputs)