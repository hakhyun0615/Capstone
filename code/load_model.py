from model.Inception import Inception_model
from model.InceptionResNet import InceptionResNet_model
from model.FaceNet import FaceNet_model

class Load_model:
    def __init__(self, model_name, image_size):
        self.model_name = model_name
        self.image_size = image_size

    def build_model(self):
        if self.model_name == 'Inception':
            model = Inception_model(self.image_size).configure_model()
        elif self.model_name == 'InceptionResNet':
            model = InceptionResNet_model(self.image_size).configure_model()
        elif self.model_name == 'FaceNet':
            model = FaceNet_model(self.image_size).configure_model()
        else:
            raise ValueError(f"unsupported model name: {self.model_name}")
            
        model.summary()

        return model