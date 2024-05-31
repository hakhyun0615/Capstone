from model.InceptionResNet import InceptionResNet_model
from model.TripletNet import TripletNet_model

class Load_model:
    def __init__(self, model_name, image_size, pretrained_checkpoint_path=None):
        self.model_name = model_name
        self.image_size = image_size
        self.pretrained_checkpoint_path = pretrained_checkpoint_path

    def build_model(self):
        if self.model_name == 'InceptionResNet':
            model = InceptionResNet_model(self.image_size).configure_model()
        elif self.model_name == 'TripletNet':
            model = TripletNet_model(self.image_size, self.pretrained_checkpoint_path).configure_model()
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")
            
        model.summary()

        return model