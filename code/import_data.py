from train_config import *
from PIL import ImageFile
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Import_data:
    def __init__(self, image_size, batch_size, train_data_path=None, val_data_path=None, test_data_path=None):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.image_size = image_size
        self.batch_size = batch_size

    def build_generators(self, which_model):
        # data 전처리
        data_generator = ImageDataGenerator(
            # rescale=1./255,
            # featurewise_std_normalization=True,
            # shear_range=0.2, 
            # zoom_range=0.2,                        
            # channel_shift_range=0.1,
            # rotation_range=20,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # horizontal_flip=True,
            # fill_mode='constant',
            # cval=0
        )
        
        if which_model == 'train':
            # batch_size만큼 train_data 불러오기
            train_generator = data_generator.flow_from_directory(
                self.train_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True
            )
            
            # batch_size만큼 val_data 불러오기
            val_generator = data_generator.flow_from_directory(
                self.val_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            return train_generator, val_generator

        elif which_model == 'test':
            # batch_size만큼 train_data 불러오기
            test_generator = data_generator.flow_from_directory(
                self.test_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True
            )

            return test_generator

        else:
            raise ValueError(f"Unsupported which_model: {which_model}")