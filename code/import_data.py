import numpy as np
from train_config import *
from PIL import ImageFile, Image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input


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
            preprocessing_function=preprocess_input,
            # horizontal_flip=True,
            # vertical_flip=True,
            # rotation_range=40,
            # shear_range=0.2,
            # brightness_range=[0.8, 1.2],
            # fill_mode='reflect'
        )
        
        if which_model == 'train':
            # make train_data into batches # len(train_generator): number of batches
            train_generator = data_generator.flow_from_directory(
                self.train_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical', # one-hot encode labels
                shuffle=True
            )
            
            # make val_data into batches
            val_generator = data_generator.flow_from_directory(
                self.val_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
            )
            
            return train_generator, val_generator

        elif which_model == 'test':
            # make test_data into batches
            test_generator = data_generator.flow_from_directory(
                self.test_data_path,
                target_size=(self.image_size, self.image_size),
                batch_size=self.batch_size,
                class_mode='categorical',
            )

            return test_generator

        else:
            raise ValueError(f"Unsupported which_model: {which_model}")
        
class Import_triplet_data(Sequence):
    def __init__(self, data_dir, batch_size, image_size):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_folders = sorted(os.listdir(data_dir))
        self.class_indices = {class_name: idx for idx, class_name in enumerate(self.class_folders)}
        self.image_paths, self.labels = self.get_image_paths_and_labels(data_dir)
        self.indices = np.arange(1,len(self.labels)+1)
        self.on_epoch_end()

    def get_image_paths_and_labels(self, data_dir):
        image_paths = []
        labels = []
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            for fname in os.listdir(class_dir):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(self.class_indices[class_name])
        return image_paths, labels # every paths and labels of images in dir

    def __getitem__(self, idx):
        batch = []
        for _ in range(self.batch_size):
            anchor, positive, negative = self.sample_triplet()
            batch.append([anchor, positive, negative])
        batch = np.array(batch) # (batch_size, 3, image_size, image_size, channels)

        return [batch[:, 0], batch[:, 1], batch[:, 2]], np.zeros((self.batch_size, 1)) # batch[:, 0]: (batch_size, image_size, image_size, channels) # np.zeros((self.batch_size, 1): not used

    def __len__(self):
        return int(np.ceil(len(self.labels)/float(self.batch_size))) # number of batches
    
    def sample_triplet(self):
        anchor_class = np.random.choice(list(self.class_indices.keys()))
        negative_class = np.random.choice(list(set(self.class_indices.keys()) - {anchor_class}))
        
        anchor_idx, positive_idx = np.random.choice(
            [i for i, label in enumerate(self.labels) if label == self.class_indices[anchor_class]], 2, replace=False)
        negative_idx = np.random.choice(
            [i for i, label in enumerate(self.labels) if label == self.class_indices[negative_class]], 1)[0]
        
        anchor = self.load_image(self.image_paths[anchor_idx])
        positive = self.load_image(self.image_paths[positive_idx])
        negative = self.load_image(self.image_paths[negative_idx])
        
        return anchor, positive, negative

    def load_image(self, path):
        img = Image.open(path).resize((self.image_size, self.image_size))
        img = np.array(img)
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        img = preprocess_input(img.astype('float32'))
        return img
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)