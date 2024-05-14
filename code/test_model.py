import tensorflow as tf
import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
from pytz import timezone
from datetime import datetime
from train_config import *
from test_config import *
from utils import *
from load_model import Load_model
from import_data import Import_data, Import_triplet_data
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import sys

class Test_model:
    def __init__(self, test_data_path, model_name, image_size, batch_size):
        self.model_name = model_name
        self.model = Load_model(model_name, image_size)
        if self.model_name == 'TripletNet':
            self.test_triplet_generator = Import_triplet_data(test_data_path, batch_size, image_size)
        else:
            self.test_generator = Import_data(image_size, batch_size, test_data_path=test_data_path).build_generators('test')

    def create_embedding_database(self, train_generator, model):
        database = {}
        for batch, labels in train_generator:
            embeddings = model.predict(batch)
            for emb, label in zip(embeddings, labels):
                label = np.argmax(label)  # Assuming labels are one-hot encoded
                if label not in database:
                    database[label] = []
                database[label].append(emb)
        # Average embeddings for each label
        for label in database:
            database[label] = np.mean(database[label], axis=0)
        return database
    
    def predict_label(self, embedding, database):
        min_dist = float('inf')
        identity = None

        for label, db_emb in database.items():
            dist = np.linalg.norm(embedding - db_emb)
            if dist < min_dist:
                min_dist = dist
                identity = label
        
        return identity, min_dist

    def evaluate_triplet_model(self, test_generator, database, model):
        y_true = []
        y_pred = []

        for batch, labels in test_generator:
            embeddings = model.predict(batch)
            for emb, label in zip(embeddings, labels):
                true_label = np.argmax(label)  # Assuming labels are one-hot encoded
                pred_label, _ = self.predict_label(emb, database)
                y_true.append(true_label)
                y_pred.append(pred_label)
        
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        precision = Precision()(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
        recall = Recall()(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()

        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
        return accuracy, precision, recall

    
    def test(self):
        model = self.model.build_model() 
        if self.model_name == 'TripletNet':
            model.add_loss(triplet_loss(model.outputs[0], model.outputs[1], model.outputs[2]))
            model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=LEARNING_RATE),
                          metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
        
        checkpoint = tf.train.latest_checkpoint(os.path.expanduser(CHECKPOINT_FILE_PATH))
        model.load_weights(checkpoint)

        if self.model_name == 'TripletNet':
            # Create database from training data or known labeled data
            train_triplet_gen = Import_triplet_data(TRAIN_DATA_PATH, BATCH_SIZE, IMAGE_SIZE)
            database = self.create_embedding_database(train_triplet_gen, model)
            # Evaluate the model on the test data
            self.evaluate_triplet_model(self.test_triplet_generator, database, model)
        else:
            eval = model.evaluate(self.test_generator)
            print(f'test_loss : {eval[0]}, test_accuracy : {eval[1]}, test_precision : {eval[2]}, test_recall : {eval[3]}')
            
            y_true = []
            y_pred = []
            preds = model.predict(self.test_generator)
            for i, (image, label) in enumerate(self.test_generator):
                label_len = len(label)
                for q in range(label_len):
                    y_true.append(np.argmax(label[q]))
                    y_pred.append(np.argmax(preds[i * self.test_generator.batch_size + q]))

            conf_mat = confusion_matrix(y_true, y_pred)
            df_conf_mat = pd.DataFrame(conf_mat, columns=[str(i) for i in range(conf_mat.shape[0])],
                                       index=[str(i) for i in range(conf_mat.shape[1])])
            sns_heatmap = sns.heatmap(data=df_conf_mat, annot=True, fmt='d', linewidths=.5, cmap='BuGn_r')
            sns_heatmap.get_figure().savefig(f"{TEST_RESULT_FILE_PATH}/confusion_matrix.png")

            target_names = [str(i) for i in range(conf_mat.shape[0])]
            print(classification_report(y_true, y_pred, digits=5, target_names=target_names))

start = datetime.now(timezone('Asia/Seoul'))
print(f"Test start : {start}")

if __name__ == '__main__':
    if not os.path.exists(TEST_RESULT_FILE_PATH):
        os.makedirs(TEST_RESULT_FILE_PATH)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(f'{RESULT_FILE_PATH}/test_log.txt', 'a', 'utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])
    logger = logging.getLogger()      

    test_model = Test_model(TEST_DATA_PATH, MODEL_NAME, IMAGE_SIZE, BATCH_SIZE)
    test_model.test()

end = datetime.now(timezone('Asia/Seoul'))
print(f"Test end : {end}")