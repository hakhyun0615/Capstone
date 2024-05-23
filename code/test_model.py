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

class Test_model:
    def __init__(self, test_data_path, model_name, image_size, batch_size, epochs, learning_rate, weight_path=None):
        self.model_name = model_name
        if self.model_name == 'TripletNet':
            self.model = Load_model(model_name, image_size, weight_path)
            self.test_triplet_generator = Import_triplet_data(test_data_path, batch_size, image_size)
        else:
            self.model = Load_model(model_name, image_size)
            self.test_generator = Import_data(image_size, batch_size, test_data_path=test_data_path).build_generators('test')
        self.epochs = epochs
        self.learning_rate = learning_rate

    def test(self):
        model = self.model.build_model() 
        if self.model_name == 'TripletNet':
            model.add_loss(triplet_loss(model.outputs[0], model.outputs[1], model.outputs[2]))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        else:
            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(learning_rate=self.learning_rate),
                          metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
        
        checkpoint = os.path.join(FINETUNE_CHECKPOINT_PATH, os.listdir(FINETUNE_CHECKPOINT_PATH)[-1])
        if checkpoint:
            print(f"Checkpoint found: {checkpoint}")
            model.load_weights(checkpoint)
        else:
            print("No checkpoint found")

        if self.model_name == 'TripletNet':
            # Create database from training data or known labeled data
            train_triplet_gen = Import_triplet_data(TRAIN_DATA_PATH, BATCH_SIZE, IMAGE_SIZE)
            database = self.create_embedding_database(train_triplet_gen, model)
            # Evaluate the model on the test data
            self.evaluate_triplet_model(self.test_triplet_generator, database, model)
        else:
            eval = model.evaluate(self.test_generator) # [test_loss, test_accuracy, test_precision, test_recall]
            with open(f"{TEST_RESULT_FILE_PATH}/evaluation.txt", "w") as file:
                file.write(f"test_loss: {eval[0]}, test_accuracy: {eval[1]}, test_precision: {eval[2]}, test_recall: {eval[3]}\n")
            print(f'test_loss : {eval[0]}, test_accuracy : {eval[1]}, test_precision : {eval[2]}, test_recall : {eval[3]}')

            y_pred = np.argmax(model.predict(self.test_generator), axis=-1) + 1 
            y_true = self.test_generator.labels + 1
            np.save(f"{TEST_RESULT_FILE_PATH}/y_pred.npy", y_pred)
            np.save(f"{TEST_RESULT_FILE_PATH}/y_true.npy", y_true)

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
    test_model = Test_model(TEST_DATA_PATH, MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE)
    test_model.test()

end = datetime.now(timezone('Asia/Seoul'))
print(f"Test end : {end}")