import tensorflow as tf
import os
import numpy as np
import pandas as pd
import seaborn as sns
from pytz import timezone
from datetime import datetime
from train_config import *
from test_config import *
from load_model import Load_model
from import_data import Import_data
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import SGD  
from sklearn.metrics import confusion_matrix, classification_report
import sys

class Test_model:
    def __init__(self, test_data_path, model_name, image_size, batch_size):
        self.model = Load_model(model_name, image_size)
        self.data = Import_data(image_size, batch_size, test_data_path=test_data_path)
    
    def test(self):
        test_generator = self.data.build_generators('test')
        optimizer = SGD(learning_rate=LEARNING_RATE, momentum=0.999, nesterov=True) 
        model = self.model.build_model() 
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy', Precision(name='precision'), Recall(name='recall')])
        checkpoint = tf.train.latest_checkpoint(os.path.expanduser(CHECKPOINT_FILE_PATH))
        model.load_weights(checkpoint)

        eval = model.evaluate(test_generator)
        print(f'test_loss : {eval[0]}, test_accuracy : {eval[1]}, test_precision : {eval[2]}, test_recall : {eval[3]}')
        
        y_true = []
        y_pred = []
        preds = model.predict(test_generator)
        for i, (image, label) in enumerate(test_generator):
            label_len = len(label)
            for q in range(label_len):
                y_true.append(np.argmax(label[q]))
                y_pred.append(np.argmax(preds[i * test_generator.batch_size + q]))

        conf_mat = confusion_matrix(y_true, y_pred)
        df_conf_mat = pd.DataFrame(conf_mat, columns=[str(i) for i in range(conf_mat.shape[0])],
                                   index=[str(i) for i in range(conf_mat.shape[1])])
        sns_heatmap = sns.heatmap(data=df_conf_mat, annot=True, fmt='d', linewidths=.5, cmap='BuGn_r')
        sns_heatmap.get_figure().savefig(f"{TEST_RESULT_FILE_PATH}/confusion_matrix.png")

        target_names = [str(i) for i in range(conf_mat.shape[0])]
        print(classification_report(y_true, y_pred, digits=5, target_names=target_names))

if not os.path.exists(TEST_RESULT_FILE_PATH):
    os.makedirs(TEST_RESULT_FILE_PATH)

log_file = open(f'{TEST_RESULT_FILE_PATH}/test_log.txt', 'a', encoding='utf8')
sys.stdout = log_file        

start = datetime.now(timezone('Asia/Seoul'))
print(f"Test start : {start}")

test_model = Test_model(TEST_DATA_PATH, MODEL_NAME, IMAGE_SIZE, BATCH_SIZE)
test_model.test()

end = datetime.now(timezone('Asia/Seoul'))
print(f"Test end : {end}")

log_file.close()
sys.stdout = sys.__stdout__