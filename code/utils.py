import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from train_config import *
from import_data import Import_data

def save_result(history):
    if MODEL_NAME == 'InceptionResNet':
        loss = history.history['loss']
        accuracy = history.history['accuracy']
        precision = history.history['precision']
        recall = history.history['recall']

        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']
        val_precision = history.history['val_precision']
        val_recall = history.history['val_recall']

        epochs = range(1,len(loss)+1)
        epoch_list = list(epochs)

        df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': accuracy, 'train_loss': loss, 'train_precision': precision, 'train_recall': recall, 'validation_accuracy': val_accuracy, 'validation_loss': val_loss, 'validation_precision': val_precision, 'validation_recall': val_recall},
                                columns=['epoch', 'train_accuracy', 'train_loss', 'train_precision', 'train_recall', 'validation_accuracy', 'validation_loss', 'validation_precision', 'validation_recall'])
        df.to_csv(os.path.join(RESULT_FILE_PATH, 'result.csv'), index=False, encoding='euc-kr')

        plt.plot(epochs, accuracy, 'b', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(RESULT_FILE_PATH, 'accuracy.png'))
        plt.cla()

    elif MODEL_NAME == 'TripletNet':
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1,len(loss)+1)
        epoch_list = list(epochs)

        df = pd.DataFrame({'epoch': epoch_list, 'train_loss': loss, 'validation_loss': val_loss}, columns=['epoch', 'train_loss', 'validation_loss'])
        df.to_csv(os.path.join(RESULT_FILE_PATH, 'result.csv'), index=False, encoding='euc-kr')

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(RESULT_FILE_PATH, 'loss.png'))
    plt.cla()

    K.clear_session()

'''
if InceptionResNet,
    initial train: with the model frozen, train the top layers 
    fine tune after early_stopping: with the the model unfrozen, train the entire model with a lower learning rate 
'''

def create_callbacks(checkpoint_path, checkpoint_file_path, tensorboard_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    check_point = ModelCheckpoint(checkpoint_file_path, verbose=1, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
    tbd_callback = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
    return [early_stop, check_point, tbd_callback]

class FineTune(Callback):
    def __init__(self, new_learning_rate, train_generator, val_generator, epochs):
        super(FineTune, self).__init__()
        self.finetune_flag = True # to ensure that finetune implements once
        self.new_learning_rate = new_learning_rate
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        if self.finetune_flag and self.model.stop_training: 
            print("\nSwitching to fine-tuning phase")
            self.finetune_flag = False

            self.model.layers[1].trainable = True 
            K.set_value(self.model.optimizer.learning_rate, self.new_learning_rate)
            callbacks = create_callbacks(FINETUNE_CHECKPOINT_PATH, FINETUNE_CHECKPOINT_FILE_PATH, FINETUNE_TSBOARD_PATH)

            self.model.fit(
                self.train_generator,
                steps_per_epoch=self.train_generator.samples//self.train_generator.batch_size,
                validation_data=self.val_generator,
                validation_steps=self.val_generator.samples//self.val_generator.batch_size,
                initial_epoch=epoch+1,
                epochs=self.epochs,
                callbacks=callbacks,
                verbose=0
            )
   

def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    
    return tf.reduce_mean(loss)

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