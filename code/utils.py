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
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

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

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
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


def create_embedding_database(train_generator, model):
    database = {}

    for batch, labels in train_generator:
        anchor, positive, negative = batch
        embeddings = model.predict([anchor, positive, negative])

        anchor_embeddings = embeddings[0]
        positive_embeddings = embeddings[1]

        all_embeddings = np.concatenate([anchor_embeddings, positive_embeddings], axis=0)
        all_labels = np.concatenate([labels, labels], axis=0)

        for emb, label in zip(all_embeddings, all_labels):
            label = int(label)
            if label not in database:
                database[label] = []
            database[label].append(emb)

    for label in database:
        database[label] = np.mean(database[label], axis=0)

    return database

def predict_closest_embedding(embedding, database):
    min_dist = float('inf')
    identity = None

    for label, db_emb in database.items():
        dist = np.linalg.norm(embedding - db_emb)
        if dist < min_dist:
            min_dist = dist
            identity = label
    
    return identity, min_dist

def evaluate_triplet_model(test_generator, database, model, output_path):
    y_true = []
    y_pred = []

    for batch, labels in test_generator:
        anchor, positive, negative = batch
        embeddings = model.predict([anchor, positive, negative])

        anchor_embeddings = embeddings[0]

        for emb, label in zip(anchor_embeddings, labels):
            true_label = int(label)
            pred_label, _ = predict_closest_embedding(emb, database)
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    precision = Precision()(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()
    recall = Recall()(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred)).numpy()

    conf_mat = confusion_matrix(y_true, y_pred)
    df_conf_mat = pd.DataFrame(conf_mat, columns=[str(i) for i in range(conf_mat.shape[0])],
                               index=[str(i) for i in range(conf_mat.shape[1])])
    sns_heatmap = sns.heatmap(data=df_conf_mat, annot=True, fmt='d', linewidths=.5, cmap='BuGn_r')
    sns_heatmap.get_figure().savefig(f"{output_path}/confusion_matrix.png")

    target_names = [str(i) for i in range(conf_mat.shape[0])]
    report = classification_report(y_true, y_pred, digits=5, target_names=target_names)

    with open(f"{output_path}/result.txt", "w") as file:
        file.write(f"test_accuracy: {accuracy}, test_precision: {precision}, test_recall: {recall}\n")
        file.write(report)
    print(f"test_accuracy: {accuracy}, test_precision: {precision}, test_recall: {recall}")
    print(report)