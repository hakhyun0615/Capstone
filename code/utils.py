import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from config import *
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, accuracy_score
import seaborn as sns
import pickle
from collections import defaultdict

def save_result(history):
    if MODEL_NAME == 'EfficientNetB7':
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

'''
if EfficientNetB7,
    initial train: with the model frozen, train the top layers 
    fine tune after early_stopping: with the the model unfrozen, train the entire model with a lower learning rate 
'''

def create_callbacks(checkpoint_path, checkpoint_file_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')
    check_point = ModelCheckpoint(checkpoint_file_path, verbose=1, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=False)
    return [early_stop, check_point]
   
def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.maximum(basic_loss, 0.0)
    
    return tf.reduce_mean(loss)

def create_embedding_database(train_triplet_generator, model):
    embedding_sum = defaultdict(lambda: np.zeros(128))
    embedding_count = defaultdict(int)
    total_batches = len(train_triplet_generator)
    batch_count = 0

    print("Starting to process batches...")
    for batch, labels in train_triplet_generator:
        if batch_count >= total_batches:
            break 
        batch_count += 1 
        print(f"Processing batch {batch_count}/{total_batches}") 
        embeddings = model.predict(batch, verbose=0) # (batch, 128)
        labels = labels.astype(int) # (batch, )
        for emb, label in zip(embeddings, labels):
            embedding_sum[label] += emb
            embedding_count[label] += 1

    print("Starting to compute mean embeddings...")
    database = {label: embedding_sum[label] / embedding_count[label] for label in embedding_sum}    

    print("Saving database...")
    with open(f"{TEST_RESULT_FILE_PATH}/database.pkl", 'wb') as f:
        pickle.dump(database, f)

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


def evaluate_triplet_model(test_triplet_generator, database, model, output_path):
    y_true = []
    y_pred = []
    total_batches = len(test_triplet_generator)
    batch_count = 0

    print("Starting to evaluate batches...")
    for batch, labels in test_triplet_generator:
        if batch_count >= total_batches:
            break 
        batch_count += 1 
        print(f"Processing batch {batch_count}/{total_batches}") 
        embeddings = model.predict(batch, verbose=0) # (batch, 128)
        labels = labels.astype(int) # (batch, )
        for emb, true_label in zip(embeddings, labels):
            pred_label, _ = predict_closest_embedding(emb, database)
            y_true.append(true_label)
            y_pred.append(pred_label)
    
    print("Saving results...")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    np.save(f"{output_path}/y_pred.npy", y_pred)
    np.save(f"{output_path}/y_true.npy", y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

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