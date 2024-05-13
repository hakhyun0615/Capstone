import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from train_config import *

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # subtract the two previous distances and add alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # take the maximum of basic_loss, 0.0 and sum over the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

def save_result(history):
    accuracy = history.history['accuracy']
    loss = history.history['loss']
    precision = history.history['precision']
    recall = history.history['recall']

    val_accuracy = history.history['val_accuracy']
    val_loss = history.history['val_loss']
    val_precision = history.history['val_precision']
    val_recall = history.history['val_recall']

    epochs = range(len(accuracy))
    epoch_list = list(epochs)

    df = pd.DataFrame({'epoch': epoch_list, 'train_accuracy': accuracy, 'train_loss': loss, 'train_precision': precision, 'train_recall': recall, 'validation_accuracy': val_accuracy, 'validation_loss': val_loss, 'validation_precision': val_precision, 'validation_recall': val_recall},
                            columns=['epoch', 'train_accuracy', 'train_loss', 'train_precision', 'train_recall', 'validation_accuracy', 'validation_loss', 'validation_precision', 'validation_recall'])
    df_save_path = os.path.join(os.path.expanduser(RESULT_FILE_PATH), 'result.csv')
    df.to_csv(df_save_path, index=False, encoding='euc-kr')

    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    save_path = os.path.join(os.path.expanduser(RESULT_FILE_PATH), 'accuracy.png')
    plt.savefig(save_path)
    plt.cla()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    save_path = os.path.join(os.path.expanduser(RESULT_FILE_PATH), 'loss.png')
    plt.savefig(save_path)
    plt.cla()

    K.clear_session()