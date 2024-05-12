from time import sleep
import tensorflow as tf
import os
import pandas as pd
import math
from IPython.display import clear_output
from preprocess.print_log import *

# 이미지 및 TFRecord 상위 경로
CONST_ROOT_PATH = os.path.expanduser('~/Desktop/Git/capstone/Capstone')

# 이미지 경로 지정
CONST_IMG_PATH = f'{CONST_ROOT_PATH}/data'

# TFRecord 저장할 폴더 경로 지정
CONST_WORK_PATH = f'{CONST_ROOT_PATH}/tfrecord'

# True or False
SHUFFLE = True

CONST_CLASS = ["1", "2", "3", "4", "5", "6", "7"]

LABEL_NAME = "_".join(CONST_CLASS)

FOLDER_TFRC = f'tfrecord_{LABEL_NAME}'
FOLDER_RESULT = 'result'
FOLDER_LOG = "logs"

CONST_TFRC_PATH = os.path.join(CONST_WORK_PATH, FOLDER_TFRC)

EXTENSION_IMAGE = '.jpg'

EXTENSION_TFRECORDS = '.tfrecords'

CONST_CSV_TFRCLIST = "tfrecord_list.csv"
CONST_CSV_SPLITTED = "splited_df.csv"

CONST_TFRECORD = "tfrecord"
CONST_RESULT_LOSS = "result_loss.jpg"

NUM_IMAGE = 1000

set_debug(False)
set_log_path(os.path.join(CONST_WORK_PATH, FOLDER_LOG), FOLDER_TFRC)

def logout(msg, stdout=True, force_flush=False):
    print_log(level='i', msg=msg, tag="make_tfrecord", on_screen_display=(stdout or is_debug()), force_flush=force_flush)

def make_input_list(start_path: str):
    image_list = []
    image_label_list = []
    except_list = []

    for (path, dirs, files) in os.walk(start_path):
        logout(f"search path: {path}")
        if any(cls in path for cls in CONST_CLASS):
            if len(files) == 0:
                logout("skip current folder")
                continue
            logout(f"{len(dirs)} dirs and {len(files)} files found")
            folder_label = path.split(os.path.sep)[-1]
            for file in files:
                if os.path.splitext(file)[-1].lower() == EXTENSION_IMAGE.lower():
                    image_label_list.append(folder_label)
                    image_list.append(os.path.join(path, file))
                else:
                    except_list.append(file)

    logout(f"# of images = {len(image_label_list)}")
    return image_label_list, image_list, except_list

def make_df(img_path):
    label, image, exp = make_input_list(img_path)
    label_df = pd.DataFrame(label, columns=['folder_name'])
    image_df = pd.DataFrame(image, columns=['image_name'])
    image_df = pd.concat((image_df, label_df), axis=1)
    image_df['image_id'] = image_df['image_name'].apply(lambda x: int(os.path.basename(x).split('.')[0]))
    image_df['label'] = image_df['folder_name']
    image_df.reset_index(drop=True, inplace=True)
    return image_df

def split_data(df):
    group = list(df['folder_name'].unique())
    ds = pd.DataFrame({'image_id':[0], 'label': [0], 'set': ['non']})
    for symp in group:
        group_df = df[df['folder_name']==symp].reset_index(drop=True, inplace=False)
        
        if SHUFFLE == True:
            group_df = group_df.sample(frac=1).reset_index(drop=True)
        elif SHUFFLE == False:
            pass

        symptomatic = group_df[['image_id', 'label']].reset_index(drop=True,inplace=False)
        
        symptomatic['set']='train'
        val_index = (len(symptomatic) * 0.8) - 0.000001
        test_index = (len(symptomatic) * 0.9) - 0.000001
        symptomatic.loc[val_index:test_index, 'set'] = 'val'
        symptomatic.loc[test_index:, 'set'] = 'test'
        ds = pd.concat((ds, symptomatic), ignore_index=True)

    df = pd.merge(df, ds, how='left', on=['image_id', 'label'])
    return df

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def image_example(image_id, image_string, label):
    image_shape = tf.io.decode_jpeg(image_string).shape
    image_id = int(image_id)
    label = int(label)

    feature = {
        'id': _int64_feature(image_id),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def tfrecord_write(df, img_path, tfrc_path):
    folder_list = list(df['folder_name'].unique())
    set_list = list(df['set'].unique())

    # class (폴더명) 분리
    for folder in folder_list:
        clear_output(wait=True)
        folder_df = df[df['folder_name']==folder]
        logout(f"start to make tfrecord...class: {folder}")
        
        # train / validation / test dataset 분리
        for set in set_list:
            set_df = folder_df[folder_df['set']==set]
            label_list = list(set_df['label'].unique())

            for label in label_list:
                label_df = set_df[set_df['label']==label]
                label_df.reset_index(drop=True, inplace=True)

                total_image = len(label_df)
                num_file = math.ceil(total_image/NUM_IMAGE)

                logout(f"start to make tfrecord for {set} (class: {folder})")

                # group(class, train/validation/test)의 tfrecord 분리
                for i in range(num_file):
                    start_index = i*NUM_IMAGE
                    end_index = (i+1)*NUM_IMAGE
                    file_list = label_df[start_index:end_index]
                    file_list.reset_index(drop=True, inplace=True)

                    folder_name = file_list['folder_name'][i]
                    record_file = CONST_TFRECORD + f"_label_{folder_name}_{set}_set_" + str(i+1) + f"_of_{num_file}" + EXTENSION_TFRECORDS
                    logout(f"process {i+1}/{num_file}th tfrecord for {NUM_IMAGE} images [{start_index}:{end_index}] from {folder_name} with label {label}")

                    with tf.io.TFRecordWriter(os.path.join(tfrc_path, record_file)) as writer:
                        for j in range(len(file_list)):
                            image_name = file_list['image_name'][j]
                            label_id = file_list['label'][j]
                            image_id = file_list['image_id'][j]
                            
                            image_file_path = os.path.join(img_path, folder_name, image_name)
                            image_string = open(image_file_path, 'rb').read()    
                            
                            tf_example = image_example(image_id, image_string, label_id)
                            writer.write(tf_example.SerializeToString())
                        sleep(1.)

            logout(f"end to make {num_file} {set} data tfrecord...class: {label}")
        logout(f"end to make tfrecord...class: {label}")

def tfrecord_write_group(df, img_path, tfrc_path):
    set_list = list(df['set'].unique())
    
    for set in set_list:
        set_df = df[df['set'] == set]
        num_file = math.ceil(len(set_df) / NUM_IMAGE)

        for i in range(num_file):
            start_index = i * NUM_IMAGE
            end_index = (i + 1) * NUM_IMAGE
            file_list = set_df[start_index:end_index]
            file_list.reset_index(drop=True, inplace=True)
            
            tfr_path = os.path.join(tfrc_path, set)
            if not os.path.exists(tfr_path):
                os.makedirs(tfr_path)
            
            record_file = f"{set}_{str(i + 1)}_{LABEL_NAME}" + EXTENSION_TFRECORDS
            logout(f"process {i + 1}/{num_file}th tfrecord for {NUM_IMAGE} images [{start_index}:{end_index}]")

            with tf.io.TFRecordWriter(os.path.join(tfr_path, record_file)) as writer:
                for j in range(len(file_list)):
                    image_name = file_list['image_name'][j]
                    label_id = file_list['label'][j]
                    image_id = file_list['image_id'][j]
                    folder_name = file_list['folder_name'][j]
                    
                    image_file_path = os.path.join(img_path, folder_name, image_name)
                    image_string = open(image_file_path, 'rb').read()    
                    
                    tf_example = image_example(image_id, image_string, label_id)
                    writer.write(tf_example.SerializeToString())
                sleep(1.)
        logout(f"end to make {num_file} {set} data tfrecord")
    logout(f"end to make tfrecord")


if __name__ == '__main__':
    logout("start progress...")
    if not os.path.exists(CONST_TFRC_PATH):
        os.makedirs(CONST_TFRC_PATH)
        logout(f"output folder {CONST_TFRC_PATH} created")
    logout(f"created TFRecord files will be stored in {CONST_TFRC_PATH}")

    # 이미지 데이터 프레임 생성
    meta_symptomatic = make_df(CONST_IMG_PATH)
    
    # 데이터 분할
    meta_symptomatic = split_data(meta_symptomatic)

    # CSV로 데이터 저장
    meta_symptomatic.to_csv(os.path.join(CONST_TFRC_PATH, CONST_CSV_TFRCLIST), index=False, encoding='utf-8-sig')
    logout("writing tfrecord files being started")
    
    # TFRecord 파일 작성 시작
    tfrecord_write_group(meta_symptomatic, CONST_IMG_PATH, CONST_TFRC_PATH)
    
    logout("done.", force_flush=True)
