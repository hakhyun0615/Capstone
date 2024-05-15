import os
import sys
import shutil

'''
python split_data.py 'directory'
'''

# original_data/cropped_data
directory = sys.argv[1]

# 원본 데이터 폴더
data_path = f'../../{directory}/'
# 클래스 목록
classes = ['1','2','3','4','5','6','7']
# 데이터 분할 비율
splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}  

# 새로운 폴더 구조 생성
for split in ['train_data', 'val_data', 'test_data']:
    sub_data_path = data_path + split # original data/train_data
    os.makedirs(sub_data_path, exist_ok=True)
    for cls in classes:
        os.makedirs(os.path.join(sub_data_path, cls), exist_ok=True) # original data/train_data/1

# 각 클래스별로 파일을 나누고 이동
for cls in classes:
    # 현재 클래스의 파일 리스트 생성
    files = sorted(os.listdir(os.path.join(data_path, cls)))

    # 분할 지점 계산
    split_train = int(len(files) * splits['train'])
    split_val = split_train + int(len(files) * splits['val'])

      # 파일 이동
    for i, file in enumerate(files):
        if i < split_train: 
            dst_folder = data_path + 'train_data' # original data/train_data
        elif i < split_val:
            dst_folder = data_path + 'val_data'
        else:
            dst_folder = data_path + 'test_data'

        # 원본 파일 경로
        src_path = os.path.join(data_path, cls, file) # original data/1/1.jpg
        # 목표 파일 경로
        dst_path = os.path.join(dst_folder, cls, file) # original data/train_data/1/1.jpg
        # 파일 복사
        shutil.copy(src_path, dst_path)

print("Data split successful.")