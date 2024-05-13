import os
import sys

'''
python clean_data.py 'directory'
'''

def clean_data(directory, limit=15000):
    jpg_files = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith('.jpg'):
                jpg_files.append(file_path)
            else:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    
    jpg_files = sorted(jpg_files)
    if len(jpg_files) > limit:
        for file_path in jpg_files[limit:]:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        jpg_files = jpg_files[:limit]

    for i, file_path in enumerate(sorted(jpg_files), start=1):
        new_file_name = f"{i}.jpg"
        new_file_path = os.path.join(directory, new_file_name)
        os.rename(file_path, new_file_path)
        print(f"Renamed: {file_path} to {new_file_path}")

if __name__ == '__main__':
    directory = sys.argv[1]
    clean_data(directory)
