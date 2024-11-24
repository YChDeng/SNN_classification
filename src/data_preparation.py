# src/data_preparation.py

import os
import shutil
import zipfile
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi

def configure_kaggle_api(kaggle_json_path="./kaggle.json"):
    kaggle_destination_path = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_destination_path, exist_ok=True)
    kaggle_json_dest = os.path.join(kaggle_destination_path, "kaggle.json")

    if not os.path.exists(kaggle_json_dest):
        shutil.copy(kaggle_json_path, kaggle_json_dest)
    os.chmod(kaggle_json_dest, 0o600)
    print("Kaggle API 配置完成。")

def download_and_extract_dataset(dataset_name, output_zip_file, output_dir):
    api = KaggleApi()
    api.authenticate()

    if not os.path.exists(output_zip_file):
        print(f"正在下载数据集：{dataset_name}")
        api.dataset_download_files(dataset_name, path=".", quiet=False)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(output_zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("数据集解压完成。")
    else:
        print(f"数据集已解压到 {output_dir}")

def prepare_fingerprint_dataset(load_dir, finger_dir, map_finger_name):
    finger_train_dir = os.path.join(finger_dir, 'train/')
    finger_val_dir = os.path.join(finger_dir, 'val/')
    finger_test_dir = os.path.join(finger_dir, 'test/')

    if not os.path.exists(finger_dir):
        os.makedirs(finger_dir)
        os.makedirs(finger_train_dir)
        os.makedirs(finger_val_dir)
        os.makedirs(finger_test_dir)

        for i in range(10):
            os.makedirs(os.path.join(finger_train_dir, str(i)))
            os.makedirs(os.path.join(finger_val_dir, str(i)))
            os.makedirs(os.path.join(finger_test_dir, str(i)))

        for dir in ['Real/', 'Altered/Altered-Easy/']:
            current_dir = os.path.join(load_dir, dir)
            if not os.path.exists(current_dir):
                continue
            for img in os.listdir(current_dir):
                ind = max([i if map_finger_name[i] in img.lower() else -1 for i in range(len(map_finger_name))])
                if ind >= 0:
                    shutil.copy2(os.path.join(current_dir, img), os.path.join(finger_train_dir, str(ind), img))
    else:
        print(f"指纹数据集已准备在 {finger_dir}")

def split_fingerprint_dataset(finger_train_dir, finger_val_dir, finger_test_dir, val_split=0.1, test_split=0.1):
    for finger_type in os.listdir(finger_train_dir):
        class_dir = os.path.join(finger_train_dir, finger_type)
        imgs = os.listdir(class_dir)
        np.random.shuffle(imgs)
        total_imgs = len(imgs)
        test_size = int(total_imgs * test_split)
        val_size = int(total_imgs * val_split)

        for img in imgs[:test_size]:
            shutil.move(os.path.join(class_dir, img), os.path.join(finger_test_dir, finger_type, img))
        for img in imgs[test_size:test_size + val_size]:
            shutil.move(os.path.join(class_dir, img), os.path.join(finger_val_dir, finger_type, img))
    print("指纹数据集已划分为训练、验证和测试集。")