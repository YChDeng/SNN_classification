# main.py

import os
import torch
import torch.nn as nn
import numpy as np
from src.data_preparation import (
    configure_kaggle_api, download_and_extract_dataset,
    prepare_fingerprint_dataset, split_fingerprint_dataset)
from src.data_loading import get_transforms, load_datasets, create_dataloaders
from src.models import SNN, CNN
from src.training import training, testing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备：{device}')

    # --- 数据准备 ---

    # 配置 Kaggle API 并下载指纹数据集
    kaggle_json_path = "./kaggle.json"
    dataset_name = "ruizgara/socofing"
    output_zip_file = "socofing.zip"
    output_dir = "./original_data"
    configure_kaggle_api(kaggle_json_path)
    download_and_extract_dataset(dataset_name, output_zip_file, output_dir)

    # 准备指纹数据集
    load_dir = os.path.join(output_dir, 'SOCOFing/')
    datadir = './data/datasets/'
    finger_dir = os.path.join(datadir, 'fingerprints/')
    map_finger_name = [
        'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_little',
        'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_little']
    prepare_fingerprint_dataset(load_dir, finger_dir, map_finger_name)

    # 划分指纹数据集
    finger_train_dir = os.path.join(finger_dir, 'train/')
    finger_val_dir = os.path.join(finger_dir, 'val/')
    finger_test_dir = os.path.join(finger_dir, 'test/')
    split_fingerprint_dataset(finger_train_dir, finger_val_dir, finger_test_dir)

    # --- 数据加载 ---

    # 获取数据转换
    fingerprints_transform, mnist_transform = get_transforms()
    transforms_dict = {'fingerprints': fingerprints_transform, 'mnist': mnist_transform}

    # 加载数据集
    emnist_dir = os.path.join(datadir, 'emnist/')
    fashion_dir = os.path.join(datadir, 'fashion_mnist/')
    datasets_dict = load_datasets(finger_dir, emnist_dir, fashion_dir, transforms_dict)

    # 创建数据加载器
    dataloaders_dict = create_dataloaders(datasets_dict)

    # --- 模型训练和测试 ---

    datasets_to_process = ['fingerprints', 'emnist', 'fashion_mnist']

    for dataset_name in datasets_to_process:
        print(f'\n处理数据集：{dataset_name}')

        # 获取数据加载器
        train_loader = dataloaders_dict[dataset_name]['train']
        val_loader = dataloaders_dict[dataset_name]['val']
        test_loader = dataloaders_dict[dataset_name]['test']

        # 获取输入形状和类别数量
        input_shape = next(iter(train_loader))[0].shape
        if dataset_name == 'fingerprints':
            num_inputs = np.prod(input_shape[1:])
            num_outputs = len(train_loader.dataset.classes)
        else:
            num_inputs = np.prod(input_shape[1:])
            num_outputs = len(train_loader.dataset.dataset.classes)

        # 定义模型
        snn_model = SNN(num_inputs, num_outputs).to(device)
        cnn_model = CNN(input_shape, num_outputs).to(device)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        snn_optimizer = torch.optim.Adam(snn_model.parameters(), lr=0.000085)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.000085)

        # 训练 SNN 模型
        print(f'训练 SNN 模型：{dataset_name}')
        snn_loss, snn_accuracy, snn_time = training(
            snn_model, train_loader, val_loader,
            snn_optimizer, criterion, device, epochs=20, snn_mode=True,
            num_steps=25, path_load_model=f'{dataset_name}_snn_model.pt')

        # 训练 CNN 模型
        print(f'训练 CNN 模型：{dataset_name}')
        cnn_loss, cnn_accuracy, cnn_time = training(
            cnn_model, train_loader, val_loader,
            cnn_optimizer, criterion, device, epochs=20, snn_mode=False,
            path_load_model=f'{dataset_name}_cnn_model.pt')

        # 加载最佳模型
        snn_model.load_state_dict(torch.load(f'{dataset_name}_snn_model.pt'))
        cnn_model.load_state_dict(torch.load(f'{dataset_name}_cnn_model.pt'))

        # 测试 SNN 模型
        print(f'测试 SNN 模型：{dataset_name}')
        true_labels, predictions = testing(
            snn_model, test_loader, device, snn_mode=True, num_steps=25)
        print("SNN 分类报告：\n", classification_report(true_labels, predictions))

        # 测试 CNN 模型
        print(f'测试 CNN 模型：{dataset_name}')
        true_labels, predictions = testing(
            cnn_model, test_loader, device, snn_mode=False)
        print("CNN 分类报告：\n", classification_report(true_labels, predictions))

        # 可视化训练过程
        plt.figure(figsize=(15, 5))

        # 验证损失
        plt.subplot(1, 2, 1)
        plt.plot(snn_loss, label='SNN')
        plt.plot(cnn_loss, label='CNN')
        plt.title(f'{dataset_name} 验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 验证准确率
        plt.subplot(1, 2, 2)
        plt.plot(snn_accuracy, label='SNN')
        plt.plot(cnn_accuracy, label='CNN')
        plt.title(f'{dataset_name} 验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main()