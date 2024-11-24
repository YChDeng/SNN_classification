# src/data_loading.py

import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def get_transforms():
    # 定义指纹数据的转换
    class CropInvertPadTransform:
        def __init__(self, size):
            self.size = size

        def __call__(self, x):
            x = TF.crop(x, 2, 2, TF.get_image_size(x)[1] - 6, TF.get_image_size(x)[0] - 6)
            x = TF.rgb_to_grayscale(x)
            x = TF.invert(x)
            a = max(TF.get_image_size(x)) - TF.get_image_size(x)[0]
            b = max(TF.get_image_size(x)) - TF.get_image_size(x)[1]
            x = TF.pad(x, [a // 2, b // 2, a - a // 2, b - b // 2], fill=0)
            if TF.get_image_size(x)[0] > self.size:
                x = TF.resize(x, [self.size, self.size], antialias=True)
            return x

    fingerprints_transform = transforms.Compose([
        transforms.ToTensor(),
        CropInvertPadTransform(97),
        transforms.Normalize((0,), (1,))
    ])

    mnist_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])

    return fingerprints_transform, mnist_transform

def load_datasets(finger_dir, emnist_dir, fashion_dir, transforms_dict, val_split=0.1):
    datasets_dict = {}

    # 加载指纹数据集
    datasets_dict['fingerprints'] = {}
    datasets_dict['fingerprints']['train'] = datasets.ImageFolder(root=os.path.join(finger_dir, 'train'), transform=transforms_dict['fingerprints'])
    datasets_dict['fingerprints']['val'] = datasets.ImageFolder(root=os.path.join(finger_dir, 'val'), transform=transforms_dict['fingerprints'])
    datasets_dict['fingerprints']['test'] = datasets.ImageFolder(root=os.path.join(finger_dir, 'test'), transform=transforms_dict['fingerprints'])

    # 加载 EMNIST 数据集
    datasets_dict['emnist'] = {}
    emnist_full_train = datasets.EMNIST(root=emnist_dir, split='digits', train=True, download=True, transform=transforms_dict['mnist'])
    train_size = int((1 - val_split) * len(emnist_full_train))
    val_size = len(emnist_full_train) - train_size
    datasets_dict['emnist']['train'], datasets_dict['emnist']['val'] = random_split(emnist_full_train, [train_size, val_size])
    datasets_dict['emnist']['test'] = datasets.EMNIST(root=emnist_dir, split='digits', train=False, download=True, transform=transforms_dict['mnist'])

    # 加载 Fashion-MNIST 数据集
    datasets_dict['fashion_mnist'] = {}
    fashion_full_train = datasets.FashionMNIST(root=fashion_dir, train=True, download=True, transform=transforms_dict['mnist'])
    train_size = int((1 - val_split) * len(fashion_full_train))
    val_size = len(fashion_full_train) - train_size
    datasets_dict['fashion_mnist']['train'], datasets_dict['fashion_mnist']['val'] = random_split(fashion_full_train, [train_size, val_size])
    datasets_dict['fashion_mnist']['test'] = datasets.FashionMNIST(root=fashion_dir, train=False, download=True, transform=transforms_dict['mnist'])

    return datasets_dict

def create_dataloaders(datasets_dict, batch_size=128):
    dataloaders_dict = {}
    for key in datasets_dict:
        dataloaders_dict[key] = {}
        for phase in datasets_dict[key]:
            shuffle = True if phase == 'train' else False
            dataloaders_dict[key][phase] = DataLoader(
                datasets_dict[key][phase], batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return dataloaders_dict