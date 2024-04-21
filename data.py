"""
Get data
"""

import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets


def build_mnist(args, cutout=False, use_mnist=True, download=True):
    aug = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
])
    train_dataset = datasets.MNIST(root=args.data_path,
                            train=True, download=download, transform=transform_train)
    val_dataset = datasets.MNIST(root=args.data_path,
                            train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset
