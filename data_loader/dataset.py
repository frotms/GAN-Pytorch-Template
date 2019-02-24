# -*- coding: utf-8 -*-

import os
import cv2
import copy
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_loader.data_processor import DataProcessor


class PyTorchDataset(Dataset):
    def __init__(self, root_dir, config, transform=None, loader = None,
                 target_transform=None,  is_train_set=True):
        self.config = config
        self.transform = transform
        self.target_transform = target_transform
        self.is_train_set = is_train_set

        alist = []
        # implement the logic of getting filenames list with alist
        # with open(txt,'r') as f:
        #     for line in f:
        #         line = line.strip('\n\r').strip('\n').strip('\r')
        #         # single label here so we use int(words[1])
        #         alist.append((words[0], words[1]))

        self.DataProcessor = DataProcessor(self.config)
        self.alist = alist

    def __getitem__(self, index):
        """
        process your image data here
        :param index:
        :return:
        """
        filenameA, filenameB = self.alist[index]

        # get image data by pre-processing with dataA and dataB
        dataA = None # imread or open
        dataB = None  # imread or open

        if self.transform is not None:
            dataA = self.transform(dataA)
            dataB = self.transform(dataB)

        return dataA, dataB


    def __len__(self):
        """
        the length of your all images for training or evaluating
        :return:
        """
        return len(self.alist)


def get_data_loader(config):
    """
    
    :param config: 
    :return: 
    """
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
    shuffle = config['shuffle']

    # configurate your dataset with root_dir param or in another way
    train_data_dir = None
    val_data_dir = None

    train_data = PyTorchDataset(root_dir=train_data_dir, config=config,
                           transform=transforms.ToTensor(), is_train_set=True)
    test_data = PyTorchDataset(root_dir=val_data_dir, config=config,
                                transform=transforms.ToTensor(), is_train_set=False)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader



