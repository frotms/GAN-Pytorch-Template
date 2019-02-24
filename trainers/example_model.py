# coding=utf-8
import os
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

from trainers.base_model import BaseModel
from nets.net_interface import NetModule
from utils import utils

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        self.create_model()


    def create_model(self):
        # import your net with self.netG and self,netD here
        from net import example_net
        self.netG, self.netD = example_net.BUILD_NET()
        if torch.cuda.is_available():
            self.netG.cuda()
            self.netD.cuda()


    def save(self, net):
        """
        implement the logic of saving model
        """
        print("Saving model...")
        self._save(net, self.config)
        print("Model saved!")


    def _save(self, net, config):
        save_path = config['save_path']
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, config['save_name'])
        state_dict = OrderedDict()
        for item, value in net.state_dict().items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        torch.save(state_dict, save_name)

    def load(self):
        """
        train_mode: 0:from scratch, 1:finetuning, 2:update
        :return:
        """
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False

        # train_mode = self.config['train_mode']
        # if train_mode == 'fromscratch':
        #     if torch.cuda.device_count() > 1:
        #         self.netG = nn.DataParallel(self.netG)
        #         self.netD = nn.DataParallel(self.netD)
        #     if torch.cuda.is_available():
        #         self.netG.cuda()
        #         self.netD.cuda()
        #     print('from scratch...')
        #
        # elif train_mode == 'finetune':
        #     self._load(self.netG, self.config)
        #     self._load(self.netD, self.config)
        #     if torch.cuda.device_count() > 1:
        #         self.netG = nn.DataParallel(self.netG,device_ids=range(torch.cuda.device_count()))
        #         self.netD = nn.DataParallel(self.netD, device_ids=range(torch.cuda.device_count()))
        #     if torch.cuda.is_available():
        #         self.netG.cuda()
        #         self.netD.cuda()
        #     print('finetuning...')
        #
        # elif train_mode == 'update':
        #     self._load(self.netG, self.config)
        #     self._load(self.netD, self.config)
        #     print('updating...')
        #
        # else:
        #     ValueError('train_mode is error...')

        raise NotImplementedError("Please implement your logic of loading weights of pretrained model here")


    def _load(self, net, config):
        """
        loading weights from a model file
        :param net:
        :param config:
        :return:
        """
        _state_dict = torch.load(os.path.join(config['pretrained_path'], config['pretrained_file']),
                                map_location=None)
        # for multi-gpus
        state_dict = OrderedDict()
        for item, value in _state_dict.items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        # for handling in case of different models compared to the saved pretrain-weight
        model_dict = net.state_dict()
        same = {k: v for k, v in state_dict.items() if \
                (k in model_dict and model_dict[k].size() == v.size())}  # or (k not in state_dict)}
        diff = {k: v for k, v in state_dict.items() if \
                (k in model_dict and model_dict[k].size() != v.size()) or (k not in model_dict)}
        # print('diff: ', [i for i, v in diff.items()])
        model_dict.update(same)
        net.load_state_dict(model_dict)

