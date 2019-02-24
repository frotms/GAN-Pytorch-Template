#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

class TagPytorchInference(object):

    def __init__(self, **kwargs):
        _input_size = 320
        self.input_size = (_input_size, _input_size)
        self.gpu_index = kwargs.get('gpu_index', '0')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index

        # build net
        self.net = self._create_model(**kwargs)
        # load weights from model
        self._load(**kwargs)
        self.net.eval()
        self.transforms = transforms.ToTensor()
        if torch.cuda.is_available():
            self.net.cuda()

    def close(self):
        torch.cuda.empty_cache()


    def _create_model(self, **kwargs):
        """
        build net
        :param kwargs:
        :return:
        """
        # build net
        net = None
        return net


    def _load(self, **kwargs):
        """
        load weights
        :param kwargs:
        :return:
        """
        model_filename = None
        state_dict = torch.load(model_filename, map_location=None)
        self.net.load_state_dict(state_dict)


    def run(self, image_data, **kwargs):
        _image_data = self.image_preprocess(image_data)
        input = self.transforms(_image_data)
        _size = input.size()
        input = input.resize_(1, _size[0], _size[1], _size[2])
        if torch.cuda.is_available():
            input = input.cuda()
        out = self.net(Variable(input))

        return out.data.cpu().numpy().tolist()


    def image_preprocess(self, image_data):
        _image = cv2.resize(image_data, self.input_size)
        _image = _image[:,:,::-1]   # bgr2rgb
        return _image.copy()

if __name__ == "__main__":

    tagInfer = TagPytorchInference(module_name=module_name,net_name=net_name,
                                   num_classes=num_classes, model_name=model_name,
                                   input_size=input_size)

    result = tagInfer.run(image)

    # post-processing with result
    pass

    print('done!')