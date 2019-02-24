# coding=utf-8

import math
import os
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

    def forward(self, x):
        res = None
        return res

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

    def forward(self, x):
        res = None
        return res

class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()

    def forward(self, x):
        res = None
        return res

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()

    def forward(self, x):
        res = None
        return res


def BUILD_NET():
    """
    net interface
    :return:
    """
    netG = NetG()
    netD = NetD()
    return netG, netD