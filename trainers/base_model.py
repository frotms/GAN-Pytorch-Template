# coding=utf-8

import math
import os
from collections import OrderedDict
import numpy as np
import torch

class BaseModel:
    def __init__(self,config):
        self.config = config

    # save function thet save the checkpoint in the path defined in configfile
    def save(self):
        """
        implement the logic of saving model
        """
        raise NotImplementedError

    # load lateset checkpoint from the experiment path defined in config_file
    def load(self):
        """
        implement the logic of loading model
        """
        raise NotImplementedError


    def build_model(self):
        """
        implement the logic of model
        """
        raise NotImplementedError