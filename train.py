# coding=utf-8
import argparse
import textwrap
import time
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from configs.config import global_config
from utils.logger import ExampleLogger
from trainers.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from data_loader.dataset import get_data_loader


class ImageClassificationPytorch:
    def __init__(self, config):
        gpu_id = config['gpu_id']
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        self.config = config
        self.init()


    def init(self):
        # create net
        self.model = ExampleModel(self.config)
        # load
        self.model.load()
        # create your data generator
        self.train_loader, self.test_loader = get_data_loader(self.config)
        # create logger
        self.logger = ExampleLogger(self.config)
        # create trainer and path all previous components to it
        self.trainer = ExampleTrainer(self.model, self.train_loader, self.test_loader, self.config, self.logger)


    def run(self):
        # here you train your model
        self.trainer.train()


    def close(self):
        # close
        self.logger.close()


def main():
    imageClassificationPytorch = ImageClassificationPytorch(config=global_config)
    imageClassificationPytorch.run()
    imageClassificationPytorch.close()


if __name__ == '__main__':
    main()
