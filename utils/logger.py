# coding=utf-8
from __future__ import print_function
import os, sys
import numpy as np
import logging
from utils import utils

class ExampleLogger:
    """
    self.log_writer.info("log")
    self.log_writer.warning("warning log)
    self.log_writer.error("error log ")

    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.flush()
    """
    def __init__(self, config):
        self.config = config
        self.log_writer = self.init()
        self.log_printer = DefinedPrinter()
        self.log_info = {}


    def init(self):
        """
        initial
        :return: 
        """
        log_writer = logging.getLogger(__name__)
        log_writer.setLevel(logging.INFO)
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        handler = logging.FileHandler(os.path.join(self.log_dir, 'alg_training.log'),encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        logging_format = logging.Formatter("@%(asctime)s [%(filename)s -- %(funcName)s]%(lineno)s : %(levelname)s - %(message)s", datefmt='%Y-%m-%d %A %H:%M:%S')
        handler.setFormatter(logging_format)
        log_writer.addHandler(handler)
        return log_writer


    def write_info_to_logger(self, variable_dict):
        """
        print
        :param variable_dict: 
        :return: 
        """
        if variable_dict is not None:
            for tag, value in variable_dict.items():
                self.log_info[tag] = value


    def write(self):
        """
        log writing
        :return: 
        """
        # _info with self.log_info
        # self.log_writer.info(_info)
        # sys.stdout.flush()
        raise NotImplementedError("Please implement the logic of writer here")


    def write_warning(self, warning_dict):
        """
        warninginfo writing
        :return: 
        """
        # _info with self.log_info
        # self.log_writer.warning(_info)
        # sys.stdout.flush()
        raise NotImplementedError("Please implement the logic of warning writer here")

    def clear(self):
        """
        clear log_info
        :return: 
        """
        self.log_info = {}


    def close(self):
        pass



class DefinedPrinter:
    """
    Printer
    """
    def iter_case_print(self, **kwargs):
        """
        print per batch
        :param kwargs:
        :return:
        """
        # print with ** kwarge
        # print(log)
        raise NotImplementedError("Please implement the logic of iter_case_print here")

    def epoch_case_print(self, **kwargs):
        """
        print per epoch
        :param kwargs:
        :return:
        """
        # print with **kwarge
        # print('\n\r', log, '\n\r')
        raise NotImplementedError("Please implement the logic of epoch_case_print here")
