# coding: UTF-8

import os
import torch
import types
import logging
from ast import Raise

class Config(types.SimpleNamespace):
    _config = None

    def createDefaultConfig(data_dir, model_type):
        Config._config = Config(data_dir, model_type)
        return Config._config
    
    def getDefaultConfig():
        if Config._config is None:
            Raise("Must call Config.createDefaultConfig first!")
        return Config._config

    def __init__(self, data_dir, model_type):
        logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger("G")
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "train.txt")
        self.eval_file = os.path.join(data_dir, "eval.txt")
        self.test_file = os.path.join(data_dir, "test.txt")
        self.label_file = os.path.join(data_dir, "label.txt")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.eval_file)
        assert os.path.isfile(self.label_file)

        self.saved_model_dir = os.path.join(data_dir, "model")
        self.saved_model = os.path.join(self.saved_model_dir, f"model_for_{model_type}.pth")
        if not os.path.exists(self.saved_model_dir):
            os.mkdir(self.saved_model_dir)

        with open(self.label_file, "r", encoding="UTF-8") as f:
            self.label_list = [label.strip() for label in f.readlines()]
            self.logger.info("classes:" + str(self.label_list))
        
        self.num_labels = len(self.label_list)
        
        self.log_batch = 100
       
        if "data_news" in data_dir:
             # for data_news
            self.num_epochs = 4
            self.batch_size = 128
            self.seq_len = 32
            self.require_improvement = 1000
        elif "data_judge" in data_dir:
            # for data_judge
            self.num_epochs = 1
            self.batch_size = 32
            self.seq_len = 512
            self.require_improvement = 300
        else:
            Raise("Invalid data_dir arguments!")

        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




