# coding: UTF-8

import torch
import random
from config import Config
from transformers import BertTokenizer

class DataProcessor(object):
    def __init__(self, path, device, tokenizer:BertTokenizer, batch_size, max_seq_len):
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data = self.load(path)
        assert len(self.data[0]) == len(self.data[1])
        self.cur_index = 0

    def load(self, path):
        contents = []
        labels = []
        with open(path, mode="r", encoding="utf-8") as f:
            all_lines = f.readlines()
            random.shuffle(all_lines)
            for line in all_lines:
                line = line.strip()
                if not line or line.find('\t') == -1:   continue
                content, label = line.split("\t")
                contents.append(content)
                labels.append(int(label))
        Config.getDefaultConfig().logger.info(f"{len(contents)} samples loaded from {path}")
        return (contents, labels)

    def __next__(self):
        start_index = self.cur_index * self.batch_size
        if start_index >=len(self.data[0]):
            self.cur_index = 0
            raise StopIteration
        end_index = min(self.cur_index * self.batch_size + self.batch_size, len(self.data[0]))
        self.cur_index += 1
        return self._to_tensor(self.data[0][start_index:end_index], self.data[1][start_index:end_index])

    def _to_tensor(self, batch_x, batch_y):
        inputs = self.tokenizer.batch_encode_plus(batch_x, padding="max_length", max_length=self.max_seq_len, truncation="longest_first", return_tensors="pt")
        inputs = inputs.to(self.device)
        labels = torch.LongTensor(batch_y).to(self.device)
        return (inputs, labels)

    def __iter__(self):
        return self
    
    def __len__(self):
        return (len(self.data[0]) - 1) // self.batch_size + 1
    
    def total_samples(self):
        return len(self.data[0])
