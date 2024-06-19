#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@date  : 2024/02/20 10:57
@version: 1.0
@desc  :
"""
import json
import os
from functools import partial
from typing import List
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta

import tokenizers
import torch
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, DataLoader

PAD, CLS = '[PAD]', '[CLS]' 

def build_dataset(config, ues_word):

    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                label, content = lin.split('\t')[0:2]
                if label == 'Label_id':
                    continue
                content = content[:pad_size-2]
                # convert characters to input ids
                tokenizer_output = config.tokenizer.encode(content)
                seq_len = len(tokenizer_output)
                bert_tokens = tokenizer_output.ids
                pinyin_tokens = convert_sentence_to_pinyin_ids(config, content, tokenizer_output)
                if len(bert_tokens) > pad_size:
                    bert_tokens = bert_tokens[0:pad_size-1] + [bert_tokens[-1]]
                    pinyin_tokens = pinyin_tokens[0:pad_size-1] + [pinyin_tokens[-1]]
                    seq_len = pad_size
                else:
                    bert_tokens += ([0] * (pad_size - len(bert_tokens)))
                    pinyin_tokens += ([[0, 0, 0, 0, 0, 0, 0, 0]] * (pad_size - len(pinyin_tokens)))
                assert len(bert_tokens) <= pad_size
                assert len(bert_tokens) == len(pinyin_tokens)

                input_ids = bert_tokens
                pinyin_ids = pinyin_tokens
                mask = []
                contents.append(((input_ids, pinyin_ids), int(label), seq_len, mask))
        return contents
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    vocab = list()
    return vocab, train, dev, test

def convert_sentence_to_pinyin_ids(config:str, sentence: str, tokenizer_output: tokenizers.Encoding) -> List[List[int]]:
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in config.pinyin2tensor:
                pinyin_locs[index] = config.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in config.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = config.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)
        return pinyin_ids

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False 
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x_1 = torch.LongTensor([_[0][0] for _ in datas]).to(self.device)
        x_2 = torch.LongTensor([_[0][1] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return ((x_1, x_2), seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
