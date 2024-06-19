# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = './dataset/ChiFraud_train.csv'
        self.dev_path = './dataset/ChiFraud_t2022.csv'     
        self.test_path = './dataset/ChiFraud_t2023.csv'       
        self.class_list = [x.strip() for x in open(
            './dataset/class.txt', encoding='utf-8').readlines()]    
        self.vocab_path = './data/vocab.pkl'
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.log_path = './log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('./data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                      
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        self.dropout = 0.5                
        self.require_improvement = 1000                           
        self.num_classes = len(self.class_list)
        self.n_vocab = 0                      
        self.num_epochs = 50                 
        self.batch_size = 1
        self.pad_size = 256
        self.learning_rate = 1e-3 
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300     
        self.filter_sizes = (2, 3, 4)                         
        self.num_filters = 256                           

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
