# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Bert'
        self.train_path = './dataset/ChiFraud_train.csv'
        self.dev_path = './dataset/ChiFraud_t2022.csv'     
        self.test_path = './dataset/ChiFraud_t2023.csv'       
        self.class_list = [x.strip() for x in open(
            './dataset/class.txt', encoding='utf-8').readlines()]    
        self.save_path = dataset + './saved_dict/' + self.model_name + '.ckpt'       
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  

        self.require_improvement = 1000                                
        self.num_classes = len(self.class_list)                        
        self.num_epochs = 50                                 
        self.batch_size = 128                                      
        self.pad_size = 256                                      
        self.learning_rate = 5e-5                              
        self.bert_path = 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.vocab_path = './new_vocab.txt'
        self.log_path = dataset + '/log/' + self.model_name


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0] 
        mask = x[2] 
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
