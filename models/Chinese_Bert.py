# coding: UTF-8
import os
import json
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertConfig
from models.modeling_glycebert import GlyceBertForSequenceClassification
from tokenizers import BertWordPieceTokenizer

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Chinese_Bert'
        self.train_path = './dataset/ChiFraud_train.csv'
        self.dev_path = './dataset/ChiFraud_t2022.csv'     
        self.test_path = './dataset/ChiFraud_t2023.csv'       
        self.class_list = [x.strip() for x in open(
            './dataset/class.txt', encoding='utf-8').readlines()]    
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'       
        self.device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')  

        self.require_improvement = 1000                                
        self.num_classes = len(self.class_list)                        
        self.num_epochs = 20                                         
        self.batch_size = 64                                           
        self.pad_size = 256                                            
        self.learning_rate = 5e-5                                      
        self.bert_path = './bert-base-uncased/'
        self.hidden_size = 768
        self.log_path = './log/' + self.model_name

        vocab_file = os.path.join(self.bert_path, "vocab.txt")
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        # self.labels2id = {value: key for key, value in enumerate(self.get_labels())}
        # load pinyin
        with open(os.path.join(self.bert_path+'config', 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map
        with open(os.path.join(self.bert_path+'config', 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map
        with open(os.path.join(self.bert_path+'config', 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert_dir = config.bert_path
        self.bert_config = BertConfig.from_pretrained(self.bert_dir,
                                                      output_hidden_states=False,
                                                      num_labels=config.num_classes)
        self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_dir,
                                                                        config=self.bert_config)
        
    def forward(self, trains):
        input_ids, pinyin_ids = trains[0][0], trains[0][1]
        attention_mask = (input_ids != 0).long()
        outputs = self.model(input_ids, pinyin_ids, attention_mask=attention_mask)
        y_logits = outputs[0].view(-1, self.config.num_classes)
        return y_logits
