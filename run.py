# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, predict
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Detection')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, FastText, TextRCNN, Transformer, Bert, Chinese_Bert')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--mode', default='train', type=str, help='train or test')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'ChiFraud'
    model_name = args.model
    embedding = args.embedding
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'Bert':
        from utils_bert import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'Chinese_Bert':
        from utils_chinesebert import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer' and model_name != 'Bert' and model_name != 'Chinese_Bert':
        init_network(model)
    print(model.parameters)
    if args.mode == 'train':
        train(config, model, train_iter, dev_iter, test_iter)
    elif args.mode == 'test':
        model.load_state_dict(torch.load(config.save_path))
        predict(config, model, test_iter)
