import multiprocessing
import pickle
import numpy as np
import sklearn
from typing import List, Tuple, Any
from transformers import *
import torch.nn as nn
import os
from collections import defaultdict
import itertools
import jsonlines
import torch
import json
import os
import random

context_models = {
    'bert-base-chinese' : {"model": BertModel, "tokenizer" : BertTokenizer },
    'openai-gpt': {"model": OpenAIGPTModel, "tokenizer": OpenAIGPTTokenizer},
    'gpt2': {"model": GPT2Model, "tokenizer": GPT2Tokenizer},
    'ctrl': {"model": CTRLModel, "tokenizer": CTRLTokenizer},
    'transfo-xl-wt103': {"model": TransfoXLModel, "tokenizer": TransfoXLTokenizer},
    'xlnet-base-cased': {"model": XLNetModel, "tokenizer": XLNetTokenizer},
    'xlm-mlm-enfr-1024': {"model": XLMModel, "tokenizer": XLMTokenizer},
    'distilbert-base-cased': {"model": DistilBertModel, "tokenizer": DistilBertTokenizer},
    'roberta-base': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'roberta-large': {"model": RobertaModel, "tokenizer": RobertaTokenizer},
    'xlm-roberta-base': {"model": XLMRobertaModel, "tokenizer": XLMRobertaTokenizer},
}



def set_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





# 对抗训练
class FGM:
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                # print(param.grad)
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# PGD
class PGD:
    def __init__(self, model,emb_name, eps=1, alpha=0.3):
        self.model = model
        self.emd_name = emb_name
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self,is_first_attack=True):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emd_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emd_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

# 文件读写
def save_data2json(newfile_path,data):
    '''
    把数据保存为json文件
    :param newfile_path:
    :param data:
    :return:
    '''
    data = json.dumps(data,ensure_ascii=False)
    with open(newfile_path, 'w', encoding='utf-8') as f:
        f.write(data)


def data2json():
    dir = 'data'
    file = 'multi_train_pair'
    path = os.path.join(dir,file+'.jsonl')
    with jsonlines.open(path) as reader:
        data = [row for row in reader]
    save_data2json(os.path.join('code/LegacyModel/new_data', file + '.json'), data)

def load_json2data(file_path):
    '''
    把数据从json中读取
    :param file_path:
    :return:
    '''
    with open(file_path,mode='r',encoding='utf-8') as f:
        data = json.load(f)
    return data



def json2jsonl(oldfile_path,newfile_path):
    '''
    json文件转换为jsonl文件格式
    :param oldfile_path:
    :param newfile_path:
    :return:
    '''
    data = load_json2data(file_path=oldfile_path)
    with jsonlines.open(newfile_path, "w") as writer:
        for d in data:
            writer.write(d)


def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())