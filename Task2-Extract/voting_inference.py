import json
import os.path
from torch.utils.data import Dataset,DataLoader
from json import dumps
from pathlib import Path
import jsonlines
import torch
from transformers import BertTokenizer, BertConfig, BertModel, ErnieConfig, ErnieModel, ElectraConfig, ElectraModel, \
    NezhaModel, NezhaConfig, AutoTokenizer, AutoConfig, AutoModel
import transformers.utils.logging

transformers.utils.logging.set_verbosity_error()
from model import *
from tqdm import tqdm
from argparse import ArgumentParser
from collections import Counter
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import f1_score,precision_score,recall_score
from model import Task2SentencePairModel


MAX_SEQ_LEN = 512

# Task2:争议观点对抽取
class TestDataset(Dataset):

    def __init__(self,data):
        # DataFrame类型
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self,batch_data):
        return batch_data



def build_test_dataloader(text_path,batch_size):
    text_data = pd.read_json(text_path, lines=True)
    text_ids = text_data['text_id'].unique().tolist()
    data = []
    for text_id in text_ids:
        sc_ids = text_data.loc[
            (text_data['position'] == 'sc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text_data.loc[
            (text_data['position'] == 'bc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                data.append({
                    "text_id": text_id,
                    "sc": text_data.loc[
                        (text_data['sentence_id'] == sc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "sc_id": sc_id,
                    "bc": text_data.loc[
                        (text_data['sentence_id'] == bc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "bc_id": bc_id
                })
    test_dataset = TestDataset(data=data)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_dataset.collate_fn,shuffle=False)
    return test_dataloader

def load_model(model_type, model_name, model_path, device):
    if model_type == "bert":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_type == "mrc-roberta":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_type == "ernie":
        model_config = ErnieConfig.from_pretrained(model_name)
        pretrained_model = ErnieModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_type == "nezha":
        model_config = NezhaConfig.from_pretrained(model_name)
        pretrained_model = NezhaModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
    else:
        model_config = AutoConfig.from_pretrained(model_name)
        pretrained_model = AutoModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Task2SentencePairModel(pretrained_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Load model {} successfully!".format(model_name))
    model.eval()
    return model, tokenizer


def get_hard_voting_result(input_list):
    counter = Counter(input_list)
    most_common_element = counter.most_common(1)[0][0]
    return most_common_element


@torch.no_grad()
def batch_soft_voting(device, model_list,batch_size,mode="valid",threshold=0.3):
    Path('output').mkdir(exist_ok=True)
    if mode == "valid":
        text_file = './data/new_valid_text.jsonl'
    else:
        text_file = './data/new_test_text.jsonl'
    out_file: str = os.path.join("output", "soft_voting_" + mode + "_pair.jsonl")

    # 原数据按照text_id分组
    text_data = pd.read_json(text_file, lines=True)
    text_ids = text_data['text_id'].unique().tolist()
    data = []
    for text_id in text_ids:
        sc_ids = text_data.loc[
            (text_data['position'] == 'sc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text_data.loc[
            (text_data['position'] == 'bc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                data.append({
                    "text_id": text_id,
                    "sc": text_data.loc[
                        (text_data['sentence_id'] == sc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "sc_id":sc_id,
                    "bc": text_data.loc[
                        (text_data['sentence_id'] == bc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "bc_id":bc_id
                })
    dataloader = build_test_dataloader(text_path=text_file,batch_size=batch_size)
    trained_models = []
    for m in model_list:
        model, tokenizer = load_model(m["model_type"], m["model_name"], m["model_path"], device)
        trained_models.append((model, tokenizer,))
    # parameters = count_parameters(model)
    # print(parameters)

    outputs = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in bar:
        probs = torch.zeros(len(batch_data)).to(device)
        max_len = max([len(d['sc'] + d['bc']) for d in batch_data]) + 3
        max_len = max_len if max_len <= 512 else 512
        for model,tokenizer in trained_models:
            input_ids = []
            attention_mask = []
            token_type_ids = []
            # +3 sc bc

            for d in batch_data:
                output = tokenizer(d['sc'], text_pair=d['bc'],padding='max_length', truncation=True,
                                            max_length=max_len)
                input_ids.append(output['input_ids'])
                attention_mask.append(output['attention_mask'])
                token_type_ids.append(output['token_type_ids'])
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            probs += torch.sigmoid(logits)
        torch.cuda.empty_cache()
        probs = probs / len(trained_models)
        preds = (probs > threshold).long().cpu().tolist()
        for i,pred in enumerate(preds):
            if pred==1:
                outputs.append({
                    "text_id":batch_data[i]['text_id'],
                    "sc_id":batch_data[i]['sc_id'],
                    "bc_id":batch_data[i]['bc_id']
                })

    # 结果写入提交要求格式文件
    with open(out_file, 'w', encoding='utf8') as f:
        for output in outputs:
            print(dumps({
                'text_id': output["text_id"],
                'sc_id': output['sc_id'],
                'bc_id': output['bc_id']
            }), file=f)
    if mode=="valid":
        valid_f1_score,valid_precision_score,valid_recall_score = cal_f1_score_from_file(text_file='./data/new_valid_text.jsonl',pair_file='./data/new_valid_pair.jsonl',pred_file=out_file)
        print("Valid F1-Score:{}".format(valid_f1_score))
        print("Valid Precision-Score:{}".format(valid_precision_score))
        print("Valid Recall-Score:{}".format(valid_recall_score))


@torch.no_grad()
def soft_voting(device, model_list,mode="valid",threshold=0.3):
    Path('output').mkdir(exist_ok=True)
    if mode == "valid":
        text_file = './data/new_valid_text.jsonl'
    else:
        text_file = './data/new_test_text.jsonl'
    out_file: str = os.path.join("output", "soft_voting_" + mode + "_pair.jsonl")

    # 原数据按照text_id分组
    text_data = pd.read_json(text_file, lines=True)
    text_ids = text_data['text_id'].unique().tolist()
    data = []
    for text_id in text_ids:
        sc_ids = text_data.loc[
            (text_data['position'] == 'sc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text_data.loc[
            (text_data['position'] == 'bc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                data.append({
                    "text_id": text_id,
                    "sc": text_data.loc[
                        (text_data['sentence_id'] == sc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "sc_id":sc_id,
                    "bc": text_data.loc[
                        (text_data['sentence_id'] == bc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "bc_id":bc_id
                })

    trained_models = []
    for m in model_list:
        model, tokenizer = load_model(m["model_type"], m["model_name"], m["model_path"], device)
        trained_models.append((model, tokenizer,))
    # parameters = count_parameters(model)
    # print(parameters)

    outputs = []
    for d in tqdm(data):
        text_id = d['text_id']
        # print(text_id)
        sc = d['sc']
        sc_id = d['sc_id']
        bc = d['bc']
        bc_id = d['bc_id']
        prob = torch.zeros(1).to(device)
        for model,tokenizer in trained_models:
            input = tokenizer(sc,text_pair=bc,max_length=512,truncation=True,return_tensors='pt')
            input_ids = input['input_ids'].to(device)
            attention_mask = input['attention_mask'].to(device)
            token_type_ids = input['token_type_ids'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            prob += torch.sigmoid(logits)
        prob = prob / len(trained_models)
        if prob > threshold:
            outputs.append({
                "text_id":text_id,
                "sc_id":sc_id,
                "bc_id":bc_id
            })

    # 结果写入提交要求格式文件
    with open(out_file, 'w', encoding='utf8') as f:
        for output in outputs:
            print(dumps({
                'text_id': output["text_id"],
                'sc_id': output['sc_id'],
                'bc_id': output['bc_id']
            }), file=f)
    if mode=="valid":
        valid_f1_score,valid_precision_score,valid_recall_score = cal_f1_score_from_file(text_file='./data/new_valid_text.jsonl',pair_file='./data/new_valid_pair.jsonl',pred_file=out_file)
        print("Valid F1-Score:{}".format(valid_f1_score))
        print("Valid Precision-Score:{}".format(valid_precision_score))
        print("Valid Recall-Score:{}".format(valid_recall_score))

def cal_f1_score_from_file(text_file,pair_file,pred_file):
    text = pd.read_json(text_file,lines=True)
    pair = pd.read_json(pair_file,lines=True)
    pred = pd.read_json(pred_file,lines=True)
    true_pairs = []
    true_labels = []
    pred_pairs = []
    pred_labels = []
    for index,row in pair.iterrows():
        text_id = row['text_id']
        sc_id = row['sc_id']
        bc_id = row['bc_id']
        true_pairs.append((text_id,sc_id,bc_id))

    for index,row in pred.iterrows():
        text_id = row['text_id']
        sc_id = row['sc_id']
        bc_id = row['bc_id']
        pred_pairs.append((text_id, sc_id, bc_id))

    text_ids = text['text_id'].unique().tolist()

    for text_id in text_ids:
        sc_ids = text.loc[(text['position'] == 'sc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text.loc[(text['position'] == 'bc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                if (text_id, sc_id, bc_id) in true_pairs:
                    true_labels.append(1)
                else:
                    true_labels.append(0)
                if (text_id, sc_id, bc_id) in pred_pairs:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)
    valid_f1_score = f1_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    valid_precision_score = precision_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    valid_recall_score = recall_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    return valid_f1_score, valid_precision_score, valid_recall_score

if __name__ == '__main__':
    model_list = [
        {"model_type":"ernie","model_name":r"E:\model\ernie-3.0-base-zh","model_path":r"E:\邀请赛2模型\0.55038_ernie-3.0-base-zh.bin"},
                  # {"model_type":"ernie","model_name":r"E:\model\ernie-3.0-xbase-zh","model_path":r"E:\【邀请赛模型】\0.41212_ernie-3.0-xbase-zh.bin"},
                  # {"model_type":"nezha","model_name":r"E:\model\nezha-cn-base","model_path":r"E:\邀请赛2模型\0.53329_nezha-cn-base.bin"},
                  {"model_type":"nezha", "model_name": r"E:\model\nezha-base-wwm","model_path":r"E:\邀请赛2模型\0.53423_nezha-base-wwm.bin"},
                  # {"model_type":"bert", "model_name": r"E:\model\chinese-roberta-wwm-ext","model_path":r"E:\【邀请赛模型】\0.41946_chinese-roberta-wwm-ext.bin"},
                  # {"model_type":"bert", "model_name": r"E:\model\chinese-roberta-wwm-ext-large","model_path":r"E:\【邀请赛模型】\0.40188_chinese-roberta-wwm-ext-large.bin"},
                  # {"model_type":"bert", "model_name": r"E:\model\chinese-macbert-large","model_path":r"E:\【邀请赛模型】\0.40560_chinese-macbert-large.bin"},
                  ]
    device = "cuda:0"

    batch_soft_voting(device, model_list,batch_size=4,mode="valid",threshold=0.3)
    # voting2tensor(device,model_list)