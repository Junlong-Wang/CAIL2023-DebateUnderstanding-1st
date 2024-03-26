import json
import os.path
from json import dumps
from pathlib import Path
import jsonlines
import torch
from transformers import BertTokenizer,BertConfig,BertModel,ErnieConfig,ErnieModel,ElectraConfig,ElectraModel,AutoTokenizer,AutoConfig,AutoModel,NezhaModel, NezhaConfig
import transformers.utils.logging
transformers.logging.set_verbosity_error()
from model import *
from tqdm import tqdm
from argparse import ArgumentParser
from utils import count_parameters
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import f1_score,precision_score,recall_score
from model import Task2SentencePairModel


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
    with open('./output/inference.txt', mode='w') as f:
        for label, pred in zip(true_labels, pred_labels):
            f.write(str(label) + "," + str(pred) + "\n")
    valid_f1_score = f1_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    valid_precision_score = precision_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    valid_recall_score = recall_score(y_true=true_labels, y_pred=pred_labels, average='binary')
    return valid_f1_score,valid_precision_score,valid_recall_score

def load_model(model_type,model_name,model_path,device):
    if model_type=="bert":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = Task2SentencePairModel(pretrained_model)
    elif model_type=="ernie":
        model_config = ErnieConfig.from_pretrained(model_name)
        pretrained_model = ErnieModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = Task2SentencePairModel(pretrained_model)
    elif model_type == "nezha":
        model_config = NezhaConfig.from_pretrained(model_name)
        pretrained_model = NezhaModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = Task2SentencePairModel(pretrained_model)
    elif model_type=="electra":
        model_config = ElectraConfig.from_pretrained(model_name)
        pretrained_model = ElectraModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Task2SentencePairModel(pretrained_model)
    else:
        model_config = AutoConfig.from_pretrained(model_name)
        pretrained_model = AutoModel.from_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = Task2SentencePairModel(pretrained_model)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(args.device)
    print("Load model {} successfully!".format(model_name))
    model.eval()
    return model,tokenizer


@torch.no_grad()
def inference(args):
    Path('output').mkdir(exist_ok=True)
    if args.mode == "valid":
        text_file = './data/new_valid_text.jsonl'
    else:
        text_file = './data/new_test_text.jsonl'
    out_file: str = os.path.join("output", args.model_type + "_" + str(args.mode)+"_pair.jsonl")

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


    model,tokenizer = load_model(args.model_type,args.model_name,args.model_path,args.device)
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
        input = tokenizer(sc,text_pair=bc,max_length=512,truncation=True,return_tensors='pt')
        input_ids = input['input_ids'].to(args.device)
        attention_mask = input['attention_mask'].to(args.device)
        token_type_ids = input['token_type_ids'].to(args.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prob = torch.sigmoid(logits)
        if prob > args.threshold:
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
    if args.mode=="valid":
        valid_f1_score,valid_precision_score,valid_recall_score = cal_f1_score_from_file(text_file='./data/new_valid_text.jsonl',pair_file='./data/new_valid_pair.jsonl',pred_file=out_file)
        print("Valid F1-Score:{}".format(valid_f1_score))
        print("Valid Precision-Score:{}".format(valid_precision_score))
        print("Valid Recall-Score:{}".format(valid_recall_score))

if __name__ == '__main__':
    parser = ArgumentParser()
    # ernie-3.0-base复现结果有点差异
    parser.add_argument("--model_name", type=str, default=r"E:\model\ernie-3.0-base-zh",
                        help="预训练模型名")
    parser.add_argument("--model_path", type=str, default=r"E:\邀请赛2模型\0.51047_ernie-3.0-base-zh.bin",
                        help="训练好用于推理的模型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--model_type", type=str, default=r"ernie",choices=['bert','ernie','nezha'],
                        help="bert,ernie,or nezha")
    parser.add_argument("--threshold", type=float, default="0.3",
                        help="分类阈值，超参数")
    parser.add_argument("--mode", type=str, default="valid",choices=['valid','test'],
                        help="选择推理的模式，是在验证集上，还是在测试集（即用于最终提交）")
    args = parser.parse_args()

    inference(args)
