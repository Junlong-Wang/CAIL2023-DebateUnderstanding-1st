import json
import os.path
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


def load_model(model_type, model_name, model_path, device):
    if model_type == "bert":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel2(pretrained_model, num_choices=5)
    elif model_type == "mrc-roberta":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type == "ernie":
        model_config = ErnieConfig.from_pretrained(model_name)
        pretrained_model = ErnieModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type == "nezha":
        model_config = NezhaConfig.from_pretrained(model_name)
        pretrained_model = NezhaModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type == "electra":
        model_config = ElectraConfig.from_pretrained(model_name)
        pretrained_model = ElectraModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    else:
        model_config = AutoConfig.from_pretrained(model_name)
        pretrained_model = AutoModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)

    # model.load_state_dict(torch.load(model_path, map_location=device)) 行从指定的文件路径加载模型的训练权重并将其映射到指定的设备（例如
    # GPU）。这允许模型加载之前学习的参数，使其能够根据学习的知识进行预测。
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Load model successfully!")
    model.eval()
    return model, tokenizer


def voting2tensor():
    pass



def voting(device, model_list ,mode='soft'):
    '''
    同时加载多个模型推理，显存要求高
    利用投票机制对多个模型预测结果进行集成
    :param device:
    :param model_list:
    :param mode: 投票模式，hard or soft
    :return:
    '''
    # 加载测试集
    Path('output').mkdir(exist_ok=True)
    in_file: str = 'data/test_entry.jsonl'
    out_file: str = os.path.join("output", "voting_" + "predict.jsonl")

    with jsonlines.open(in_file) as reader:
        data = [row for row in reader]

    trained_models = []
    for m in model_list:
        model, tokenizer = load_model(m["model_type"], m["model_name"], m["model_path"], device)
        trained_models.append((model, tokenizer,))
    preds = []
    # TODO：有点慢，改成批次输入输出
    for d in tqdm(data):
        id = d["id"]
        sc = [d["sc"] for _ in range(5)]
        bc_1 = d["bc_1"]
        bc_2 = d["bc_2"]
        bc_3 = d["bc_3"]
        bc_4 = d["bc_4"]
        bc_5 = d["bc_5"]
        bc = [bc_1, bc_2, bc_3, bc_4, bc_5]
        # `prob = torch.zeros(1, 5).to(device)` 行正在创建一个形状为 (1, 5) 且填充零的张量，并将其移动到指定设备（例如 GPU）。
        prob = torch.zeros(1, 5).to(device)
        for model, tokenizer in trained_models:
            input = tokenizer(sc, text_pair=bc, padding='max_length', truncation=True, max_length=512,
                              return_tensors='pt')
            input_ids = input["input_ids"].unsqueeze(0).to(device)
            attention_mask = input["attention_mask"].unsqueeze(0).to(device)
            token_type_ids = input["token_type_ids"].unsqueeze(0).to(device)
            output = model(input_ids, attention_mask, token_type_ids).to(device)
            # 叠加概率
            prob += torch.softmax(output, dim=1)
        # 概率取平均
        prob = prob / len(trained_models)
        # +1，映射到1-5之间
        pred = prob.argmax(dim=1).cpu().item() + 1
        preds.append({
            "id": id,
            "answer": pred
        })

    # 结果写入提交要求格式文件
    with open(out_file, 'w', encoding='utf8') as f:
        for pred in preds:
            print(dumps({
                'id': pred["id"],
                'answer': pred['answer']
            }), file=f)


if __name__ == '__main__':
    model_list = [{"model_type": "ernie", "model_name": r"/root/autodl-tmp/ernie-3.0-xbase-zh",
                   "model_path": "./checkpoint/0.84909_ernie-3.0-xbase-zh.bin"},
                  {"model_type": "ernie", "model_name": r"/root/autodl-tmp/ernie-3.0-base-zh",
                   "model_path": "./checkpoint/0.84479_ernie-3.0-base-zh.bin"},
                  {"model_type": "bert", "model_name": r"/root/autodl-tmp/chinese-roberta-wwm-ext-large",
                   "model_path": "./checkpoint/0.83942_chinese-roberta-wwm-ext-large.bin"},
                  {"model_type": "mrc-roberta",
                   "model_name": r"/root/autodl-tmp/chinese_pretrain_mrc_roberta_wwm_ext_large",
                   "model_path": "./checkpoint/0.83996_luhuachinese-pretrain-mrc-roberta-wwm-ext-large.bin"},
                  {"model_type": "nezha", "model_name": r"/root/autodl-tmp/nezha-large-wwm",
                   "model_path": "/root/autodl-tmp/0.83727_nezha-large-wwm.bin"}
                  ]
    device = "cuda:0"
    voting(device, model_list)