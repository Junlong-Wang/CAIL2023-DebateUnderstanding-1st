import torch
from transformers import BertModel,BertConfig,BertTokenizer,ErnieModel,ErnieConfig,NezhaConfig,NezhaModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from pandas import DataFrame
from json import dumps
import transformers.utils.logging
transformers.logging.set_verbosity_error()

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

# Task2
class Task2SentencePairModel(nn.Module):

    def __init__(self,pretrained_model):
        super(Task2SentencePairModel, self).__init__()
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self,input_ids,attention_mask,token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs[0]
        pooled_output = MeanPooling()(last_hidden_state,attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1)
        # shape:[batch_size,num_choices]
        return reshaped_logits


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings
def generate_new_test_data():
    # 已检查过，没有重复的行
    test_text = pd.read_json('./data/test_text.jsonl', lines=True)
    multi_test_text = pd.read_json('./data/multi_test_text.jsonl',lines=True)
    print(len(test_text))
    print(len(multi_test_text))
    new_test_text = pd.concat([test_text,multi_test_text], axis=0)
    new_test_text.reset_index(drop=True)

    new_test_text.to_json('./data/new_test_text.jsonl', orient='records', lines=True)
    print("合并测试集！")



def load_model(model_type, config_path, model_path, device):

    if model_type == "bert":
        model_config = BertConfig.from_pretrained(config_path)
        pretrained_model = BertModel(model_config)
    elif model_type == "ernie":
        model_config = ErnieConfig.from_pretrained(config_path)
        pretrained_model = ErnieModel(model_config)
    elif model_type == "nezha":
        model_config = NezhaConfig.from_pretrained(config_path)
        pretrained_model = NezhaModel(model_config)
    tokenizer = BertTokenizer.from_pretrained(config_path)

    model = Task2SentencePairModel(pretrained_model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Load model {} successfully!".format(config_path))
    model.eval()
    return model, tokenizer

@torch.no_grad()
def inference(device, model_list, batch_size,threshold=0.3):
    Path('output').mkdir(exist_ok=True)
    text_file = './data/new_test_text.jsonl'
    out_file: str = os.path.join("output", "test_pair.jsonl")

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
                    "sc_id": sc_id,
                    "bc": text_data.loc[
                        (text_data['sentence_id'] == bc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "bc_id": bc_id
                })
    dataloader = build_test_dataloader(text_path=text_file, batch_size=batch_size)
    trained_models = []
    for m in model_list:
        model, tokenizer = load_model(m["model_type"], m["config_path"], m["model_path"], device)
        trained_models.append((model, tokenizer,))
    # parameters = count_parameters(model)
    # print(parameters)

    outputs = []
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in bar:
        probs = torch.zeros(len(batch_data)).to(device)
        max_len = max([len(d['sc'] + d['bc']) for d in batch_data]) + 3
        max_len = max_len if max_len <= 512 else 512
        for model, tokenizer in trained_models:
            input_ids = []
            attention_mask = []
            token_type_ids = []
            # +3 sc bc

            for d in batch_data:
                output = tokenizer(d['sc'], text_pair=d['bc'], padding='max_length', truncation=True,
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
        for i, pred in enumerate(preds):
            if pred == 1:
                outputs.append({
                    "text_id": batch_data[i]['text_id'],
                    "sc_id": batch_data[i]['sc_id'],
                    "bc_id": batch_data[i]['bc_id']
                })

    # 结果写入提交要求格式文件
    with open(out_file, 'w', encoding='utf8') as f:
        for output in outputs:
            print(dumps({
                'text_id': output["text_id"],
                'sc_id': output['sc_id'],
                'bc_id': output['bc_id']
            }), file=f)

if __name__ == '__main__':
    model_list = [
        {"model_type":"ernie","config_path":"./config/ernie-3.0-base-zh","model_path":"./model/0.55038_ernie-3.0-base-zh.bin"},
        {"model_type":"ernie","config_path":"./config/ernie-3.0-xbase-zh","model_path":"./model/0.55090_ernie-3.0-xbase-zh.bin"},
        {"model_type":"nezha","config_path":"./config/nezha-cn-base","model_path":"./model/0.53329_nezha-cn-base.bin"},
        {"model_type":"nezha","config_path":"./config/nezha-base-wwm","model_path":"./model/0.53423_nezha-base-wwm.bin"},
                  ]
    device = "cuda:0"
    inference(device, model_list,batch_size=4,threshold=0.3)