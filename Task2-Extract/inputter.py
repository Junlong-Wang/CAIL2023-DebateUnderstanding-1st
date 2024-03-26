from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from pandas import DataFrame
import torch
import jsonlines
import os
from utils import save_data2json
from utils import set_seed
# set_seed(42)
import transformers.utils.logging
transformers.logging.set_verbosity_error()
MAX_SEQ_LEN = 512

def get_data(text_path,pair_path):
    text_data = pd.read_json(text_path, lines=True)
    pair_data = pd.read_json(pair_path, lines=True)
    pairs = []
    data = []
    for index, row in pair_data.iterrows():
        text_id = row['text_id']
        sc_id = row['sc_id']
        bc_id = row['bc_id']
        pairs.append((text_id, sc_id, bc_id))

    text_ids = text_data['text_id'].unique().tolist()
    pair_ids = pair_data['text_id'].unique().tolist()
    text_ids = [value for value in text_ids if value in pair_ids]

    for text_id in text_ids:
        sc_ids = text_data.loc[(text_data['position'] == 'sc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text_data.loc[(text_data['position'] == 'bc') & (text_data['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                if (text_id, sc_id, bc_id) in pairs:
                    label = 1
                else:
                    label = 0
                data.append({
                    "text_id":text_id,
                    "sc":text_data.loc[(text_data['sentence_id'] == sc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "bc":text_data.loc[(text_data['sentence_id'] == bc_id) & (text_data['text_id'] == text_id), 'sentence'].values[0],
                    "label":label
                })
    return data

# Task2:争议观点对抽取
class ExtractDataset(Dataset):

    def __init__(self,data,tokenizer):
        # DataFrame类型
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self,batch_data):
        input_ids = []
        attention_mask = []
        token_type_ids = []
        # +3 sc bc
        # print(batch_data)
        max_len = max([len(data['sc']+data['bc']) for data in batch_data])+3
        max_len = max_len if max_len<=512 else 512
        for data in batch_data:
            output = self.tokenizer(data['sc'],text_pair=data['bc'],padding='max_length', truncation=True, max_length=max_len,)
            input_ids.append(output['input_ids'])
            attention_mask.append(output['attention_mask'])
            token_type_ids.append(output['token_type_ids'])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        labels = torch.tensor([data["label"] for data in batch_data], dtype=torch.float)
        # shape:[bz,num_choice,seq_len]
        return input_ids, attention_mask, token_type_ids, labels


def build_dataloader(tokenizer, batch_size):
    train_data = get_data(text_path='./data/new_train_text.jsonl', pair_path='./data/new_train_pair.jsonl')
    pseudo_data = get_data(text_path='./data/new_test_text.jsonl', pair_path='./data/pseudo_data.jsonl')
    train_data = train_data + pseudo_data
    valid_data = get_data(text_path='./data/new_valid_text.jsonl', pair_path='./data/new_valid_pair.jsonl')

    train_dataset = ExtractDataset(data=train_data, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)

    valid_dataset = ExtractDataset(data=valid_data,tokenizer=tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=valid_dataset.collate_fn,shuffle=False)
    return train_dataloader,valid_dataloader

if __name__ == '__main__':

    from transformers import BertTokenizer,set_seed
    set_seed(42)
    tokenizer = BertTokenizer.from_pretrained(r'E:\model\bert-base-chinese')
    device = "cpu"
    train_dataloader,valid_dataloader = build_dataloader(tokenizer,1)
    for idx,batch in enumerate(valid_dataloader):
        # 第一条数据
        input_ids, attention_mask, token_type_ids, labels = batch
        print(input_ids.size(1))
        print(input_ids[0])
        print(input_ids[1])
        break