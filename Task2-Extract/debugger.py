import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd
from sklearn.metrics import f1_score

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
    print(valid_f1_score)
    print(true_labels.count(1))
if __name__ == '__main__':
    # text1 = pd.read_json('./output/soft_voting_test_pair.jsonl', lines=True)
    # text2 = pd.read_json('./data/pseudo_data.jsonl',lines=True)
    # print(text1.equals(text2))
    multi_data = pd.read_json('./data/multi_test_text.jsonl',lines=True)
    multi_text = multi_data.groupby(['text_id'], sort=False)
    multi_text_ids = multi_text.groups.keys()

    pseduo_data = pd.read_json('./data/pseudo_data.jsonl', lines=True)
    pseduo_text = pseduo_data.groupby(['text_id'], sort=False)
    pseudo_text_ids = pseduo_text.groups.keys()
    count = 0
    print(len(multi_text_ids))
    for id in pseudo_text_ids:
        if id in multi_text_ids:
            count+=1
    print(count)