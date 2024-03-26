import pandas as pd
from inputter import get_data
from sklearn.model_selection import train_test_split

from transformers import BertForSequenceClassification
def generate_new_train_valid_data():
    data1 = get_data('data/train_text.jsonl', './data/train_pair.jsonl')
    data2 = get_data('data/multi_train_text.jsonl', './data/multi_train_pair.jsonl')
    data3 = get_data('data/dirty_train_text.jsonl', './data/dirty_train_pair.jsonl')
    data = data1 + data2 + data3

    train_data, valid_data = train_test_split(data, test_size=0.2)
    new_train_text = pd.concat([term['text'] for term in train_data], axis=0)
    new_train_text.reset_index(drop=True)

    new_valid_text = pd.concat([term['text'] for term in valid_data], axis=0)
    new_valid_text.reset_index(drop=True)

    new_train_pair = pd.concat([term['pair'] for term in train_data], axis=0)
    new_train_pair.reset_index(drop=True)

    new_valid_pair = pd.concat([term['pair'] for term in valid_data], axis=0)
    new_valid_pair.reset_index(drop=True)

    new_train_text.to_json('./data/new_train_text.jsonl', orient='records', lines=True)
    new_train_pair.to_json('./data/new_train_pair.jsonl', orient='records', lines=True)
    new_valid_text.to_json('./data/new_valid_text.jsonl', orient='records', lines=True)
    new_valid_pair.to_json('./data/new_valid_pair.jsonl', orient='records', lines=True)

def generate_new_test_data():
    # 已检查过，没有重复的行
    test_text = pd.read_json('./data/test_text.jsonl', lines=True)
    multi_test_text = pd.read_json('./data/multi_test_text.jsonl',lines=True)
    print(len(test_text))
    print(len(multi_test_text))
    new_test_text = pd.concat([test_text,multi_test_text], axis=0)
    new_test_text.reset_index(drop=True)

    new_test_text.to_json('./data/new_test_text.jsonl', orient='records', lines=True)


def count_data():
    # 训练集和测试集无重合
    test_text = pd.read_json('./data/test_text.jsonl',lines=True)
    test_text_groups = test_text.groupby(['text_id'],sort=False)
    test_text_ids = test_text_groups.groups.keys()
    test_text_ids = set(test_text_ids)
    print(test_text_ids)

    train_text = pd.read_json('./data/train_text.jsonl',lines=True)
    train_text_groups = train_text.groupby(['text_id'],sort=False)
    train_text_ids = train_text_groups.groups.keys()
    train_text_ids = set(train_text_ids)
    print(train_text_ids)

    if train_text_ids.intersection(test_text_ids):
        print("有重合")
    else:
        print("无重合")
    # dup = text.duplicated()

def check_output_data():
    '''
    检查测试集推理数据的sc_id和bc_id是否能和原text文件对应
    :return:
    '''
    text = pd.read_json('./data/new_test_text.jsonl',lines=True)
    text_ids = text['text_id'].unique().tolist()
    total_pairs = []
    for text_id in text_ids:
        sc_ids = text.loc[(text['position'] == 'sc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text.loc[(text['position'] == 'bc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                total_pairs.append((text_id,sc_id,bc_id))
    result = pd.read_json(r'./data/pseudo_data.jsonl',lines=True)
    result_text_ids = result['text_id'].unique().tolist()

    for text_id in result_text_ids:
        sc_ids = text.loc[(text['position'] == 'sc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        bc_ids = text.loc[(text['position'] == 'bc') & (text['text_id'] == text_id), 'sentence_id'].tolist()
        for sc_id in sc_ids:
            for bc_id in bc_ids:
                if (text_id,sc_id,bc_id) not in total_pairs:
                    print("错误！")

if __name__ == '__main__':
    check_output_data()