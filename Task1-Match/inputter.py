import pandas as pd

import torch
from torch.utils.data import Dataset,DataLoader
import jsonlines
from sklearn.model_selection import train_test_split
from transformers import set_seed
from utils import *
## ToDo :trans the raw data into dataset and dataloaer
"""
上面的代码定义了一个 MatchDataset 类，它将原始数据转换为数据集和一个用于训练和验证的数据加载器。
"""



# 固定随机种子
set_seed(42)


# 用不上了
def data2json():
    path = "./data/0fold_train.jsonl"
    with jsonlines.open(path) as reader:
        data = [row for row in reader]
    #   按比例划分训练集和测试机，8：2
    # `train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=True)` 行将数据分成训练集和验证集。它随机打乱数据并将其分成两组，其中 80%
    # 的数据分配给训练集（“train_data”），20% 的数据分配给验证集（“valid_data”）。
    train_data, valid_data = train_test_split(data, test_size=0.2, shuffle=True)
    save_data2json('./data/train.json',train_data)
    save_data2json('./data/valid.json',valid_data)

#数据划分
def partition_dataset(data, idx,args):
    total_samples = len(data)
    samples_per_partition = total_samples // args.num_users

    start_idx = idx * samples_per_partition
    end_idx = (idx + 1) * samples_per_partition if idx < args.num_users - 1 else total_samples

    sliced_data = data[start_idx:end_idx]

    return sliced_data


# 读取数据
def get_data(args,idx,train_path='./data/train.json',valid_path='./data/valid.json'):
    # `train_data = pd.DataFrame(load_json2data(train_path))` 行将加载的 JSON 数据转换为 pandas DataFrame。函数
    # load_json2data(train_path) 读取 JSON 文件并返回字典列表，其中每个字典代表一个数据条目。然后，“pd.DataFrame()”函数将此字典列表转换为 DataFrame，其中每个字典都成为
    # DataFrame 中的一行。
    train_data = pd.DataFrame(load_json2data(train_path))
    sliced_data = partition_dataset(train_data, idx,args)

    valid_data = pd.DataFrame(load_json2data(valid_path))
    return sliced_data, valid_data



# 参考:https://www.biendata.xyz/models/category/6353/
# Task1:争议观点对匹配
class MatchDataset(Dataset):

    def __init__(self,df_data,tokenizer):
        # DataFrame类型
        self.df_data = df_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        # 将一条数据从(问题,4个选项)转成(问题,选项1)、(问题,选项2).

        # 注意，数据标签为1-5，映射到0-4
        label = self.df_data.answer.values[index]-1
        # 诉方观点，复制*5
        sc = [self.df_data.sc.values[index] for _ in range(5)]
        # 辩方观点选项
        choices = [self.df_data.bc_1.values[index],
                   self.df_data.bc_2.values[index],
                   self.df_data.bc_3.values[index],
                   self.df_data.bc_4.values[index],
                   self.df_data.bc_5.values[index]]

        # 计算最大长度  # [CLS]sc[SEP]choice[SEP] 所以+3
        max_len = max([len(choice) for choice in choices]) + len(sc[0]) + 3
        max_len = max_len if max_len<=512 else 512
        return {
            "sc":sc,
            "choices":choices,
            "label":label,
            "max_len":max_len
        }

    # paddding,token2id
    # 将问题选项拼在一起后，得到分词后的数字id，输出的size是(batch, n_choices, max_len)
    def collate_fn(self,batch_data):
        input_ids, attention_mask, token_type_ids = [], [], []
        # 不要全补齐到512，补齐成batch数据中的最大长度，节约一下显存。
        max_len = max([data["max_len"] for data in batch_data])
        for data in batch_data:
            # 拼接sc和bc，补齐，超长截断。
            # TODO：这里截断和补全可能有些得不合适的
            input = self.tokenizer(data["sc"], text_pair=data["choices"], padding='max_length', truncation=True, max_length=max_len,
                             return_tensors='pt')
            input_ids.append(input['input_ids'].tolist())
            attention_mask.append(input['attention_mask'].tolist())
            token_type_ids.append(input['token_type_ids'].tolist())
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        attention_mask = torch.tensor(attention_mask,dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids,dtype=torch.long)
        labels = torch.tensor([data["label"] for data in batch_data],dtype=torch.long)
        # shape:[bz,num_choice,seq_len]
        return input_ids,attention_mask,token_type_ids,labels

def build_dataloader(args,tokenizer, batch_size,idx):
    """
    函数“build_dataloader”将分词器和批量大小作为输入，检索训练和验证数据，使用数据和分词器创建数据集，并返回训练和验证数据加载器。

    :param tokenizer: 分词器负责将文本转换为标记，这是神经网络输入的基本单位。它用于预处理文本数据并将其转换为可以输入模型的格式。
    :param batch_size: batch_size 参数确定数据加载器每次迭代中将处理的样本数量。它用于控制内存使用和训练速度。
    :return: 函数“build_dataloader”返回两个数据加载器：“train_dataloader”和“valid_dataloader”。
    """
    train_data,valid_data = get_data(args,idx,train_path='./data/train.json',valid_path='./data/valid.json')
    train_dataset = MatchDataset(df_data=train_data, tokenizer=tokenizer)
    valid_dataset = MatchDataset(df_data=valid_data, tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size,collate_fn=valid_dataset.collate_fn,shuffle=True)
    return train_dataloader,valid_dataloader

if __name__ == '__main__':
    pass
