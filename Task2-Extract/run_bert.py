import torch
import torch.nn as nn
from transformers import ErnieConfig,BertModel,BertConfig,AutoTokenizer,BertTokenizer,ErnieForSequenceClassification,ErnieModel,ElectraModel,NezhaModel,AutoModel
from argparse import ArgumentParser
from trainer import train
import warnings
from pathlib import Path
import jsonlines
import pandas as pd
from json import dumps
from model import Task2SentencePairModel
warnings.filterwarnings("ignore")
from utils import set_seed
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default=r"E:/model/bert-base-chinese",
                        help="Pretrained Model Name")
    parser.add_argument("--model_type", type=str, default=r"bert",
                        help="bert,ernie,nezha")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/",
                        help="Path or URL of the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for train and Validation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--n_accumulate", type=int, default=1,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--threshold", type=float, default="0.3",
                        help="分类阈值，超参数")
    parser.add_argument("--loss_func",type=str,default="ce",choices=["ce","focal"],help="损失函数类型")
    # 对抗训练类型
    parser.add_argument('--ad_train', type=str, default="",
                        help="The type of Adversarial training Function,Default None")


    args = parser.parse_args()
    print(args)
    MODEL_NAME = args.model_name
    # 固定随机种子
    set_seed(42)
    # 加载模型
    if args.model_type == "bert":
        pretrained_model = BertModel.from_pretrained(MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    elif args.model_type == "ernie":
        pretrained_model = ErnieModel.from_pretrained(MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    elif args.model_type == "nezha":
        pretrained_model = NezhaModel.from_pretrained(MODEL_NAME)
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    else:
        pretrained_model = AutoModel.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Task2SentencePairModel(pretrained_model)
    train(args,model,tokenizer,model_name=MODEL_NAME)
