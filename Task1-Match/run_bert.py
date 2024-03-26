# 您提供的代码是一个 Python 脚本，用于使用 BERT（来自 Transformers 的双向编码器表示）模型训练匹配模型。以下是代码功能的细分：
import torch
import torch.nn as nn
from transformers import BertTokenizer,BertConfig,BertModel,set_seed,ErnieConfig,ErnieModel,ElectraConfig,ElectraModel,\
    AutoTokenizer,AutoConfig,AutoModel,NezhaModel, NezhaConfig
from argparse import ArgumentParser
from trainer import *
from model import MatchModel2
import warnings
from pathlib import Path
import jsonlines
import pandas as pd
from json import dumps
import tqdm
import numpy as np
import copy
warnings.filterwarnings("ignore")

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg
def save_model(model, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
def save_aggregated_model(global_model, output_dir):
    # 保存聚合后的全局模型
    # 确保 output_dir 是一个目录
    os.makedirs(output_dir, exist_ok=True)
    save_model(global_model, output_dir)
#python run_bert.py --batch_size 4 --n_accumulate 8 --learning_rate 5e-5 --n_epochs 5 --device cuda:2 --epochs 4 --num_users 2 --frac 1


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default=r"checkpoint/ernie-3.0-base-zh",
                        help="Pretrained Model Name")

    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint/",
                        help="Path or URL of the model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for train and Validation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--n_accumulate", type=int, default=4,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                        help="Dropout")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    # 对抗训练类型
    parser.add_argument('--ad_train', type=str, default="",
                        help="The type of Adversarial training Function,Default None")



    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument('--num_users', type=int, default=2, help="参与联邦学习的用户数量number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="参与每轮训练的用户比例the fraction of clients: C")

    args = parser.parse_args()
    print(args)
    MODEL_NAME = args.model_name
    # 固定随机种子
    set_seed(args.seed)
    # 加载模型
    pretrain_model = ErnieModel.from_pretrained(MODEL_NAME)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer.save_pretrained('./tokenizer/' + MODEL_NAME)
    global_model = MatchModel(pretrain_model, 5)






    global_model.to(args.device)
    global_model.train()

    # copy weights
    # `global_weights = global_model.state_dict()` 正在创建一个字典对象 `global_weights`，其中包含全局模型权重的当前状态。 `state_dict()`
    # 方法返回一个字典对象，它将每个参数名称映射到其相应的参数张量。这样可以轻松保存和加载模型权重。
    global_weights = global_model.state_dict()

    # Training
    # 代码片段“train_loss, train_accuracy = [], []”初始化空列表以存储每个时期的训练损失和准确度值。
    loss_train, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    # 此代码片段是联邦学习框架中的主要训练循环。它迭代指定数量的纪元（“args.epochs”）并执行以下步骤：
    for epoch in range(args.epochs):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # `m = max(int(args.frac * args.num_users), 1)` 行正在计算参与每轮训练的用户数量。
        m = max(int(args.frac * args.num_users), 1)
        print("每轮客户端数量",m)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        # 此代码片段正在联邦学习框架中执行本地更新步骤。
        for idx in idxs_users:
            w, loss = train(args, tokenizer , idx,copy.deepcopy(global_model),model_name=MODEL_NAME)
            print("loss值为",loss)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = FedAvg(local_weights)
        global_model.load_state_dict(global_weights)

        print("local_losses：", local_losses)

        # 此代码计算每轮训练的平均损失并将其附加到“loss_train”列表中。
        loss_avg = sum(local_losses) / len(local_losses)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))

        loss_train.append(loss_avg)



        t, valid_loader = build_dataloader(args, tokenizer, args.batch_size, idx)
        valid_epoch_loss, valid_acc = valid(model=global_model, model_name=MODEL_NAME,
                                                      dataloader=valid_loader,
                                                      device=args.device, epoch=epoch)

        print(f'Valid Loss: {valid_epoch_loss}')
        print(f'Valid Accuracy: {valid_acc}')

        # save_aggregated_model(global_model, "output/final_global_model")