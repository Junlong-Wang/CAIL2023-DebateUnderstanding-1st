import json
import os.path
from json import dumps
from pathlib import Path
import jsonlines
import torch
from transformers import BertTokenizer,BertConfig,BertModel,ErnieConfig,ErnieModel,ElectraConfig,ElectraModel,AutoTokenizer,AutoConfig,AutoModel
import transformers.utils.logging
transformers.utils.logging.set_verbosity_error()
from model22 import *
from tqdm import tqdm
from argparse import ArgumentParser

'''
执行在测试集上的推理，生成推理结果，写入predict.jsonl
'''


def load_model(model_type,model_name,model_path,device):
    #注意空壳模型Pool方式要和加载的模型一致
    if model_type=="bert":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type=="ernie":
        model_config = ErnieConfig.from_pretrained(model_name)
        pretrained_model = ErnieModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type == "mrc-roberta":
        model_config = BertConfig.from_pretrained(model_name)
        pretrained_model = BertModel(model_config)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    elif model_type=="electra":
        model_config = ElectraConfig.from_pretrained(model_name)
        pretrained_model = ElectraModel(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)
    else:
        model_config = AutoConfig.from_pretrained(model_name)
        pretrained_model = AutoModel.from_config(model_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MatchModel(pretrained_model, num_choices=5)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(args.device)
    print("Load model successfully!")
    model.eval()
    return model,tokenizer

@torch.no_grad()
def inference(args):
    Path('output').mkdir(exist_ok=True)
    in_file: str = 'data/test_entry.jsonl'
    out_file: str = os.path.join("output",args.model_type+"_"+"out2.json")

    with jsonlines.open(in_file) as reader:
        data = [row for row in reader]

    model,tokenizer = load_model(args.model_type,args.model_name,args.model_path,args.device)

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
        bc = [bc_1,bc_2,bc_3,bc_4,bc_5]
        input = tokenizer(sc,text_pair=bc,padding='max_length', truncation=True, max_length=512,
                             return_tensors='pt')
        input_ids = input["input_ids"].unsqueeze(0).to(args.device)
        attention_mask = input["attention_mask"].unsqueeze(0).to(args.device)
        token_type_ids = input["token_type_ids"].unsqueeze(0).to(args.device)
        output = model(input_ids,attention_mask,token_type_ids).to(args.device)
        # +1，映射到1-5之间
        pred = output.argmax(dim=1).cpu().item()+1
        d["answer"] = pred  # 将预测的答案添加到原数据
        # `d["probability"] = output.softmax(dim=1).max().cpu().item()` 行计算预测答案的概率并将其添加到原始数据中。
        d["probability"] = output.softmax(dim=1).max().cpu().item()# 将预测的概率添加到原数据
        preds.append(d)


    with open(out_file, 'w', encoding='utf-8') as file:
        json.dump(preds, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default=r"E:\CAIL\CAIL-LBLJ\checkpoint\ernie-3.0-xbase-zh",
                        help="预训练模型名")
    parser.add_argument("--model_path", type=str, default=r"E:\CAIL\CAIL-LBLJ\运行模型\ernie-xbase\0.85124_ernie-3.0-xbase-zh.bin",
                        help="训练好用于推理的模型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--model_type", type=str, default=r"ernie",
                        help="bert,ernie,nezha")

    # parser.add_argument("--model_name", type=str, default=r"E:\CAIL\CAIL-LBLJ\checkpoint\luhuachinese-pretrain-mrc-roberta-wwm-ext-large",
    #                     help="预训练模型名")
    # parser.add_argument("--model_path", type=str, default=r"E:\CAIL\CAIL-LBLJ\checkpoint\0.84479_luhuachinese-pretrain-mrc-roberta-wwm-ext-large.bin",
    #                     help="训练好用于推理的模型")
    # parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
    #                     help="Device (cuda or cpu)")
    # parser.add_argument("--model_type", type=str, default=r"mrc-roberta",
    #                     help="bert,ernie,nezha,mrc-roberta")


    args = parser.parse_args()
    inference(args)
