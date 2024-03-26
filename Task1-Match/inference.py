import json
import os.path
from json import dumps
from pathlib import Path
import jsonlines
import torch
from transformers import BertTokenizer,BertConfig,BertModel,ErnieConfig,ErnieModel,ElectraConfig,ElectraModel,\
    AutoTokenizer,AutoConfig,AutoModel,NezhaModel, NezhaConfig
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
    elif model_type == "nezha":
        model_config = NezhaConfig.from_pretrained(model_name)
        pretrained_model = NezhaModel(model_config)
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
    out_file: str = os.path.join("output",args.model_type+"_"+"predict.jsonl")

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
        preds.append({
            "id":id,
            "answer":pred
        })

    # 结果写入提交要求格式文件
    with open(out_file, 'w', encoding='utf8') as f:
        for pred in preds:
            print(dumps({
                'id': pred["id"],
                'answer': pred['answer']
            }), file=f)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str, default=r"E:\CAIL\CAIL-LBLJ\checkpoint\ernie-3.0-xbase-zh",
                        help="预训练模型名")
    parser.add_argument("--model_path", type=str, default=r"E:/CAIL/CAIL-LBLJ/运行模型/erniexbase_train_fold_1/0.87666_ernie-3.0-xbase-zh.bin",
                        help="训练好用于推理的模型")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--model_type", type=str, default=r"ernie",
                        help="bert,ernie,nezha")
    args = parser.parse_args()
    # model_list = [{"model_type":"bert","model_name":r"G:\huggingface-hub\chinese-roberta-wwm-ext-large","model_path":"./checkpoint/0.81847_chinese-roberta-wwm-ext-large.bin"},
    #               {"model_type":"ernie","model_name":r"G:\huggingface-hub\ernie-3.0-xbase-zh","model_path":"./checkpoint/0.84640_ernie-3.0-xbase-zh.bin"},
    #               ]
    # voting(args.device,model_list)
    inference(args)
