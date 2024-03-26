from argparse import ArgumentParser
import gc
import copy
import time

import torch
import transformers.utils.logging
from torch.cuda import amp
from transformers import AdamW, get_linear_schedule_with_warmup
transformers.utils.logging.set_verbosity_error()
from tqdm import tqdm
from inputter import *
import os
import warnings
warnings.filterwarnings("ignore")
from model_utils import *
from utils import *
from pathlib import Path
from sklearn.metrics import f1_score,precision_score,recall_score



def train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader, epoch):

    model.train()
    # 自动混合精度
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    if args.loss_func == "ce":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_func == "focal":
        criterion = BCEFocalLoss()
    for step, data in bar:
        input_ids, attention_mask, token_type_ids, labels = data
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        labels = labels.to(args.device)
        batch_size = input_ids.size(0)

        # 正常训练
        with amp.autocast(enabled=True):
            logits = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
            loss = criterion(logits,labels)
            # logits = output[1]
            # print(logits)
            # print(logits.shape)
            # print(batch_labels.dtype)
            # print(batch_labels.shape)
            # loss = criterion(logits, labels)
            # outputs1 = model(input_ids, attention_mask, token_type_ids)
            # outputs2 = model(input_ids, attention_mask, token_type_ids)
            # loss = RDrop()(outputs1, outputs2, labels)
            # loss = loss.mean()
            loss = loss / args.n_accumulate
        # 反向传播得到正常的grad
        scaler.scale(loss).backward()

        #----------------对抗训练----------------------#
        #FGM对抗训练#
        if args.ad_train=='FGM':

            attacker.attack()

            with amp.autocast(enabled=True):
                logits = model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                loss_adv = criterion(logits,labels)
                # logits = output[1]
                # loss_adv = criterion(logits,labels)
                # outputs_adv1 = model(input_ids, attention_mask, token_type_ids)
                # outputs_adv2 = model(input_ids, attention_mask, token_type_ids)
                # loss_adv = RDrop()(outputs_adv1, outputs_adv2, labels)
                # loss_adv = loss_adv.mean()
                loss_adv = loss_adv / args.n_accumulate

            scaler.scale(loss_adv).backward()
            attacker.restore()
        #PGD对抗训练#
        elif args.ad_train=='PGD':
            # 这行代码别忘了
            attacker.backup_grad()
            # 这个参数可调，暂时不调
            k = 3
            for t in range(k):
                # 在embedding上添加对抗扰动, first attack时备份param.data
                attacker.attack(is_first_attack=(t == 0))
                if t != k - 1:
                    model.zero_grad()
                else:
                    attacker.restore_grad()
                with amp.autocast(enabled=True):
                    logits= model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
                    loss_adv = criterion(logits,labels)
                    # outputs_adv1 = model(input_ids, attention_mask, token_type_ids)
                    # outputs_adv2 = model(input_ids, attention_mask, token_type_ids)
                    # loss_adv = RDrop()(outputs_adv1, outputs_adv2, labels)
                    # loss_adv = loss_adv.mean()
                    loss_adv = loss_adv / args.n_accumulate
                    # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    scaler.scale(loss_adv).backward()

            # 恢复embedding参数
            attacker.restore()



        if (step + 1) % args.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,Model=args.model_name,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()

    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model,model_name, optimizer, dataloader, device, threshold,epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    PREDS = []
    LABELS = []
    for step, data in bar:
        input_ids, attention_mask, token_type_ids, labels = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        batch_size = input_ids.size(0)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).long().cpu().tolist()
        PREDS.extend(preds)
        LABELS.extend(labels.long().cpu().tolist())
        bar.set_postfix(Epoch=epoch,Valid_Loss=epoch_loss,Model=model_name,
                        LR=optimizer.param_groups[0]['lr'])
    print("验证时预测的正样本数：",PREDS.count(1))
    print("实际标签正样本数：",LABELS.count(1))
    valid_f1_score = f1_score(y_true=LABELS,y_pred=PREDS,average='binary')
    valid_precision_score = precision_score(y_true=LABELS,y_pred=PREDS,average='binary')
    valid_recall_score = recall_score(y_true=LABELS,y_pred=PREDS,average='binary')

    gc.collect()

    return epoch_loss,valid_f1_score,valid_precision_score,valid_recall_score





def train(args,model,tokenizer,model_name):
    model.to(args.device)

    train_loader,valid_loader = build_dataloader(tokenizer=tokenizer,batch_size=args.batch_size)
    print("Len of train loader:", len(train_loader))
    print("Len of valid loader:",len(valid_loader))

    # Defining Optimizer with weight decay to params other than bias and layer norms
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    attacker = None

    if args.ad_train == "FGM":
        print('------Adversarial Training Method is {}!------'.format(args.ad_train))
        attacker = FGM(model, epsilon=1, emb_name='embeddings.word_embeddings')
    elif args.ad_train == "PGD":
        print('------Adversarial Training Method is {}!------'.format(args.ad_train))
        attacker = PGD(model,emb_name='embeddings.word_embeddings')

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate)
    # Defining LR Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * args.n_epochs
    )


    start = time.time()
    # 初始化
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化为负无穷
    best_epoch_f1_score = -np.inf

    for epoch in range(1, args.n_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader=train_loader, epoch=epoch)
        valid_epoch_loss, valid_f1_score,valid_precision_score,valid_recall_score = valid_one_epoch(model= model,model_name= model_name,
                                                     optimizer=optimizer,
                                                     dataloader=valid_loader,
                                                     device=args.device,threshold=args.threshold,epoch=epoch)

        print(f'Valid Loss: {valid_epoch_loss}')
        print(f'Valid F1-Score: {valid_f1_score}')
        print(f'Valid Precision: {valid_precision_score}')
        print(f'Valid Recall: {valid_recall_score}')
        # deep copy the model
        # 保存具有最好ACC的模型权重.
        if valid_f1_score >= best_epoch_f1_score:
            print(f"Validation F1-Score Improved ({best_epoch_f1_score} ---> {valid_f1_score})")
            best_epoch_f1_score = valid_f1_score
            best_model_wts = copy.deepcopy(model.state_dict())
        torch.cuda.empty_cache()
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    print("Best F1-Score: {:.5f}".format(best_epoch_f1_score))

    # -----------保存模型-------------
    # 检查文件夹地址是否存在，不在则创建
    Path(args.checkpoint_path).mkdir(exist_ok=True)
    # example: hfl/chinese-bert-wwm-ext -> chinese-bert-wwm-ext
    model_saved_dir = model_name.split('/')[-1]
    Path(os.path.join(args.checkpoint_path,model_saved_dir))
    PATH = os.path.join(args.checkpoint_path,
                        "{:.5f}_{}.bin".format(best_epoch_f1_score,model_saved_dir))
    torch.save(best_model_wts, PATH)
    print("Model Saved!")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model





