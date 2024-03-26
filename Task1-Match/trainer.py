from argparse import ArgumentParser

import gc
import copy
import time

import transformers.utils.logging
from torch.cuda import amp
from transformers import AdamW, get_linear_schedule_with_warmup
transformers.utils.logging.set_verbosity_error()
from tqdm import tqdm
from inputter import *
import os
import warnings
warnings.filterwarnings("ignore")
from model_utils import criterion
from utils import *
from pathlib import Path


def train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader, epoch):

    model.train()
    # 自动混合精度
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids, attention_mask, token_type_ids, labels = data
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        labels = labels.to(args.device)

        batch_size = input_ids.size(0)

        # 正常训练
        with amp.autocast(enabled=True):
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            loss = loss / args.n_accumulate
        # 反向传播得到正常的grad
        scaler.scale(loss).backward()

        #----------------对抗训练----------------------#
        #FGM对抗训练#
        if args.ad_train=='FGM':

            attacker.attack()

            with amp.autocast(enabled=True):
                outputs_adv = model(input_ids, attention_mask, token_type_ids)
                loss_adv = criterion(outputs_adv, labels)
                loss_adv = loss_adv / args.n_accumulate

            scaler.scale(loss_adv).backward()
            attacker.restore()
        #PGD对抗训练#
        # 该代码块是 PGD（投影梯度下降）对抗训练方法的一部分。
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
                    outputs_adv = model(input_ids, attention_mask, token_type_ids)
                    loss_adv = criterion(outputs_adv, labels)
                    loss_adv = loss_adv / args.n_accumulate
                    # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    scaler.scale(loss_adv).backward()

            # 恢复embedding参数
            attacker.restore()



        # 该代码块负责在一定数量的累积步骤后更新模型的参数和优化器的梯度。
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
def valid_one_epoch(model,model_name, optimizer, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    acc = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        input_ids, attention_mask, token_type_ids, labels = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        preds = outputs.argmax(dim=1)
        # P@1:top-1准确率
        acc += (preds == labels).sum().item()


        bar.set_postfix(Epoch=epoch,Valid_Loss=epoch_loss,Model=model_name,
                        LR=optimizer.param_groups[0]['lr'])
    # TODO：检查
    valid_acc = acc / dataset_size


    gc.collect()

    return epoch_loss,valid_acc


def valid(model,model_name, dataloader, device, epoch):
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    acc = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        input_ids, attention_mask, token_type_ids, labels = data
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)

        batch_size = input_ids.size(0)

        outputs = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(outputs, labels)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        preds = outputs.argmax(dim=1)
        # P@1:top-1准确率
        acc += (preds == labels).sum().item()


        bar.set_postfix(Epoch=epoch,Valid_Loss=epoch_loss,Model=model_name)
    # TODO：检查
    valid_acc = acc / dataset_size


    gc.collect()

    return epoch_loss,valid_acc


def train(args,tokenizer,idx,model,model_name):

    model.to(args.device)
    print(f'train现在是第{idx}个客户端的训练')

    # train_loader, valid_loader = prepare_cv_data(args,tokenizer)
    train_loader,valid_loader = build_dataloader(args,tokenizer,args.batch_size,idx)
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

    # 此代码块检查“args.ad_train”变量的值。如果等于“FGM”，则打印一条消息，指示对抗性训练方法是FGM（快速梯度法），并初始化FGM攻击者对象。如果“args.ad_train”等于“PGD”，它会打印一条消息，指示对抗性训练方法是PGD（投影梯度下降），并初始化PGD攻击者对象。攻击者对象在训练过程中使用来生成对抗性示例并提高模型针对对抗性攻击的鲁棒性。
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
    # `best_model_wts = copy.deepcopy(model.state_dict())`
    # 正在创建模型参数当前状态的深层副本。这样做是为了在训练期间保存最佳模型权重。通过创建深度副本，我们确保最佳模型权重单独保存，并且不会受到模型参数后续任何更改的影响。
    best_model_wts = copy.deepcopy(model.state_dict())
    # 初始化为负无穷
    best_epoch_acc = -np.inf

    for epoch in range(1, args.n_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, args, optimizer, scheduler, attacker, dataloader=train_loader, epoch=epoch)
        valid_epoch_loss, valid_acc = valid_one_epoch(model= model,model_name= model_name,
                                                     optimizer=optimizer,
                                                     dataloader=valid_loader,
                                                     device=args.device,epoch=epoch)

        print(f'Valid Loss: {valid_epoch_loss}')
        print(f'Valid Accuracy: {valid_acc}')
        # deep copy the model
        # 保存具有最好ACC的模型权重.
        if valid_acc >= best_epoch_acc:
            print(f"Validation Accuracy Improved ({best_epoch_acc} ---> {valid_acc})")
            best_epoch_acc = valid_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    print("Best Acc: {:.5f}".format(best_epoch_acc))

    # -----------保存模型-------------
    # 检查文件夹地址是否存在，不在则创建
    Path(args.checkpoint_path).mkdir(exist_ok=True)
    # example: hfl/chinese-bert-wwm-ext -> chinese-bert-wwm-ext
    model_saved_dir = model_name.split('/')[-1]
    Path(os.path.join(args.checkpoint_path,model_saved_dir))
    PATH = os.path.join(args.checkpoint_path,
                        "{:.5f}_{}.bin".format(best_epoch_acc,model_saved_dir))
    # torch.save(best_model_wts, PATH)
    print("Model Saved!")
    # load best model weights
    model.load_state_dict(best_model_wts)
    return best_model_wts,valid_epoch_loss


if __name__ == '__main__':
    pass




