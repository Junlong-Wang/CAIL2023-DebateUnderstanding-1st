import torch
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
# 交叉熵损失
def criterion(reshaped_logits, labels):
    return nn.CrossEntropyLoss()(reshaped_logits, labels)

class RDrop(nn.Module):
    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.kld = nn.KLDivLoss(reduction='none')

    def forward(self, logits1, logits2, target, kl_weight=1.):
        ce_loss = (self.ce(logits1, target) + self.ce(logits2, target))/2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + kl_weight * kl_loss
        return loss
    



# class LabelSmoothingLoss(nn.Module):
#     def __init__(self, label_smoothing, num_classes):
#         super(LabelSmoothingLoss, self).__init__()
#         self.label_smoothing = label_smoothing
#         self.num_classes = num_classes
#
#     def forward(self, logits, labels):
#         # 计算真实标签的交叉熵损失
#         ce_loss = nn.CrossEntropyLoss()(logits, labels)
#
#         # 计算平滑标签的损失
#         smooth_loss = -torch.log_softmax(logits, dim=1)
#         smooth_loss = smooth_loss.mean()
#
#         # 组合真实标签损失和平滑标签损失
#         loss = (1.0 - self.label_smoothing) * ce_loss + self.label_smoothing * smooth_loss
#
#         return loss




# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class LabelSmoothingLoss(nn.Module):
    """
    标签平滑Loss
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        """
        :param classes: 类别数目
        :param smoothing: 平滑系数
        :param dim: loss计算平均值的维度
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.loss = nn.KLDivLoss()

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # torch.mean(torch.sum(-true_dist * pred, dim=self.dim))就是按照公式来计算损失
        loss = torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        # 采用KLDivLoss来计算
        loss = self.loss(pred, true_dist)
        return loss



class MultiSampleClassifier(nn.Module):
    def __init__(self, args, input_dim=128, num_labels=2):
        super(MultiSampleClassifier, self).__init__()
        self.args = args

        self.linear = nn.Linear(input_dim, num_labels)

        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(args.dropout_rate) for _ in range(self.args.dropout_num)]
        )

    def forward(self, x):
        logits = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)

            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits

        if self.args.ms_average:
            logits = logits / self.args.dropout_num

        return logits


class WeightedLayerPooling(nn.Module):
    def __init__(self,num_hidden_layer,layer_start:int = 4,layer_weights = None):
        super(WeightedLayerPooling).__init__()
        self.layer_start = layer_start
        self.num_hidden_layer = num_hidden_layer
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
            torch.tensor([1] * (num_hidden_layer+1 - layer_start), dtype=torch.float)
        )

    def forward(self,ft_all_layers):
        all_layer_embedding = torch.stack(ft_all_layers)
        all_layer_embedding = all_layer_embedding[self.layer_start:,:,:,:]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weight_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()

        return weight_average


class AttentionPooling(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim,in_dim),
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim,1)
        )

    def forward(self,last_hidden_sate,attention_mask):
        w = self.attention(last_hidden_sate).float()
        w[attention_mask==0] = float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_sate,dim=1)
        return attention_embeddings


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling,self).__init__()

    def forward(self,last_hidden_sate,attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_sate.size())
        sum_embeddings = torch.sum(last_hidden_sate * input_mask_expanded,1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask,min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return mean_embedding

