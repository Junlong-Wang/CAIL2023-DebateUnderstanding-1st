import torch
import torch.nn as nn
from torch.nn import functional as F


# 交叉熵损失
def criterion(logits, labels):
    return nn.BCEWithLogitsLoss()(logits,labels)

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoid获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt)- (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

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

    def forward(self, last_hidden_state, attention_mask):
        w = self.attention(last_hidden_state).float()
        w[attention_mask==0] = float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings


class MeanPooling(nn.Module):

    def __init__(self):
        super(MeanPooling,self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask,min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return mean_embedding

