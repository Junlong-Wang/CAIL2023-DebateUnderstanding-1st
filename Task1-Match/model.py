import torch
import torch.nn as nn
from transformers import BertModel,BertConfig,ErnieModel,NezhaModel,DebertaModel,BertTokenizer
from transformers import BertForMultipleChoice
from torch.cuda import amp


# Task1:争议观点对匹配模型
class MatchModel(nn.Module):

    def __init__(self,pretrained_model,num_choices=5):
        super(MatchModel, self).__init__()
        self.num_choices = num_choices
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self,input_ids,attention_mask,token_type_ids):
        # copy from:BertForMultipleChoice
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # CLS池化
        # TODO：尝试其他池化
        # 您提供的代码片段来自 MatchModel 类的forward 方法。
        pooled_output = outputs[1]
        # `pooled_output = self.dropout(pooled_output)` 行正在对 `pooled_output` 张量应用 dropout 正则化。 Dropout
        # 是一种用于防止神经网络过度拟合的正则化技术。它在每个训练步骤中随机将一部分输入单元设置为 0，这有助于防止模型过于依赖任何特定的输入特征。在这种情况下，“dropout”模块的 dropout 率为
        # 0.3，这意味着“pooled_output”中 30% 的元素在训练期间将被设置为 0。
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        # shape:[batch_size,num_choices]
        return reshaped_logits






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

class MatchModel1(nn.Module):
    def __init__(self, pretrained_model, num_choices=5):
        super(MatchModel, self).__init__()
        self.num_choices = num_choices
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, 1)
        self.attention_pooling = AttentionPooling(pretrained_model.config.hidden_size)  # 使用AttentionPooling进行Attention Pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Attention Pooling
        attention_pooled = self.attention_pooling(outputs.last_hidden_state, attention_mask)

        pooled_output = attention_pooled  # 使用Attention Pooling，不包括Mean Pooling

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        return reshaped_logits





class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embedding = sum_embeddings / sum_mask
        return mean_embedding

class MatchModel2(nn.Module):
    def __init__(self, pretrained_model, num_choices=5):
        super(MatchModel, self).__init__()
        self.num_choices = num_choices
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, 1)
        self.mean_pooling = MeanPooling()

    def forward(self, input_ids, attention_mask, token_type_ids):
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用 MeanPooling 代替 CLS 池化
        mean_pooled_output = self.mean_pooling(outputs[0], attention_mask)
        mean_pooled_output = self.dropout(mean_pooled_output)
        logits = self.classifier(mean_pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)
        return reshaped_logits


