import torch
import torch.nn as nn


class Task2SentencePairModel(nn.Module):

    def __init__(self,pretrained_model):
        super(Task2SentencePairModel, self).__init__()
        self.bert = pretrained_model
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(pretrained_model.config.hidden_size, 1)

    def forward(self,input_ids,attention_mask,token_type_ids):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        last_hidden_state = outputs[0]
        pooled_output = MeanPooling()(last_hidden_state,attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1)
        # shape:[batch_size,num_choices]
        return reshaped_logits


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings