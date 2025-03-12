import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import BertForQuestionAnswering
class BertSQuADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert_model = BertForQuestionAnswering(config)
        
    def forward(self, x, targets=None, attention_mask=None, longformer_mask=None, linformer_mask=None):
       
        start_pos_targets = targets[:,0]
        end_pos_targets = targets[:,1]
        out = self.bert_model(x, ~attention_mask, start_positions=start_pos_targets, end_positions=end_pos_targets)
        loss = out.loss
        return out, loss