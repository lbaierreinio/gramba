import torch.nn as nn
from models.GrambaModel import GrambaModel
import torch.nn.functional as F


class GrambaSequenceClassificationModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, embedding_weights, num_layers, window_size, pad_token_id, attention_probs_dropout_prob=0.3, ratio=2, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.gramba_model = GrambaModel(embedding_dim, vocab_size, embedding_weights, num_layers, window_size, pad_token_id, attention_probs_dropout_prob, ratio, expansion_factor, bidirectional)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x, mask=None, is_sequential=False):
        x = self.gramba_model(x, mask, is_sequential=is_sequential) # B_S, S_L, H_D
        x = x[:, -1, :]  # Should be the CLS token
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x