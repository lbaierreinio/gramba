import torch.nn as nn
from models.GrambaModel import GrambaModel

class GrambaSequenceClassificationModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_layers, window_size, pad_token_id, attention_probs_dropout_prob=0.3, ratio=2, expansion_factor=4, bidirectional=False, num_classes=2):
        super().__init__()
        self.gramba_model = GrambaModel(hidden_dim, vocab_size, num_layers, window_size, pad_token_id, attention_probs_dropout_prob, ratio, expansion_factor, bidirectional)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        x = self.gramba_model(x, mask) # B_S, S_L, H_D
        x = self.classifier(x[: -1, :]) # Predict on final token (should be CLS token)
        return x