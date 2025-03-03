import torch.nn as nn
from models.GrambaModel import GrambaModel


class GrambaSequenceClassificationModel(nn.Module):
    def __init__(self, config):
        """
        Config should be an instance of GrambaConfig.
        """
        super().__init__()
        self.config = config
        self.gramba_model = GrambaModel(config)
        self.ln = nn.LayerNorm(config.embedding_dim)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x, attention_mask=None, longformer_mask=None, is_sequential=False):
        x = self.gramba_model(x, attention_mask, longformer_mask, is_sequential=is_sequential) # B_S, S_L, H_D
        x = x[:, -1, :]  # Should be the CLS token
        x = self.classifier(self.ln(x))
        return x