import torch.nn as nn
from models.GrambaModel import GrambaModel


class GrambaSequenceClassificationModel(nn.Module):
    def __init__(self, config):
        """
        Config should be an instance of GrambaConfig.
        """
        super().__init__()
        self.gramba_model = GrambaModel(config)
        self.classifier = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x, mask=None, is_sequential=False):
        x = self.gramba_model(x, mask, is_sequential=is_sequential) # B_S, S_L, H_D
        x = x[:, -1, :]  # Should be the CLS token
        x = self.classifier(x)
        return x