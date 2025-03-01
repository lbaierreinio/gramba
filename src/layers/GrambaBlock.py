import torch.nn as nn
from layers.Gramba import Gramba

class GrambaBlock(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.gramba = Gramba(hidden_dim, expansion_factor, bidirectional)
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x, mask=None, is_sequential=False, token_type_ids=None):
        # Gramba with residual connection
        x = x + self.gramba(x, mask, is_sequential=is_sequential)
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
