import torch.nn as nn


class Block(nn.Module):
    def __init__(self, a, hidden_dim, expansion_factor=4, dropout=0.3):
        super().__init__()
        self.a = a
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x, mask=None, is_sequential=False):
        # Residual connection and LayerNorm
        x = x + self.a(self.ln1(x), mask, is_sequential=is_sequential)

        # MLP 1 with residual connection
        return x + self.mlp(self.ln2(x))
