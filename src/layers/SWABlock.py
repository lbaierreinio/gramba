import torch.nn as nn

class SWABlock(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4):
        super().__init__()
        self.swa = nn.Identity() # TODO: Remove placeholder for SWA
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x, mask=None):
        # SWA with residual connection
        x = x + self.swa(x)
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
