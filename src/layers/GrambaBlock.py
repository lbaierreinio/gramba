import torch.nn as nn
from layers.Gramba import Gramba

class GrambaBlock(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.gramba = Gramba(hidden_dim, expansion_factor, bidirectional)
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.mlp_1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )
        self.swa = nn.Identity() # Placeholder for SWA
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, x, mask=None):
        # Gramba with residual connection
        x = x + self.gramba(x, mask)
        x = self.ln_1(x)

        # MLP 1 with residual connection
        x = x + self.mlp_1(x)

        # SWA layer
        x = self.swa(x)

        # LayerNorm before MLP 2
        x = self.ln_2(x)

        # MLP 2 with residual connection
        x = x + self.mlp_2(x)

        return x
