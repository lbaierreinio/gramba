import torch
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

    def forward(self, x, mask=None, is_sequential=False):
        # Gramba with residual connection
        if is_sequential:
            h_prev = torch.zeros((x.shape[0], x.shape[2])).to(x.device) # Initial hidden state
            for t in range(x.shape[1]): # Iterate over sequence length dimension
                if mask is not None:
                    x_t, h_prev = self.gramba(x[:, t], mask=mask[:, t], h_prev=h_prev)
                else: 
                    x_t, h_prev = self.gramba(x[:, t], h_prev=h_prev)
                x[:, t] = x_t
        else: 
            x = x + self.gramba(x, mask)
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
