import torch.nn as nn
from attention.windowedAttention import SparseWindowedAttention

class SWABlock(nn.Module):
    def __init__(self, hidden_dim, window_size, pad_token_id, expansion_factor=4, num_attention_heads=2):
        super().__init__()
        self.swa = SparseWindowedAttention(
            hidden_dim, 
            num_attention_heads,
            window_size
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )
        self.window_size = window_size
        self.pad_token_id = pad_token_id

    def forward(self, x, mask=None):
        # SWA with residual connection
        x = x +  self.swa(x, mask)
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
