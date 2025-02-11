import torch.nn as nn
import torch
from attention.widowedAttention import WidowedSelfAttention, pad_to_window_size

class SWABlock(nn.Module):
    def __init__(self, hidden_dim, window_size, pad_token_id, expansion_factor=4):
        super().__init__()
        self.swa = WidowedSelfAttention(hidden_dim, num_attention_heads=8, attention_probs_dropout_prob=0.3, window_size=window_size, output_attentions=False)
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
        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        attention_mask = torch.ones(x.shape, dtype=torch.long, device=x.device) # initialize to local attention

        # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
        x, attention_mask = pad_to_window_size(x, attention_mask, self.window_size, self.pad_token_id)
        x = x +  self.swa(x, mask=attention_mask)
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
