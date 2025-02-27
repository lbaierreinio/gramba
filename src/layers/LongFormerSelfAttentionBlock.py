import torch.nn as nn
import torch.nn.functional as F
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

class LongFormerSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, window_size, pad_token_id, expansion_factor=4, num_attention_heads=2):
        super().__init__()
        assert window_size % 2 == 0, "Window size must be even"
        self.pad_token_id = pad_token_id
        self.longformer = LongformerSelfAttention(
            hidden_size = hidden_dim,
            num_attention_heads = num_attention_heads,
            embed_dim = hidden_dim,
            attention_window = window_size,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )
        self.window_size = window_size
        self.pad_token_id = pad_token_id

    def forward(self, x, mask, is_sequential=False):
        # Ensure that the sequence length is even
        if x.size(1) % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), value=self.pad_token_id)
            longformer_mask = F.pad(mask, (0, 1), value=True)

        longformer_mask = longformer_mask * -10000 # Local attention to all non-padded values
        longformer_mask[:, -1] = 10000 # Global attention to CLS token

        # LongFormer with residual connection
        x = x + self.longformer(x, longformer_mask)
        # Remove the padding token
        x = x[:, :-1]
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
