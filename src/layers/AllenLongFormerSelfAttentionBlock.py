import torch.nn as nn
import torch.nn.functional as F
from longformer.longformer import LongformerConfig, LongformerSelfAttention

class AllenLongFormerSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, window_size, pad_token_id, expansion_factor=4, num_attention_heads=2):
        """
        NOTE: This implementation requires an early version of Transformers (e.g. 3.3.1)
        """
        super().__init__()
        assert window_size % 2 == 0, "Window size must be even"
        self.pad_token_id = pad_token_id
        self.window_size = window_size

        # Create an object for LongFormerSelfAttention's configuration
        config = LongformerConfig(
            attention_window=[window_size],
            attention_dilation=[1],
            autoregressive=False,
            attention_mode = 'sliding_chunks',
            hidden_size = hidden_dim,
            embed_dim = hidden_dim,
            num_attention_heads = num_attention_heads,
        )

        self.longformer = LongformerSelfAttention(config, layer_id=0)

        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.GELU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )
        self.window_size = window_size
        self.pad_token_id = pad_token_id

    def forward(self, x, mask, is_sequential=False):
        padding_needed = 2 * self.window_size - (x.size(1) % (2 * self.window_size))
        longformer_mask = mask
        if padding_needed > 0: # Ensure padding works as expeted
            x = F.pad(x, (0, 0, 0, padding_needed), value=self.pad_token_id)
            longformer_mask = F.pad(mask, (0, padding_needed), value=True)
        
        longformer_mask = (longformer_mask * 1) - 1
        longformer_mask[:, -1] = 1 # Global attention to CLS token
        longformer_mask = longformer_mask.unsqueeze(1).unsqueeze(2) # Match dimension expected by library
    
        # LongFormer with residual connection
        x = x + self.longformer(x, longformer_mask)[0]
        if padding_needed > 0: # Remove unnecessary padding
            x = x[:, :-padding_needed]
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
