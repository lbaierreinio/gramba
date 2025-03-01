import torch.nn as nn
import torch.nn.functional as F
from transformers import LongformerConfig
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention

class HFLongFormerSelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, window_size, pad_token_id, task, expansion_factor=4, num_attention_heads=2):
        """
        NOTE: This implementation requires a later version of Transformers (e.g. 4.46.3)
        """
        super().__init__()
        assert window_size > 2, "Window size must be greater than 2"
        self.pad_token_id = pad_token_id
        self.window_size = window_size
        self.task = task

        # Create an object for LongFormerSelfAttention's configuration
        config = LongformerConfig(
            hidden_size = hidden_dim,
            num_attention_heads = num_attention_heads,
            attention_window = [window_size],
            embed_dim = hidden_dim,
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

    def forward(self, x, mask, is_sequential=False, token_type_ids=None):
        """
        Ensure sufficient padding added to adhere to window size.
        """
        padding_needed = self.window_size - (x.size(1) % self.window_size)
        longformer_mask = mask
        if padding_needed > 0:
            x = F.pad(x, (0, 0, 0, padding_needed), value=self.pad_token_id)
            longformer_mask = F.pad(mask, (0, padding_needed), value=True)
        
        longformer_mask = longformer_mask * -10000 # Local attention to all non-padded values

        if self.task == 'cls': # Global attention to CLS token
            longformer_mask[:, -1] = 10000 
        elif self.task == 'qa': # Global attention to question tokens
            # TODO
            print(token_type_ids)
            pass

        is_index_masked = longformer_mask < 0
        is_index_global_attn = longformer_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        # LongFormer with residual connection
        x = x + self.longformer(x, longformer_mask, None, is_index_masked, is_index_global_attn, is_global_attn, False)[0]
        # Remove the padding token
        if padding_needed > 0:
            x = x[:, :-padding_needed]
        x = self.ln(x)

        # MLP 1 with residual connection
        x = x + self.mlp(x)

        return x
