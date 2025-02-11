import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sparse_coo_tensor as coo_tensor

def create_sparse_attention_mask(seq_len, window_size, device):
    """Creates a sparse mask for local windowed attention."""
    indices = []
    values = []
    for i in range(seq_len):
        for j in range(max(0, i - window_size), min(seq_len, i + window_size + 1)):
            indices.append([i, j])
            values.append(1.0)
    
    indices = torch.tensor(indices, dtype=torch.long, device=device).t()
    values = torch.tensor(values, dtype=torch.float, device=device)
    sparse_mask = coo_tensor(indices, values, size=(seq_len, seq_len))
    return sparse_mask

class SparseWindowedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        device = x.device
        
        # Project input to queries, keys, values
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  # Split into Q, K, V
        
        # Reshape for multi-head attention
        q = q.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Create sparse mask
        sparse_mask = create_sparse_attention_mask(seq_len, self.window_size, device)
        
        # Compute scaled dot-product attention
        attn_scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / (self.head_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(sparse_mask.to_dense() == 0, float('-inf'))
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.einsum("bhqk, bhkd -> bhqd", attn_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        
        return self.output_proj(output)

# Example usage
batch_size, seq_len, embed_dim, num_heads, window_size = 2, 16, 64, 4, 2
x = torch.randn(batch_size, seq_len, embed_dim)
attn_layer = SparseWindowedAttention(embed_dim, num_heads, window_size)
output = attn_layer(x)
print(output.shape)  # Should output: (batch_size, seq_len, embed_dim)