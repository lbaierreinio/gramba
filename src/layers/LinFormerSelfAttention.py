import torch.nn as nn
import torch.nn.functional as F
from linformer import LinformerSelfAttention

class LinFormerSelfAttention(nn.Module):
    def __init__(self, embedding_dim, pad_token_id, dropout):
        """
        NOTE: This implementation requires a later version of Transformers (e.g. 4.46.3)
        """
        super().__init__()
        self.pad_token_id = pad_token_id

        self.linformer = LinformerSelfAttention(embedding_dim, seq_len = 870, dropout=dropout, heads = 2)
    
    def forward(self, x, mask, is_sequential=None):
        """
        Apply mask to ignore padding tokens
        """
        # Ensure padding so that the input size is a multiple of the window size
        #add mask to ignore padding tokens before passing to linformer as it does not have mask parameter
        x = x.masked_fill(mask.unsqueeze(-1), self.pad_token_id)
        x = self.linformer(x)
        return x