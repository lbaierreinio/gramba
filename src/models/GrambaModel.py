import torch.nn as nn
from layers.GrambaBlock import GrambaBlock
from layers.SWABlock import SWABlock

class GrambaModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, embedding_weights, num_layers, window_size, pad_token_id=0, attention_probs_dropout_prob=0.3, ratio=2, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id, _weight=embedding_weights, _freeze=True)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            for _ in range(ratio):
                self.layers.append(GrambaBlock(embedding_dim, expansion_factor, bidirectional))
            self.layers.append(SWABlock(embedding_dim, window_size, pad_token_id))

    def forward(self, x, mask=None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        return x