import torch.nn as nn
from layers.GrambaBlock import GrambaBlock
from layers.SWABlock import SWABlock

class GrambaModel(nn.Module):
    def __init__(self, hidden_dim, vocab_size, num_layers, ratio = 2, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            for _ in range(ratio):
                self.layers.append(GrambaBlock(hidden_dim, expansion_factor, bidirectional))
            self.layers.append(SWABlock(hidden_dim, expansion_factor))

    def forward(self, x, mask=None):
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, mask)
        
        return x