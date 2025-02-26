import torch.nn as nn
from layers.MinGRU import MinGRU
from layers.BiMinGRU import BiMinGRU

class Gramba(nn.Module):
    def __init__(self, hidden_dim, expansion_factor=4, bidirectional=False):
        super().__init__()
        self.expansion_dim = hidden_dim * expansion_factor

        self.projection_one = nn.Sequential(
            nn.Linear(hidden_dim, self.expansion_dim),
            nn.SiLU()
        )

        self.projection_two = nn.Sequential(
            nn.Linear(hidden_dim, self.expansion_dim),
            nn.SiLU()
        )

        self.minGRU = MinGRU(self.expansion_dim, self.expansion_dim)
        self.linear_out = nn.Linear(self.expansion_dim, hidden_dim)

    def forward(self, x, mask=None, h_prev=None):
        x_in = self.projection_one(x)
        x_skip = self.projection_two(x)
        if h_prev is not None:
            h = self.minGRU(x_in, mask=mask, h_prev=h_prev)
        else:
            h = self.minGRU(x_in, mask=mask)
        
        x_out = self.linear_out(x_skip + h)
        if h_prev is not None:
            return x_out, h
        return x_out