import torch
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

        self.minGRU = BiMinGRU(self.expansion_dim, self.expansion_dim) if bidirectional else MinGRU(self.expansion_dim, self.expansion_dim)
        self.linear_out = nn.Linear(self.expansion_dim, hidden_dim)

    def forward(self, x, mask=None, is_sequential=False):
        assert not (is_sequential and mask is not None), "Cannot use mask and is_sequential at the same time"
        assert not (is_sequential and self.minGRU.bidirectional), "Cannot use bidirectional GRU in sequential mode"
        x_in = self.projection_one(x)
        x_skip = self.projection_two(x)
        if is_sequential:
            h_prev = torch.zeros((x_in.shape[0], x_in.shape[2])).to(x_in.device) # Initial hidden state
            for t in range(x.shape[1]): # Iterate over sequence length dimension
                h_prev = self.minGRU(x_in[:, t], h_prev=h_prev)
                x_in[:, t] = h_prev
        else: 
            x_in = self.minGRU(x_in, mask=mask)
        
        x_out = self.linear_out(x_skip + x_in)

        return x_out