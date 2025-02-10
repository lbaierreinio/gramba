import torch.nn as nn
from layers.rnn.MinGRU import MinGRU
from layers.rnn.BiMinGRU import BiMinGRU


class FFN(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.w_1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)
        self.w_2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout_1(self.relu(self.w_1(x)))
        h = self.dropout_2(self.w_2(h))
        return h


class MinGRUBlock(nn.Module):
    """A MinGRU Block (analogous to an attention block in functionality) """
    def __init__(self, hidden_dim, bidirectional=False, dropout=0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.minGRU = BiMinGRU(hidden_dim, hidden_dim) if bidirectional else MinGRU(hidden_dim, hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ffn = FFN(hidden_dim, dropout)

    def forward(self, x, mask=None, h_prev=None):
        if h_prev is not None:
            h = self.minGRU(self.ln_1(x), mask=mask, h_prev=h_prev)
        else:
            h = self.minGRU(self.ln_1(x), mask=mask)
        x = x + h
        x = x + self.ffn(self.ln_2(x))
        if h_prev is not None:
            return x, h
        return x