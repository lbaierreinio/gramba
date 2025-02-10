import torch
from layers.Gramba import Gramba


class TestGramba:
    def test_gramba(self):
        hidden_dim, expansion_factor, bidirectional = 32, 4, True
        gramba = Gramba(hidden_dim, expansion_factor, bidirectional)

        x = torch.randn(16, 10, hidden_dim) # B_S, S_L, H_D

        x_out = gramba(x)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out))
        # assert the output shape is correct
        assert x_out.shape == x.shape

