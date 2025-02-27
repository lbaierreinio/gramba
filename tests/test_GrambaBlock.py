import torch
from layers.GrambaBlock import GrambaBlock


class TestGrambaBlock:
    def test_gramba_block(self):
        hidden_dim, expansion_factor, bidirectional = 32, 4, True
        gramba_block = GrambaBlock(hidden_dim, expansion_factor, bidirectional)

        x = torch.randn(16, 10, hidden_dim) # B_S, S_L, H_D

        x_out = gramba_block(x)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out))

        # assert the output shape is correct
        assert x_out.shape == x.shape