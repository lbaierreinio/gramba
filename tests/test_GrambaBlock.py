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
    
    def test_gramba_block_modes(self):
        hidden_dim, expansion_factor, bidirectional = 32, 2, False
        gramba_block = GrambaBlock(hidden_dim, expansion_factor, bidirectional)

        x = torch.randn(16, 10, hidden_dim)
        mask = torch.randint(0, 2, (16, 10)).bool()

        x_out_p = gramba_block(x, mask, is_sequential = False)
        x_out_s = gramba_block(x, mask, is_sequential = True)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out_p))
        assert not torch.any(torch.isnan(x_out_s))

        # assert the output shape is correct
        assert x_out_p.shape == x.shape
        assert x_out_s.shape == x.shape

        # assert parallel and sequential yield same results
        assert torch.allclose(x_out_p, x_out_s)

