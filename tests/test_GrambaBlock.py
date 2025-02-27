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
<<<<<<< HEAD
        assert x_out.shape == x.shape
    
    def test_gramba_block_modes(self):
        hidden_dim, expansion_factor, bidirectional = 32, 2, False
        gramba_block = GrambaBlock(hidden_dim, expansion_factor, bidirectional)

        x = torch.randn(16, 10, hidden_dim)
        # TODO: Fix masking

        x_out_p = gramba_block(x, is_sequential = False)
        x_out_s = gramba_block(x, is_sequential = True)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out_p))
        assert not torch.any(torch.isnan(x_out_s))

        # assert the output shape is correct
        assert x_out_p.shape == x.shape
        assert x_out_s.shape == x.shape

        # assert parallel and sequential yield same results
        assert torch.allclose(x_out_s, x_out_p, rtol=1e-4, atol=1e-6)


=======
        assert x_out.shape == x.shape
>>>>>>> main
