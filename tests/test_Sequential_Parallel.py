import torch
from layers.Gramba import Gramba


class TestSequentialParallel:
    def test_gramba_sequential_parallel(self):
        hidden_dim, expansion_factor, bidirectional = 32, 4, False
        gramba = Gramba(hidden_dim, expansion_factor, bidirectional)

        x = torch.randn(16, 10, hidden_dim) # B_S, S_L, H_D

        x_out_seq = gramba(x, is_sequential=True)
        x_out_par = gramba(x, is_sequential=False)
        
        # assert none are nan
        assert not torch.any(torch.isnan(x_out_seq))
        assert not torch.any(torch.isnan(x_out_par))
        # assert the output shape is correct
        assert x_out_seq.shape == x.shape
        assert x_out_par.shape == x.shape

        # assert outputs are the same
        assert torch.allclose(x_out_seq, x_out_par, rtol=1e-4, atol=1e-6)

