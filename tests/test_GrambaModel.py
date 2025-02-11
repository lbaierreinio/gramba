import torch
from models.GrambaModel import GrambaModel


class TestGrambaModel:
    def test_gramba_model(self):
        hidden_dim, vocab_size, num_layers, window_size, pad_token_id = 32, 100, 2, 512, 0
        gramba_model = GrambaModel(hidden_dim, vocab_size, num_layers, window_size, pad_token_id)

        x = torch.randint(0, vocab_size, (16, 10))

        x_out = gramba_model(x)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out))

        # assert the output shape is correct
        assert x_out.shape == (16, 10, hidden_dim)