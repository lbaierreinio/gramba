import torch
from models.GrambaModel import GrambaModel


class TestGrambaModel:
    def test_gramba_model(self):
        hidden_dim, vocab_size, num_layers, window_size, pad_token_id = 32, 100, 2, 2, 0
        gramba_model = GrambaModel(hidden_dim, vocab_size, num_layers, window_size, pad_token_id)

        x = torch.randint(0, vocab_size, (32, 512))

        mask = torch.ones_like(x).bool()

        x_out = gramba_model(x, mask)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out))

        # assert the output shape is correct
        assert x_out.shape == (32, 512, hidden_dim)