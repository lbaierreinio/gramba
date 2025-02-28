import torch
from models.GrambaModel import GrambaModel
from models.GrambaConfig import GrambaConfig


class TestGrambaModel:
    def test_gramba_model(self):
        config = GrambaConfig(vocab_size=100)
        gramba_model = GrambaModel(config)

        x = torch.randint(0, config.vocab_size, (32, 512))

        mask = torch.ones_like(x).bool()

        x_out = gramba_model(x, mask)

        # assert none are nan
        assert not torch.any(torch.isnan(x_out))

        # assert the output shape is correct
        assert x_out.shape == (32, 512, config.embedding_dim)