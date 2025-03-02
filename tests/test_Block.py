import torch
from layers.Block import Block
from layers.Gramba import Gramba
from layers.HFLongFormerSelfAttention import HFLongFormerSelfAttention
from models.GrambaConfig import GrambaConfig



class TestBlock:
    def test_block_with_gramba(self):
        config = GrambaConfig()
        
        g = Gramba(config.embedding_dim, config.expansion_factor, config.bidirectional)
        b = Block(g, config.embedding_dim, config.expansion_factor, config.dropout)

        x = torch.randn(2, 4, config.embedding_dim)
        mask = torch.zeros(2, 4).bool()
        out = b(x, mask)

        assert out.shape == x.shape
        assert not torch.any(torch.isnan(out))


    def test_block_with_HFLongFormerSelfAttention(self):
        config = GrambaConfig()
        
        l = HFLongFormerSelfAttention(config.embedding_dim, config.window_size, config.pad_token_id)
        b = Block(l, config.embedding_dim, config.expansion_factor, config.dropout)

        x = torch.randn(2, 4, config.embedding_dim)
        mask = torch.zeros(2, 4).bool()
        out = b(x, mask)

        assert out.shape == x.shape
        assert not torch.any(torch.isnan(out))
   
