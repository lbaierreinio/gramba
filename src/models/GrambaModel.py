import torch.nn as nn
from layers.GrambaBlock import GrambaBlock
from layers.HFLongFormerSelfAttentionBlock import HFLongFormerSelfAttentionBlock

class GrambaModel(nn.Module):
    def __init__(self, config):
        """
        Initialize the Gramba model. Note that if the ratio is 0, 
        the attention mechanism will be omitted, and the model will
        consist of num_layers of GrambaBlocks only. If embedding weights 
        are provided, the model will use them to initialize the embedding
        layer. If bidirectional is True, the model will use bidirectional
        GRU layers. Note that sequential mode does not support bidirectionality.
        """
        super().__init__()
        attention_mechanisms = ['longformer']
        assert config.attention_mechanism in ['longformer'], f"Attention mechanism must be one of {attention_mechanisms}"
        if config.embedding_weights is None:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.pad_token_id, _weight=config.embedding_weights, _freeze=True)
        self.layers = nn.ModuleList()

        if config.ratio == 0:
            for _ in range(config.num_layers):
                self.layers.append(GrambaBlock(config.embedding_dim, config.expansion_factor, config.bidirectional))
        else:
            for _ in range(config.num_layers):
                for _ in range(config.ratio):
                    self.layers.append(GrambaBlock(config.embedding_dim, config.expansion_factor, config.bidirectional))
                if config.attention_mechanism == 'longformer':
                    self.layers.append(HFLongFormerSelfAttentionBlock(config.embedding_dim, config.window_size, config.pad_token_id))
                # TODO: Add different attention mechanisms here

    def forward(self, x, attention_mask=None, longformer_mask=None, is_sequential=False):
        x = self.embedding(x)

        for layer in self.layers:
            if isinstance(layer, HFLongFormerSelfAttentionBlock):
                x = layer(x, longformer_mask, is_sequential=is_sequential)
            elif isinstance(layer, GrambaBlock):
                x = layer(x, attention_mask, is_sequential=is_sequential)
        
        return x