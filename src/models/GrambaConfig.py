import torch


class GrambaConfig:
    """
    Configuration class for Gramba.
    """
    def __init__(
            self,
            embedding_dim: int = 50,
            expansion_factor: int = 1,
            num_layers: int = 4,
            num_classes: int = 2,
            ratio: int = 6,
            window_size: int = 32,
            dropout: float = 0.3,
            bidirectional: bool = True,
            pad_token_id: int = 0,
            vocab_size: int = 30522,
            embedding_weights: torch.Tensor = None,
            attention_mechanism: str = 'longformer',
    ):
        """
        Initialize the configuration class for Gramba.
        """
        self.embedding_dim = embedding_dim
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.ratio = ratio
        self.window_size = window_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.embedding_weights = embedding_weights
        self.attention_mechanism = attention_mechanism