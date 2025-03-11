import torch


class GrambaConfig:
    """
    Configuration class for Gramba. Descriptions:
    - embedding_dim: The dimension of the embeddings.
    - expansion_factor: The expansion factor of the minGRU layers and MLPs.
    - num_layers: The number of layers in the model.
    - num_classes: The number of classes in the classification task.
    - ratio: The ratio of minGRU to attention layers in the model (e.g. 2 means 2 minGRU layers for every 1 attention layer).
    - window_size: The window size for the attention mechanism.
    - dropout: The dropout rate.
    - bidirectional: Whether the minGRU layers are bidirectional.
    - pad_token_id: The token ID for padding.
    - vocab_size: The size of the vocabulary.
    - embedding_weights: The weights for the embeddings (if not provided the embedding weights are initialized from scratch.).

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