import torch.nn as nn
import torch.nn.functional as F
from models.GrambaModel import GrambaModel

class GrambaSQuADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gramba_model = GrambaModel(config)
        self.head = nn.Linear(config.embedding_dim, config.num_classes)
        
    def forward(self, x, targets=None, mask=None):
        x = self.gramba_model(x, mask)
        logits = self.head(x)

        loss = self.loss(logits, targets, mask) if targets is not None else None
        return logits, loss

    def loss(self, logits, targets, mask=None):
        """Sum of cross entropy of start and end positions, weighed equally.

        Inputs:
        - logits [B, T, 2]: each [T, 2] item is a pair of logits for the start and end position for each token in the sequence
        - targets: [B, 2]: the correct start and end positions for each example
        - mask (optional): torch.Tensor [batch_size, seq_len]
        
        NOTE: mask is True for positions that should be masked out, False everywhere else
        """
        start_pos_logits = logits[:,:,0]
        start_pos_targets = targets[:,0]
        end_pos_logits = logits[:,:,1]
        end_pos_targets = targets[:,1]

        if mask is not None:
            # Set the logits for the masked positions to -inf. This will ensure that
            # they have no contribution in the cross entropy loss, since they will
            # be fed into a softmax first.
            start_pos_logits = start_pos_logits.masked_fill(mask, float("-inf"))
            end_pos_logits = end_pos_logits.masked_fill(mask, float("-inf"))
    
        return F.cross_entropy(start_pos_logits, start_pos_targets) + F.cross_entropy(end_pos_logits, end_pos_targets)