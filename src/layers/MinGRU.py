import torch
import torch.nn as nn
import torch.nn.functional as F


class MinGRU(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        # Linear layer for producing z from x
        self.linear_z = nn.Linear(dim_in, dim_hidden)
        # Linear layer for producing candidate state h_tilde from x
        self.linear_h = nn.Linear(dim_in, dim_hidden)

    def parallel_scan_log(self, log_a, log_b, mask=None):
        """
        Given sequences log(a) and log(b) of length t, compute h[0:t-1],
        where h[0] = b[0], and h[i] = a[i]*h[i-1] + b[i] for i > 0.

        Args:
            log_a: torch.Tensor
            log_b: torch.Tensor

        Returns:
            h: torch.Tensor
        """
        if mask is not None:
            log_a = log_a.masked_fill(mask, 0)
            log_b = log_b.masked_fill(mask, 0)

        # Take cumulative sum across seq_len dimension
        log_a_star = torch.cumsum(log_a, dim=1)

        log_a_b_star = log_b - log_a_star

        if mask is not None:
            log_a_b_star = log_a_b_star.masked_fill(mask, float('-inf'))

        log_x0_plus_b_star = torch.logcumsumexp(log_a_b_star, dim=1)
        
        log_x = log_a_star + log_x0_plus_b_star

        return log_x.exp()

    def g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))

    def log_g(self, x):
        """
        Appendix B.3: Were RNNs All We Needed?
        """
        return torch.where(x >= 0, torch.log(F.relu(x)+0.5), -F.softplus(-x))

    def forward(self, x, *, h_prev=None, mask=None):
        """
        Compute the forward pass. Note that if h_prev is not none,
        then we assume the model is processing tokens sequentially.
        Otherwise, we enter parallel mode. We return the
        output of the RNN and the hidden state if return_hidden is True.

        If mask is provided, mask out the hidden states corresponding to the True positions
        in the mask (this is used to mask out padding tokens in the sequence).

        Args:
            x: torch.Tensor [batch_size, seq_len, dim_in] (parallel), [batch_size, dim_in] (sequential)
            h_prev (optional): torch.Tensor [batch_size, dim_hidden]
            mask (optional): torch.Tensor [batch_size, seq_len]
        Returns:
            h: torch.Tensor [batch_size, seq_len, dim_hidden]
        """
        k = self.linear_z(x)
        tilde_h = self.linear_h(x)  # Candidate state

        if mask is not None:
            mask = mask.unsqueeze(-1)
        if h_prev is not None:  # Sequential mode
            z = torch.sigmoid(k)
            tilde_h = self.g(tilde_h)
            h = ((1 - z) * h_prev) + (z * tilde_h)
            del z, tilde_h, k
        else:  # Parallel Mode
            # NOTE: the implementation provided in the paper allows providing an explicit
            #       starting state h_0; we fix h_0 (implicitly) to be zero initialized
            log_z = -F.softplus(-k)  # Log (z)
            log_one_minus_z = -F.softplus(k)  # Log (1 - z)
            log_tilde_h = self.log_g(tilde_h)  # Log candidate state
  
            h = self.parallel_scan_log(
                log_one_minus_z, log_z + log_tilde_h, mask)  # Hidden states
        if mask is not None:
            h = h.masked_fill(mask, 0)
        return h
