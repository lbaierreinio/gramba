#Code from https://github.com/allenai/longformer/tree/74523f7f2897b3a5ffcdb537f67a03f83aa1affb
import torch
import torch.nn.functional as F
from typing import Union
from functools import lru_cache

import torch
import os.path

def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x

def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)

def _get_invalid_locations_mask_fixed_dilation(seq_len: int, w: int, d: int):
    diagonals_list = []
    for j in range(-d * w, d, d):
        diagonal_mask = torch.zeros(seq_len, device='cpu', dtype=torch.uint8)
        diagonal_mask[:-j] = 1
        diagonals_list.append(diagonal_mask)
    return torch.stack(diagonals_list, dim=-1)

@lru_cache()
def _get_invalid_locations_mask(w: int, d: Union[torch.Tensor,int], autoregressive: bool, device: str):
    if isinstance(d, int):
        affected_seq_len = w * d
        mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
        mask = mask[None, :, None, :]
    else:
        affected_seq_len = w * d.max()
        head_masks = []
        d_list = d.cpu().numpy().tolist()
        for d in d_list:
            one_head_mask = _get_invalid_locations_mask_fixed_dilation(affected_seq_len, w, d)
            head_masks.append(one_head_mask)
        mask = torch.stack(head_masks, dim=-2)
        mask = mask[None, :, :, :]

    ending_mask = None if autoregressive else mask.flip(dims=(1, 3)).bool().to(device)
    return affected_seq_len, mask.bool().to(device), ending_mask

def mask_invalid_locations(input_tensor: torch.Tensor, w: int, d: Union[torch.Tensor, int], autoregressive: bool) -> torch.Tensor:
    affected_seq_len, beginning_mask, ending_mask = _get_invalid_locations_mask(w, d, autoregressive, input_tensor.device)
    seq_len = input_tensor.size(1)
    beginning_input = input_tensor[:, :affected_seq_len, :, :w+1]
    beginning_mask = beginning_mask[:, :seq_len].expand(beginning_input.size())
    beginning_input.masked_fill_(beginning_mask, -float('inf'))
    if not autoregressive:
        ending_input = input_tensor[:, -affected_seq_len:, :, -(w+1):]
        ending_mask = ending_mask[:, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))


def windowed_attention(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    chunk_q = _chunk(q, w)
    chunk_k = _chunk(k, w)

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    mask_invalid_locations(diagonal_attn, w, 1, False)
    return diagonal_attn