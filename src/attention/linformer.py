from linformer_pytorch import LinearAttentionHead, get_EF
import torch



E_learn = get_EF(1000, 100)
F_learn = get_EF(100, 1000)

model = LinearAttentionHead(
        dim=64, # Dim 2 of the input
        dropout=0.1, # Dropout of the P matrix
        E_proj=E_learn, F_proh=F_learn, # The E and F layers
        full_attention=False, # Use Full Attention instead
        )
x = torch.randn(1, 512, 64)
y = model(x, x, x)
print(y) # (1, 512, 64)



##Multihead Attention
from linformer_pytorch import MHAttention
import torch

model = MHAttention(
        input_size=512, # Dimension 1 of the input
        channels=64, # Dimension 2 of the input
        dim=8, # Dim of each attn head
        dim_k=128, # What to sample the input length down to
        nhead=8, # Number of heads
        dropout=0, # Dropout for each of the heads
        activation="gelu", # Activation after attention has been concat'd
        checkpoint_level="C2", # If C2, checkpoint each of the heads
        parameter_sharing="layerwise", # What level of parameter sharing to do
        E_proj=E_learn, F_proh=F_learn, # The E and F layers
        full_attention=False, # Use full attention instead
        w_o_intermediate_dim=None, # If not None, have 2 w_o matrices, such that instead of `dim*nead,channels`, you have `dim*nhead,w_o_int`, and `w_o_int,channels`
        )
x = torch.randn(1, 512, 64)
y = model(x)
print(y) # (1, 512, 64)