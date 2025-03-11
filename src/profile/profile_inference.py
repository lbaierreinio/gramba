import time
import torch
import numpy as np
from models.GrambaConfig import GrambaConfig
from transformers import AutoTokenizer
import torch.profiler
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_path = "src/glove/embedding_matrix_50.npy"
log_file = "profile_inference.txt"
batch_size = 32

config = GrambaConfig(
    num_classes=2,
    vocab_size=tokenizer.vocab_size,
    embedding_weights=torch.tensor(np.load(embedding_path), dtype=torch.float32),
    embedding_dim=50,
    expansion_factor=2,
    num_layers=2,
    window_size=32,
    ratio=3,
    bidirectional=False,
    attention_mechanism='longformer',
    pad_token_id=tokenizer.pad_token_id
)
model = GrambaSequenceClassificationModel(config).cuda()
model.eval()

with open(log_file, "w") as file:
    file.write(f"# num_classes={config.num_classes} vocab_size={config.vocab_size} embedding_dim={config.embedding_dim} expansion_factor={config.expansion_factor} num_layers={config.num_layers} attention_mechanism={config.attention_mechanism} ratio={config.ratio} bidirectional={config.bidirectional} pad_token_id={config.pad_token_id}\n")
    # print device
    file.write(f"# device={torch.cuda.get_device_name(torch.cuda.current_device())}\n")
    # print model parameters
    file.write(f"# parameters:={(sum(p.numel() for p in model.parameters()))}\n")
    # print batch size
    file.write(f"# batch_size={batch_size}\n")
    # print sequence length
    # print headers to log file
    file.write("sequence_length,inference_time,inference_memory\n")

for sequence_length in range(32, 196610, 64):
   # try:
    inference_memory = 0
    inference_time = 0
    batch = torch.randint(0, config.vocab_size, (batch_size, sequence_length)).cuda()
    labels = torch.zeros(batch_size).long().cuda()
    # Local attention mask
    longformer_mask = torch.zeros(batch_size, sequence_length).cuda()

    for i in range(5): # Warmup period
        model(batch, longformer_mask=longformer_mask)
    
    torch.cuda.reset_peak_memory_stats() 
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
    ) as prof:
        cur_max_memory = 0
        model(batch, longformer_mask=longformer_mask)
        inference_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    for i in range(5):
        t0 = time.time()
        model(batch, longformer_mask=longformer_mask)
        torch.cuda.synchronize()
        inference_time += time.time() - t0
    
    inference_time /= 5

    with open(log_file, "a") as file:
        file.write(f"{sequence_length},{inference_time},{inference_memory}\n")

   # except Exception as e:
        # print exception
     #   print(e)