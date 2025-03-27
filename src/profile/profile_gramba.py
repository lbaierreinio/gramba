import time
import torch
import numpy as np
from models.GrambaConfig import GrambaConfig
from transformers import AutoTokenizer
import torch.profiler
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel

# Profile for train time

# Keep minGRU layers fixed, increase sliding windows
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_path = "src/glove/embedding_matrix_50.npy"
log_file = "profile_log.txt"
batch_size = 32
sequence_length = 2048
config = GrambaConfig(
        num_classes=2,
        vocab_size=tokenizer.vocab_size,
        embedding_weights=torch.tensor(np.load(embedding_path), dtype=torch.float32),
        embedding_dim=50,
        expansion_factor=2,
        num_layers=2,
        attention_mechanism='longformer',
        ratio=3,
        bidirectional=False,
        pad_token_id=tokenizer.pad_token_id
    )

# Fake batch for profiling
batch = torch.randint(0, config.vocab_size, (batch_size, sequence_length)).cuda()
labels = torch.zeros(batch_size).long().cuda()
# No padding
attention_mask = torch.ones(batch_size, sequence_length).bool().cuda()
# Local attention mask
longformer_mask = torch.zeros(batch_size, sequence_length).cuda()

# Write model parameters to log file
with open(log_file, "w") as file:
    file.write(f"# num_classes={config.num_classes} vocab_size={config.vocab_size} embedding_dim={config.embedding_dim} expansion_factor={config.expansion_factor} num_layers={config.num_layers} attention_mechanism={config.attention_mechanism} ratio={config.ratio} bidirectional={config.bidirectional} pad_token_id={config.pad_token_id}\n")
    file.write("# ratio fixed")
    # print device
    file.write(f"# device={torch.cuda.get_device_name(torch.cuda.current_device())}\n")
    # print model parameters
    # print batch size
    file.write(f"# batch_size={batch_size}\n")
    # print sequence length
    file.write(f"# sequence_length={sequence_length}\n")
    # print headers to log file
    file.write("window_size,parameters,train_time,train_memory,inference_time,inference_memory\n")


def train(model, batch, attention_mask, longformer_mask, labels, optimizer):
    logits = model(batch, attention_mask, longformer_mask)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
 
# Window size from 32 to 2048
for window_size in range(32, 2050, 32):
    try:
        config.window_size = window_size
        model = GrambaSequenceClassificationModel(config)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_memory = 0
        train_time = 0
        inference_memory = 0
        inference_time = 0

        for i in range(5): # Warmup period
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
        
        torch.cuda.reset_peak_memory_stats() 
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            cur_max_memory = 0
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
            train_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        for i in range(5):
            t0 = time.time()
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
            torch.cuda.synchronize()
            train_time += time.time() - t0
        
        train_time /= 5

        with open(log_file, "a") as file:
            file.write(f"{window_size},{sum(p.numel() for p in model.parameters())},{train_time},{train_memory},{inference_time},{inference_memory}\n")

    except Exception as e:
        break
    

# Keep window size fixed, increase minGRU layers
with open(log_file, "a") as file:
    file.write("# window size fixed\n\n")
    file.write("ratio,parameters,train_time,train_memory,inference_time,inference_memory\n")
config.window_size = 32
for ratio in range(2, 20):
    try:
        config.ratio = ratio
        model = GrambaSequenceClassificationModel(config)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_memory = 0
        train_time = 0
        inference_memory = 0
        inference_time = 0

        for i in range(5): # Warmup period
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
        
        torch.cuda.reset_peak_memory_stats() 
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            cur_max_memory = 0
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
            train_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        for i in range(5):
            t0 = time.time()
            train(model, batch, attention_mask, longformer_mask, labels, optimizer)
            torch.cuda.synchronize()
            train_time += time.time() - t0
        
        train_time /= 5

        with open(log_file, "a") as file:
            file.write(f"{ratio},{sum(p.numel() for p in model.parameters())},{train_time},{train_memory},{inference_time},{inference_memory}\n")
    except Exception as e:
        break