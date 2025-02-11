import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel
from transformers import get_cosine_schedule_with_warmup

dataset = torch.load('imdb_dataset.pt')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

batch = next(iter(train_dataloader))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hidden_dim = 256
vocab_size = tokenizer.vocab_size
num_layers = 2
window_size = 8
ratio = 3
pad_token_id = 0
num_classes = 2
bidirectional = False
expansion_factor = 4

model = GrambaSequenceClassificationModel(hidden_dim, vocab_size, num_layers, window_size, pad_token_id, ratio=ratio, expansion_factor=expansion_factor, bidirectional=False, num_classes=num_classes).to(device)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

print("---Model Details---")
print(f"Hidden Dim: {hidden_dim}")
print(f"Vocab Size: {vocab_size}")
print(f"Num Layers: {num_layers}")
print(f"Window Size: {window_size}")
print(f"Ratio: {ratio}")
print(f"Num Classes: {num_classes}")
print(f"Expansion Factor: {expansion_factor}")
print(f"Bidirectional: {bidirectional}")
print(f"Num Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_training_steps = 100
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)
loss_fn = torch.nn.CrossEntropyLoss()

inputs = batch['input_ids'].to(device)
labels = batch['labels'].to(device)
mask = ~batch['attention_mask'].bool().to(device)


model.train()
for i in range(num_training_steps):
    optimizer.zero_grad()
    logits = model(inputs, mask)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(f"Epoch {i}, loss {loss.item()}")

