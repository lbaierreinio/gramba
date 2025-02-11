import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel

dataset = torch.load('imdb_dataset.pt')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

batch = next(iter(train_dataloader))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GrambaSequenceClassificationModel(hidden_dim=256, vocab_size=tokenizer.vocab_size, num_layers=2, window_size=8, ratio=3, pad_token_id=0, num_classes=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

inputs = batch['input_ids'].to(device)
labels = batch['labels'].to(device)
mask = ~batch['attention_mask'].bool().to(device)


model.train()
for i in range(100):
    optimizer.zero_grad()
    logits = model(inputs, mask)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {i}, loss {loss.item()}")

