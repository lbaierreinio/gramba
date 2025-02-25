import torch
import numpy as np
from torch.utils.data import DataLoader

dataset = torch.load('src/imdb/IMDBDataset.pt')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

for batch in train_dataloader:
    for i in range(5):
        print(batch['input_ids'][i].shape, batch['input_ids'][i].dtype, batch['input_ids'][i])
        print(batch['attention_mask'][i].shape, batch['attention_mask'][i].dtype, batch['attention_mask'][i])
        print(batch['labels'][i].shape, batch['labels'][i].dtype, batch['labels'][i])
    break