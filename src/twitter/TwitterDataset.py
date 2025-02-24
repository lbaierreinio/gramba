from torch.utils.data import Dataset
import torch

class TwitterDataset(Dataset):
    def __init__(self, sequences, labels, masks):
        self.embedding = sequences
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.masks = masks

    def __len__(self):
        return len(self.labels)

    def update_masks(self, masks):
        self.masks = masks

    def __getitem__(self, idx):
        return {
            'input_ids' : self.embedding[idx],
            'attention_mask': self.masks[idx],
            'labels': self.labels[idx]
        }