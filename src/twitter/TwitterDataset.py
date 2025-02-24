from torch.utils.data import Dataset
import torch

class TwitterDataset(Dataset):
    def __init__(self, sequences, labels):
        self.embedding = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'embeddings' : self.embedding[idx],
            'labels': self.labels[idx]
        }