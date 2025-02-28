from torch.utils.data import Dataset
import torch

class TwitterDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer):
        self.encodings = tokenizer(
            sequences,
            padding=True,
            add_special_tokens=True,
            truncation=False,
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }