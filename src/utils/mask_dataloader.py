import torch

def classification_collate_fn(batch):
    # Extract attention_mask from each sample in the batch
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    # Multiply the attention_mask by -10000
    longformer_mask = attention_mask * -10000

    # Set the last value in the attention_mask to 10000
    longformer_mask[:, -1] = 10000

    # Stack input_ids and labels as usual
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # Return the processed batch with the modified attention_mask
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'longformer_mask': longformer_mask,
    }
 