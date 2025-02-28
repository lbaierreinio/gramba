import time
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel
import torch.optim.lr_scheduler as lr_scheduler
from transformers import BertTokenizer

is_twitter = 0
is_save = False
is_load = False
is_log = True
ampere_gpu = True

if is_twitter: 
    BATCH_SIZE = 512
    dataset = torch.load('src/twitter/twitter.pt')
else:
    BATCH_SIZE = 64
    dataset = torch.load('src/imdb/imdb.pt')

hidden_dim = 50

embedding_matrix = torch.tensor(np.load('src/glove/embedding_matrix.npy'), dtype=torch.float32)
vocab_size = BertTokenizer.from_pretrained('bert-base-uncased').vocab_size
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 1
window_size = 8
ratio = 4
pad_token_id = 0
bidirectional = False
expansion_factor = 1
num_training_steps = 10

if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible

model = GrambaSequenceClassificationModel(hidden_dim, vocab_size, num_layers, window_size, pad_token_id, ratio=ratio, embedding_weights=embedding_matrix, expansion_factor=expansion_factor, bidirectional=bidirectional).to(device)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

print("---Model Details---")
print(f"Hidden Dim: {hidden_dim}")
print(f"Vocab Size: {vocab_size}")
print(f"Num Layers: {num_layers}")
print(f"Window Size: {window_size}")
print(f"Ratio: {ratio}")
print(f"Expansion Factor: {expansion_factor}")
print(f"Bidirectional: {bidirectional}")
print(f"Num Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Training on {train_size} samples, validating on {val_size} samples")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=num_training_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()

with open("train_log.txt", "w") as file:
    file.write(f"# expansion_factor={expansion_factor} hidden_dim={hidden_dim} num_layers={num_layers} ratio={ratio} window_size={window_size} bidirectional={bidirectional} num_training_steps={num_training_steps}\n")
    file.write("epoch,train_loss,val_loss,val_accuracy,epoch_time,tokens/s\n")


for i in range(num_training_steps):
    model.train()
    #iter over the training data by batch
    with tqdm(total=len(train_dataloader), leave=False, desc=f"Epoch {i}") as pbar:
        train_loss = 0
        tokens_processed = 0
        t0 = time.time()
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].float().to(device)
            #add labels at the end of inputs
            mask = ~batch['attention_mask'].bool().to(device)

            #forward pass
            logits = model(inputs, mask)
            logits = logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            train_loss += loss.item()
            tokens_processed += inputs.size(0) * inputs.size(1) # (b_s * s_l)
            optimizer.step()
            pbar.update(1)
        scheduler.step()
        if torch.cuda.is_available():
            # wait for all cuda processes to finish to get accurate timing
            torch.cuda.synchronize()
        t1 = time.time()

        train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].float().to(device)
            mask = ~batch['attention_mask'].bool().to(device)
            logits = model(inputs, mask)
            logits = logits.squeeze(-1)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            #sigmoid activation function
            logits = torch.sigmoid(logits)
            preds = (logits > 0.5).float()
            val_accuracy += (preds == labels).float().sum().item()
    val_loss /= len(val_dataloader)
    val_accuracy /= 5000

    print(f"epoch: {i} train_loss: {round(train_loss, 2)}, val_loss: {round(val_loss, 2)}, val_accuracy: {round(val_accuracy, 2)} tokens/s: {round(tokens_processed/(t1-t0), 2)}")
    with open("train_log.txt", "a") as file:
        file.write(f"{i},{round(train_loss, 4)},{round(val_loss, 4)},{round(val_accuracy,2)},{t1-t0},{round(tokens_processed/(t1-t0), 2)}\n")



