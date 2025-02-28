import time
import torch
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.GrambaConfig import GrambaConfig
import torch.optim.lr_scheduler as lr_scheduler
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel

is_twitter = 0
is_save = False
is_load = False
is_log = True
ampere_gpu = True

if is_twitter: 
    batch_size = 512
    dataset = torch.load('src/twitter/twitter.pt')
else:
    batch_size = 64
    dataset = torch.load('src/imdb/imdb.pt')

split = 0.9
train_size = int(split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


config = GrambaConfig(
    num_classes=1,
    vocab_size=BertTokenizer.from_pretrained('bert-base-uncased').vocab_size,
    embedding_weights=torch.tensor(np.load('src/glove/embedding_matrix.npy'), dtype=torch.float32),
    embedding_dim=50,
    expansion_factor=1,
    num_layers=1,
    window_size=8,
    ratio=4,
    bidirectional=False,
    pad_token_id=0
)

if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GrambaSequenceClassificationModel(config).to(device)
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

num_training_steps = 10
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=num_training_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()

with open("train_log.txt", "w") as file:
    file.write(f"# gpu_name={torch.cuda.get_device_name(torch.cuda.current_device())} dataset_size={len(dataset)} train_split={split} batch_size={batch_size} parameters={parameters} expansion_factor={config.expansion_factor} hidden_dim={config.embedding_dim}\n")
    file.write(f"# num_layers={config.num_layers} ratio={config.ratio} window_size={config.window_size} bidirectional={config.bidirectional} num_training_steps={num_training_steps} vocab_size={config.vocab_size}\n")
    file.write("epoch,train_loss,val_loss,val_accuracy,epoch_time,tokens/s\n")

for i in range(num_training_steps):
    model.train()
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
    val_accuracy /= val_size

    print(f"{i},{round(train_loss, 4)},{round(val_loss, 4)},{round(val_accuracy,2)},{t1-t0},{round(tokens_processed/(t1-t0), 2)}\n")

    with open("train_log.txt", "a") as file:
        file.write(f"{i},{round(train_loss, 4)},{round(val_loss, 4)},{round(val_accuracy,2)},{round(t1-t0,2)},{round(tokens_processed/(t1-t0), 2)}\n")