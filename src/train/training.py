import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from models.GrambaConfig import GrambaConfig
import torch.optim.lr_scheduler as lr_scheduler
from models.GrambaSequenceClassificationModel import GrambaSequenceClassificationModel

is_twitter = 0
is_save = False
ampere_gpu = True

######## CHECK BEFORE RUNNING ########
if is_twitter: 
    BATCH_SIZE = 512
    dataset = torch.load('src/twitter/twitter.pt')
else:
    BATCH_SIZE = 64
    dataset = torch.load('src/imdb/imdb.pt')

#####################################
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

saving_folder = 'src/train/saving_train'


if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible

model = GrambaSequenceClassificationModel(config).to(device)

print("---Model Details---")
print(f"Hidden Dim: {config.embedding_dim}")
print(f"Vocab Size: {config.vocab_size}")
print(f"Num Layers: {config.num_layers}")
print(f"Window Size: {config.window_size}")
print(f"Ratio: {config.ratio}")
print(f"Expansion Factor: {config.expansion_factor}")
print(f"Bidirectional: {config.bidirectional}")
print(f"Num Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
print(f"Training on {train_size} samples, validating on {val_size} samples")


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
num_training_steps = 100
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=num_training_steps)
loss_fn = torch.nn.BCEWithLogitsLoss()

#load the last existing model if there is one
try:
    #get all the models
    models = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(saving_folder) if f.startswith('model_')]
    #get the last model
    last_model = max(models)
    model.load_state_dict(torch.load(f"{saving_folder}/model_{last_model}.pt",weights_only=True))
    last_model += 1
    print(f"Model {last_model} loaded")
except:
    print("No existing model found")
    last_model = 0

with tqdm(total=num_training_steps-last_model, desc="Training") as pbar1:
    for i in tqdm(range(last_model,num_training_steps)):
        model.train()
        #iter over the training data by batch
        with tqdm(total=len(train_dataloader), leave=False, desc="Iteration over batches") as pbar2:
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
                optimizer.step()
                pbar2.update(1)
                pbar2.set_postfix({"loss": loss.item()})
            scheduler.step()

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

        if is_save:
            #saving the model
            torch.save(model.state_dict(), f"{saving_folder}/model_{i}.pt")
            #saving acc and loss in log file
            with open(f"{saving_folder}/log.txt", "a") as f:
                f.write(f"{i} {val_loss} {val_accuracy}\n")

        #update progress bar
        pbar1.set_postfix({"val_loss": val_loss, "val_accuracy": val_accuracy})
        pbar1.update(1)

    


