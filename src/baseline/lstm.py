import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

num_training_steps = 10

class TextClassificationModel(nn.Module):
    def __init__(self):
        super(TextClassificationModel, self).__init__()
        
        self.spatial_dropout = nn.Dropout2d(0.2)  # Equivalent to SpatialDropout1D
        self.conv1d = nn.Conv1d(in_channels=50, out_channels=64, kernel_size=5)
        self.lstm = nn.LSTM(64, 64, bidirectional=True, dropout=0.2, batch_first=True)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)
        
    def forward(self, x, mask):
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = x.permute(0, 2, 1)  # Conv1D expects (batch_size, embedding_dim, seq_length)
        x = self.spatial_dropout(x.unsqueeze(2)).squeeze(2)  # Equivalent to SpatialDropout1D
        x = F.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)  # LSTM expects (batch_size, seq_length, features)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
model = TextClassificationModel()


# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Define loss function
criterion = nn.BCELoss()  # Binary Cross Entropy Loss

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=0.01, verbose=True)

dataset = torch.load('src/twitter/twitter.pt')
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

loss_fn = nn.BCELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with tqdm(total=num_training_steps, desc="Training") as pbar1:
    for i in tqdm(range(num_training_steps)):
        model.train()
        #iter over the training data by batch
        with tqdm(total=len(train_dataloader), leave=False, desc="Iteration over batches") as pbar2:
            for batch in train_dataloader:
                optimizer.zero_grad()
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                #add labels at the end of inputs
                mask = ~batch['attention_mask'].bool().to(device)

                #forward pass
                logits = model(inputs,mask)
                logits = logits.squeeze(-1)
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                pbar2.update(1)
                pbar2.set_postfix({"loss": loss.item()})
            scheduler.step(metrics=loss)


        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
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

        #update progress bar
        pbar1.set_postfix({"val_loss": val_loss, "val_accuracy": val_accuracy})
        pbar1.update(1)

    


