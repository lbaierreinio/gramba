import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data
file_path = "train_log_twitter.txt"  # Change this if your file has a different name

df = pd.read_csv(file_path)

# Compute statistics
avg_tokens_per_s = df['tokens/s'].mean()
avg_epoch_time = df['epoch_time'].mean()
lowest_val_loss = df['val_loss'].min()
highest_val_accuracy = df['val_accuracy'].max()

# Print statistics
print(f"Average Tokens/s: {avg_tokens_per_s:.2f}")
print(f"Average Epoch Time: {avg_epoch_time:.2f} seconds")
print(f"Lowest Validation Loss: {lowest_val_loss:.4f}")
print(f"Highest Validation Accuracy: {highest_val_accuracy:.2f}")

# Plot Train Loss vs. Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend(loc='best')
plt.grid()
plt.savefig("train_val_loss.png", bbox_inches='tight')  # Save the figure with tight layout
plt.show()