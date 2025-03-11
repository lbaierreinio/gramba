import pandas as pd
import matplotlib.pyplot as plt

# Read the data from a text file
data = pd.read_csv("profile_seq_len.txt")

# Plot training time vs window size
plt.figure(figsize=(10, 5))
plt.plot(data["sequence_length"], data["train_time"], marker="o", label="Training Time (s)")
plt.xlabel("Window Size")
plt.ylabel("Training Time (s)")
plt.title("Training Time vs Window Size")
plt.legend()
plt.grid()
plt.savefig("training_time_vs_sequence_length.png")
plt.show()

# Plot training memory vs window size
plt.figure(figsize=(10, 5))
plt.plot(data["sequence_length"], data["train_memory"], marker="s", color="r", label="Training Memory (MB)")
plt.xlabel("Sequence Length")
plt.ylabel("Training Memory (MB)")
plt.title("Training Memory vs Sequence Length")
plt.legend()
plt.grid()
plt.savefig("training_memory_vs_sequence_length.png")
plt.show()
