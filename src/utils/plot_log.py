import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def parse_log(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    hyperparams = []
    train_loss, val_loss, em, f1, times = [], [], [], [], []
    epochs = []
    epochs_em_f1 = []
    
    for line in lines:
        if 'Model configurations' in line:
            #merge the two following lines (line+1 and line+2) to get the hyperparameters
            text = line.strip() + "\n" + lines[lines.index(line)+1].strip() + "\n" + lines[lines.index(line)+2].strip()
            # Using regex to parse key-value pairs
            config = dict(re.findall(r"(\S+)=([\S]+)", text))

            # Convert numeric values to integers or floats where applicable
            for key, value in config.items():
                if value.isdigit():
                    config[key] = int(value)
                elif value.replace('.', '', 1).isdigit():  # Check for float
                    config[key] = float(value)
                elif value.lower() in ["true", "false"]:  # Convert boolean values
                    config[key] = value.lower() == "true"
            #drop gpu_name, vocab_size
            config.pop('gpu_name', None)
            config.pop('vocab_size', None)
            temp_hyperparams = [f"{key}: {value}" for key, value in config.items()]
            temp_hyperparams = "\n".join(temp_hyperparams)
            hyperparams.append(temp_hyperparams)

        
        match = re.search(r"Epoch\s+(\d+) \| train_loss: ([\d\.]+) \| val_loss: ([\d\.]+) .*? dt \(train\): ([\d\.]+) .*? EM: ([\d\.N/A]+) \| F1: ([\d\.N/A]+)", line)
        if match:
            epoch = int(match.group(1))
            train_loss.append(float(match.group(2)))
            val_loss.append(float(match.group(3)))
            times.append(float(match.group(4)))
            em_val = match.group(5)
            f1_val = match.group(6)
            
            if em_val != 'N/A':
                em.append(float(em_val))
                epochs_em_f1.append(epoch)
           
            if f1_val != 'N/A':
                f1.append(float(f1_val))
                
            epochs.append(epoch)
    
    return epochs, epochs_em_f1, train_loss, val_loss, em, f1, times, "\n".join(hyperparams)

def plot_graphs(log_files):
    plt.figure(figsize=(15, 10))
    
    # Colors for each log file
    colors = plt.cm.viridis(np.linspace(0, 1, len(log_files)))
    legend_labels = []

    
    for idx, file in enumerate(log_files):
        epochs, epochs_em_f1, train_loss, val_loss, em, f1, times, hyperparams = parse_log(file)
        color = colors[idx]
        label = f"{hyperparams}"
        legend_labels.append((color, label))
        
        # Plot losses
        plt.subplot(3, 1, 1)
        plt.tight_layout()
        plt.plot(epochs, train_loss, color=color, linestyle='-', label="Train")
        plt.plot(epochs, val_loss, color=color, linestyle='--', label="Val")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        
        # Plot EM and F1
        if any(not np.isnan(e) for e in em) or any(not np.isnan(f) for f in f1):
            plt.subplot(3, 1, 2)
            plt.tight_layout()
            plt.plot(epochs_em_f1, em, color=color, linestyle='-', label="EM")
            plt.plot(epochs_em_f1, f1, color=color, linestyle='--', label="F1")
            plt.xlabel("Epochs")
            plt.ylabel("Score")
            plt.title("Exact Match (EM) & F1 Score")
        
        # Plot training times
        plt.subplot(3, 1, 3)
        plt.tight_layout()
        plt.plot(epochs, times, color=color, label="Train Time")
        plt.xlabel("Epochs")
        plt.ylabel("Time (s)")
        plt.title("Training Time per Epoch")


    plt.subplot(3, 1, 1)
    plt.legend(handles=[plt.Line2D([0], [0], color='black', linestyle='-', label='Train'),
                        plt.Line2D([0], [0], color='black', linestyle='--', label='Val')],
                loc='upper right')
    plt.subplot(3, 1, 2)
    plt.legend(handles=[plt.Line2D([0], [0], color='black', linestyle='-', label='EM'),
                        plt.Line2D([0], [0], color='black', linestyle='--', label='F1')],
                loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    
    # Add global legend at the bottom
    plt.figlegend([plt.Line2D([0], [0], color=color, label=label) for color, label in legend_labels],
                  [label for color, label in legend_labels],
                  loc='center right',
                  ncol=1)
    
    plt.show()
    plt.subplots_adjust(bottom=0.05, top=0.95, right=0.75)
    plt.savefig("plot.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs='+', help="List of log files to process")
    args = parser.parse_args()
    plot_graphs(args.files)