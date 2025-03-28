# Gramba
Gramba is a hybrid Transformer RNN deep learning architecture that interleaves minGRU layers with sub-quadratic attention layers.

## Model Architecture
![gramba](https://github.com/user-attachments/assets/9bd19851-3ed5-4232-8f52-5727244b4faf)



## Requirements 
- Conda

## Setup
1) Clone the repository onto your machine.
2) Create the virtual environment: ```conda env create -f environment.yml```
> **Note:** This may take several minutes.
3) Activate the environment: ```conda activate gramba```
4) Ensure the src directory is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository/src"```
> **Note:** This will only add the src directory to your path for your current session.  Write the export command to your .bashrc, .zshrc, or equivalent to add the path permanently.
5) Verify that test cases work: ```pytest```

## Codebase (src directory)

### `glove`
The `glove` directory contains the logic to load the GloVe embeddings into a numpy matrix to be later used as the embeddings for Gramba.

#### Steps to Prepare GloVe Embeddings:
1) Download the [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip) and unzip them into the `glove` directory.
2) Run the `load_glove.py` script to process the embeddings and save the resulting embedding matrix.

#### Usage:
```
python load_glove.py --embedding_dim <DIM> --glove_path <PATH_TO_GLOVE_TXT> --output_path <OUTPUT_FILE>
```

### `twitter`
The `twitter` directory contains the logic to load and process the Twitter dataset.

#### Steps to Prepare the Twitter Dataset:
1) Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it inside the `twitter` directory.
2) Run the `load_twitter.py` script to process the dataset and save the cleaned version.

#### Usage:
```bash
python load_imdb.py --dataset_in_path <INPUT_CSV> --dataset_out_path <OUTPUT_CSV>
```

### `imdb`
The `imdb` directory contains the logic to load and process the IMDB movie reviews dataset.

#### Steps to Prepare the IMDb Dataset:
1) Download the [IMDb dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it inside the `imdb` directory.
2) Run the `load_imdb.py` script to process the dataset and save the cleaned version.

#### Usage:
```bash
python load_imdb.py --dataset_in_path <INPUT_CSV> --dataset_out_path <OUTPUT_CSV>
```

### `squad`
Helper functions for loading the SQuAD dataset.

### `train`
Scripts to train models on various datasets. These scripts accept no arguments. The code needs to be adjusted to change the model's hyperparameters. These scripts should be ran on a machine with access to CUDA (e.g. U of T's SLURM Cluster). All of these scripts produce log files, which can be visualized by scripts in the `utils` directory.
1) `train_squad.py`: Train a Gramba model on SQuAD.
2) `bert_squad.py`: Train a BERT model on SQuAD
3) `train_classification`: Train a Gramba model either on Twitter or IMDB.

### `baseline`
Contains code to load an LSTM model.

### `utils`
Various utilities, including scripts which accept log files and produce visualizations.

### `profile`
Two scripts for profiling the Gramba model wirh respect to sequence length and model hyperparameters.

### `layers`
Building blocks of the Gramba model.

### `model`
Three Models: 
1) The base Gramba Model (`GrambaModel`)
2) Gramba for Sequence Classification (`GrambaForSequenceClassification`)
3) Gramba for Question Answering (`GrambaSQuADModel`).

These can be instantiated along with an instance of the `GrambaConfig` file, which specify the default hyperparameters of the Gramba model.

#### Usage
```bash
config = GrambaConfig(
    num_classes=2,
    embedding_dim=50,
    expansion_factor=2
    # Other hyperparameters
)

x = torch.randint(0, config.vocab_size, (32, 512))
attention_mask = torch.ones_like(x).bool()
longformer_mask = torch.zeros_like(x).bool()

model = GrambaModel(config)
output = model(x, attention_mask, longformer_mask)
```



