# Gramba

## Requirements 
- Conda

## Setup
1) Clone the repository onto your machine (note that the conda environment was created on the U of T servers).
2) Create the virtual environment: ```conda env create -f environment.yml```
> **Note:** This may take several minutes.
3) Activate the environment: ```conda activate gramba```
4) Ensure the src directory is in your Python Path: ```export PYTHONPATH="$PYTHONPATH:/path/to/repository/src"```
> **Note:** This will only add the src directory to your path for your current session.  Write the export command to your .bashrc, .zshrc, or equivalent to add the path permanently.
5) Verify that test cases work: ```pytest```

## How to train and run the model
1) Download GloVe embeddings and unzip them in the twitter directory [link for glove](https://nlp.stanford.edu/data/glove.6B.zip)
2) Download the twitter dataset csv and put in in the twitter directory. Make sure it is named ```twitter.csv``` [link for twitter dataset](https://storage.googleapis.com/kaggle-data-sets/2477/4140/compressed/training.1600000.processed.noemoticon.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250226%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250226T184356Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=b48132a7a1da63ca57b8f8d3e971770b186b2ceb63e5fbe29b97747ffe0c90c1e964b1fb14c203901b3cbb2527c5bf730c56ad0ecca6a2da87397795dbbf3707222b82290112fc76c9dffcf8bc7190d22cd3465e71a83de016d25203d4281d5543e7da9e9f4ea315230be1338e058ec6fe30d56bc0ddb8e311f0aad2e0ec0bd3d6810ef1e39053bc033dceb3a8e275a1e11fec206fcbf4dd952e641fc4538a512c72ed03039e53a5f79e3a84d796f6407ce1c2b896098cbb1b5f347971328cdffac16f1daa9a9fea0feaab2fe695249bb8731c137a81189c61dafa3d86e448c27b385bda913680ee186699c37a177a80098d117de336d2327c4bd248caa40f1b)
3) Run ```load_twitter.py``` with the required embeddings dim (50, 100, 200 or 300) Command: ```python src/twitter/load_twitter.py --dataset_in_path src/twitter/twitter.csv --dataset_out_path src/twitter/twitter.pt```
4) Before running the file ```training.py```, make sure saving_train folder is empty if you don't want to start from a previous checkpoint. Select the right ```datatset```, ```vocab_size```, ```hidden_dim```(should be equal to the embeddings dim), ```batch_size``` and ```embedding_matrix```.