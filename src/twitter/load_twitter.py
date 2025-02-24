import os 
import re
import nltk
import torch
import json
import argparse
import pandas as pd
from nltk.corpus import stopwords
from twitter.TwitterDataset import TwitterDataset
from nltk.stem import SnowballStemmer
import gensim
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_in_path', type=str,
                    help='CSV dataset path')
parser.add_argument('--dataset_out_path', type=str,
                    help='Path to save the processed dataset')

args = parser.parse_args()

if not args.dataset_out_path or not args.dataset_in_path:
    raise Exception('Please provide dataset paths')

if not os.path.exists(args.dataset_in_path):
    raise Exception('Dataset path does not exist')

nltk.download('stopwords')
nltk.download('punkt_tab')  # Download required dataset
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


data = pd.read_csv(args.dataset_in_path, encoding = 'latin',header=None)
data.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
data.head()
#remove the lignes with neutral sentiment
data = data[data['sentiment'] != 'neutral']
data['sentiment'] = data['sentiment'].map({0: 0, 4: 1}) #turning negative sentiment to 0 and positive to 1
data = data.drop(['id', 'date', 'query', 'user_id'], axis=1) #don't need these columns


def clean_text(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
              tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def tokenize_text(text):
    tokenized_sentences = word_tokenize(text)  # Split sentences into words
    return tokenized_sentences

# Apply function to the dataframe
data.text = data.text.apply(lambda x: clean_text(x))

#Tokenizing the text
print("Tokenizing the text")
tokenized_text = data.text.apply(lambda x: tokenize_text(x))
print(tokenized_text[0])

print("Initializing the Word2Vec model")
embedder = gensim.models.Word2Vec(tokenized_text, min_count=1, vector_size=50, window=5)
embedder.save('src/twitter/word2vec.model')

# Convert the tokenized text and store the embeddings
embedded_sentences = []
max_len = 0
for sentence in tqdm(tokenized_text):
    embedded_sentence = [embedder.wv[word] for word in sentence]
    embedded_sentences.append(embedded_sentence)
    max_len = max(max_len, len(embedded_sentence))

for i in tqdm(range(len(embedded_sentences))):
    curr_len = len(embedded_sentences[i])
    for j in range(curr_len, max_len):
        embedded_sentences[i].append([0]*50)
embedded_sentences = np.array([np.concatenate(row) for row in embedded_sentences])

print(embedded_sentences.shape)
print(embedded_sentences)

embedded_tweets = torch.tensor(embedded_sentences, dtype=torch.float)

twitter_dataset = TwitterDataset(embedded_tweets, data['sentiment'].tolist())

torch.save(twitter_dataset, args.dataset_out_path)